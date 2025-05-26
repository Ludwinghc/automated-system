# Encapsular la logica de la deteccion de las posturas de las personas - VERSION MEJORADA
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import warnings
from .Config import PostureConfig

class PostureDetector:
    def __init__(self):
        try:
            self.model = YOLO(PostureConfig.MODEL_PATH)
        except Exception as e:
            print(f"Error al cargar el modelo YOLO: {e}")
            raise
        
        # Umbrales más específicos
        self.sitting_threshold = 0.5  # Aumentado para mayor precisión
        self.confidence_threshold = PostureConfig.CONFIDENCE_THRESHOLD
        self.posture_history = defaultdict(lambda: deque(maxlen=PostureConfig.HISTORY_LENGTH))
        self.min_confidence_change = PostureConfig.MIN_CONFIDENCE_CHANGE
        
        # Nuevos parámetros para mejor detección
        self.min_keypoints_required = 8  # Mínimo de keypoints visibles
        self.angle_tolerance = 15  # Tolerancia para ángulos en grados

    def safe_divide(self, a, b, default=1.0):
        return a / b if abs(b) > 1e-6 else default

    def calculate_angle(self, p1, p2, p3):
        """Calcula el ángulo entre tres puntos"""
        try:
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            
            return angle
        except:
            return 0

    def validate_keypoints(self, kpts):
        """Valida si hay suficientes keypoints para hacer una predicción confiable"""
        valid_points = 0
        required_joints = [5, 6, 11, 12, 13, 14]  # Hombros, caderas, rodillas
        
        for joint_idx in required_joints:
            if joint_idx < len(kpts) and not np.isnan(kpts[joint_idx][0]) and not np.isnan(kpts[joint_idx][1]):
                valid_points += 1
        
        return valid_points >= 4  # Al menos 4 de los 6 puntos críticos

    def calculate_posture(self, kpts, frame_height, frame_width):
        try:
            # Índices de keypoints YOLO pose
            LEFT_HIP, RIGHT_HIP = 11, 12
            LEFT_KNEE, RIGHT_KNEE = 13, 14
            LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
            LEFT_ANKLE, RIGHT_ANKLE = 15, 16
            LEFT_ELBOW, RIGHT_ELBOW = 7, 8

            # Validar keypoints
            if not self.validate_keypoints(kpts):
                return "desconocido"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Calcular promedios de articulaciones
                hip_avg_x = np.nanmean([kpts[LEFT_HIP][0], kpts[RIGHT_HIP][0]])
                hip_avg_y = np.nanmean([kpts[LEFT_HIP][1], kpts[RIGHT_HIP][1]])
                
                knee_avg_x = np.nanmean([kpts[LEFT_KNEE][0], kpts[RIGHT_KNEE][0]])
                knee_avg_y = np.nanmean([kpts[LEFT_KNEE][1], kpts[RIGHT_KNEE][1]])
                
                shoulder_avg_x = np.nanmean([kpts[LEFT_SHOULDER][0], kpts[RIGHT_SHOULDER][0]])
                shoulder_avg_y = np.nanmean([kpts[LEFT_SHOULDER][1], kpts[RIGHT_SHOULDER][1]])

            sitting_score = 0
            confidence_factors = []

            # CRITERIO 1: Análisis de la altura total y proporción corporal
            person_height = abs(knee_avg_y - shoulder_avg_y)
            if person_height > 0:
                height_ratio = person_height / frame_height
                
                # Personas sentadas tienden a ocupar menos altura vertical
                if height_ratio < 0.35:
                    sitting_score += 0.8
                    confidence_factors.append("altura_baja")
                elif height_ratio > 0.55:
                    sitting_score -= 0.6
                    confidence_factors.append("altura_alta")

            # CRITERIO 2: Análisis del ángulo de las piernas (mejorado)
            try:
                # Ángulo cadera-rodilla
                if not any(np.isnan([hip_avg_x, hip_avg_y, knee_avg_x, knee_avg_y])):
                    # Verificar si las rodillas están al nivel o por encima de las caderas
                    vertical_diff = knee_avg_y - hip_avg_y
                    vertical_ratio = vertical_diff / frame_height
                    
                    # Si las rodillas están muy cerca o por encima de las caderas = sentado
                    if vertical_ratio < 0.05:  # Rodillas al nivel o arriba de caderas
                        sitting_score += 0.7
                        confidence_factors.append("rodillas_nivel_caderas")
                    elif vertical_ratio > 0.15:  # Rodillas claramente abajo = parado
                        sitting_score -= 0.5
                        confidence_factors.append("rodillas_bajo_caderas")
                        
            except Exception as e:
                pass

            # CRITERIO 3: Análisis de la posición del torso
            torso_height = abs(shoulder_avg_y - hip_avg_y)
            leg_segment = abs(hip_avg_y - knee_avg_y)
            
            if torso_height > 0 and leg_segment > 0:
                torso_leg_ratio = torso_height / leg_segment
                
                # Cuando está sentado, el torso es más prominente
                if torso_leg_ratio > 2.0:
                    sitting_score += 0.5
                    confidence_factors.append("torso_prominente")
                elif torso_leg_ratio < 1.2:
                    sitting_score -= 0.3
                    confidence_factors.append("proporcion_equilibrada")

            # CRITERIO 4: Análisis de la compacidad corporal mejorado
            body_width = 0
            try:
                shoulder_width = abs(kpts[LEFT_SHOULDER][0] - kpts[RIGHT_SHOULDER][0])
                hip_width = abs(kpts[LEFT_HIP][0] - kpts[RIGHT_HIP][0])
                body_width = max(shoulder_width, hip_width)
            except:
                pass

            if body_width > 0 and person_height > 0:
                aspect_ratio = person_height / body_width
                
                # Personas sentadas tienden a tener un aspect ratio menor (más anchas que altas)
                if aspect_ratio < 2.0:
                    sitting_score += 0.4
                    confidence_factors.append("cuerpo_compacto")
                elif aspect_ratio > 3.5:
                    sitting_score -= 0.3
                    confidence_factors.append("cuerpo_alargado")

            # CRITERIO 5: Análisis de la posición de los tobillos (si están disponibles)
            ankle_visible = False
            try:
                if (LEFT_ANKLE < len(kpts) and RIGHT_ANKLE < len(kpts) and
                    not np.isnan(kpts[LEFT_ANKLE][1]) and not np.isnan(kpts[RIGHT_ANKLE][1])):
                    
                    ankle_avg_y = np.nanmean([kpts[LEFT_ANKLE][1], kpts[RIGHT_ANKLE][1]])
                    ankle_visible = True
                    
                    # Distancia total del cuerpo
                    total_height = ankle_avg_y - shoulder_avg_y
                    if total_height > 0:
                        total_height_ratio = total_height / frame_height
                        
                        # Si la persona completa ocupa poca altura = sentado
                        if total_height_ratio < 0.45:
                            sitting_score += 0.6
                            confidence_factors.append("persona_completa_baja")
                        elif total_height_ratio > 0.7:
                            sitting_score -= 0.4
                            confidence_factors.append("persona_completa_alta")
            except:
                pass

            # CRITERIO 6: Análisis de la posición relativa en el frame
            person_bottom = max(knee_avg_y, hip_avg_y)
            relative_position = person_bottom / frame_height
            
            # Personas sentadas suelen aparecer en la parte media-alta del frame
            if relative_position < 0.7:
                sitting_score += 0.2
                confidence_factors.append("posicion_media_frame")

            # Normalizar el score y aplicar threshold más inteligente
            max_possible_score = 3.2  # Suma de todos los scores positivos máximos
            normalized_score = sitting_score / max_possible_score
            
            # Threshold adaptativo basado en confianza
            threshold = self.sitting_threshold
            if len(confidence_factors) < 2:
                threshold += 0.1  # Ser más conservador si hay poca información
            
            posture = "sentado" if normalized_score >= threshold else "parado"
            
            # Debug información (opcional)
            # print(f"Score: {normalized_score:.2f}, Factores: {confidence_factors}, Resultado: {posture}")
            
            return posture
            
        except Exception as e:
            print(f"Error al calcular postura: {e}")
            return "desconocido"

    def detect_postures(self, frame):
        try:
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            posture_counts = {"parado": 0, "sentado": 0, "desconocido": 0}

            if not results or not results[0].boxes or not results[0].keypoints:
                return posture_counts, frame

            for i, (box, keypoints) in enumerate(zip(results[0].boxes, results[0].keypoints)):
                if box.conf < self.confidence_threshold:
                    continue

                if keypoints.xy is None or len(keypoints.xy) == 0:
                    continue

                kpts = keypoints.xy[0].cpu().numpy()
                current_posture = self.calculate_posture(kpts, frame.shape[0], frame.shape[1])

                # Filtrar posturas desconocidas del historial
                if current_posture != "desconocido":
                    self.posture_history[i].append(current_posture)

                # Aplicar suavizado temporal mejorado
                if len(self.posture_history[i]) >= 3:  # Reducido para respuesta más rápida
                    recent_postures = list(self.posture_history[i])[-5:]  # Últimas 5 detecciones
                    counts = {
                        "parado": recent_postures.count("parado"),
                        "sentado": recent_postures.count("sentado")
                    }
                    
                    if counts["parado"] + counts["sentado"] > 0:
                        confidence_ratio = max(counts.values()) / len(recent_postures)
                        if confidence_ratio >= 0.6:  # 60% de consistencia
                            posture = max(counts, key=counts.get)
                        else:
                            posture = current_posture
                    else:
                        posture = current_posture
                else:
                    posture = current_posture

                # Solo contar posturas conocidas
                if posture in posture_counts:
                    posture_counts[posture] += 1

                # Visualización mejorada
                if posture != "desconocido":
                    color = (0, 0, 255) if posture == "sentado" else (0, 255, 0)
                    x, y = int(box.xyxy[0][0]), int(box.xyxy[0][1])
                    
                    # Mostrar confianza de la detección
                    confidence_text = f"{posture.upper()} ({box.conf[0]:.2f})"
                    cv2.putText(frame, confidence_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Dibujar bounding box
                    cv2.rectangle(frame, 
                                (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                                color, 2)

            return posture_counts, frame
            
        except Exception as e:
            print(f"Error en detect_postures: {e}")
            return {"parado": 0, "sentado": 0, "desconocido": 0}, frame