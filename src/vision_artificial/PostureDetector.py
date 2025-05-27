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
            print("Modelo YOLO cargado correctamente")
        except Exception as e:
            print(f"Error al cargar el modelo YOLO: {e}")
            raise
        
        self.sitting_threshold = PostureConfig.SITTING_THRESHOLD
        self.confidence_threshold = PostureConfig.CONFIDENCE_THRESHOLD
        self.posture_history = defaultdict(lambda: deque(maxlen=PostureConfig.HISTORY_LENGTH))
        self.min_confidence_change = PostureConfig.MIN_CONFIDENCE_CHANGE
        self.min_keypoints_required = PostureConfig.MIN_KEYPOINTS_VISIBLE
        self.angle_tolerance = 15
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
        required_joints = [5, 6, 11, 12]  # Solo hombros y caderas
        
        for joint_idx in required_joints:
            if joint_idx < len(kpts) and kpts[joint_idx][0] != 0 and kpts[joint_idx][1] != 0:
                valid_points += 1
        
        print(f"Keypoints válidos: {valid_points}")
        return valid_points >= self.min_keypoints_required

    def calculate_posture(self, kpts, frame_height, frame_width):
        try:
            LEFT_HIP, RIGHT_HIP = 11, 12
            LEFT_KNEE, RIGHT_KNEE = 13, 14
            LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
            LEFT_ANKLE, RIGHT_ANKLE = 15, 16

            if not self.validate_keypoints(kpts):
                print("Detección descartada: insuficientes keypoints")
                return "desconocido"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hip_avg_x = np.nanmean([kpts[LEFT_HIP][0], kpts[RIGHT_HIP][0]])
                hip_avg_y = np.nanmean([kpts[LEFT_HIP][1], kpts[RIGHT_HIP][1]])
                knee_avg_x = np.nanmean([kpts[LEFT_KNEE][0], kpts[RIGHT_KNEE][0]])
                knee_avg_y = np.nanmean([kpts[LEFT_KNEE][1], kpts[RIGHT_KNEE][1]])
                shoulder_avg_x = np.nanmean([kpts[LEFT_SHOULDER][0], kpts[RIGHT_SHOULDER][0]])
                shoulder_avg_y = np.nanmean([kpts[LEFT_SHOULDER][1], kpts[RIGHT_SHOULDER][1]])

            sitting_score = 0
            confidence_factors = []

            # CRITERIO 1: Altura total
            person_height = abs((knee_avg_y if not np.isnan(knee_avg_y) else hip_avg_y) - shoulder_avg_y)
            if person_height > 0:
                height_ratio = person_height / frame_height
                if height_ratio < PostureConfig.HEIGHT_RATIO_THRESHOLD:
                    sitting_score += 0.9
                    confidence_factors.append(f"altura_baja: {height_ratio:.2f}")
                elif height_ratio > PostureConfig.STANDING_HEIGHT_THRESHOLD:
                    sitting_score -= 0.7
                    confidence_factors.append(f"altura_alta: {height_ratio:.2f}")
                print(f"Height ratio: {height_ratio:.2f}")
            else:
                print("Altura inválida, omitiendo criterio 1")

            # CRITERIO 2: Posición de rodillas (opcional)
            valid_knees = not any(np.isnan([hip_avg_x, hip_avg_y, knee_avg_x, knee_avg_y]))
            if valid_knees:
                vertical_diff = knee_avg_y - hip_avg_y
                vertical_ratio = vertical_diff / frame_height
                if vertical_ratio < PostureConfig.VERTICAL_RATIO_THRESHOLD:
                    sitting_score += 0.7
                    confidence_factors.append(f"rodillas_nivel_caderas: {vertical_ratio:.2f}")
                elif vertical_ratio > 0.15:
                    sitting_score -= 0.6
                    confidence_factors.append(f"rodillas_bajo_caderas: {vertical_ratio:.2f}")
                print(f"Vertical ratio: {vertical_ratio:.2f}")
            else:
                print("Rodillas no detectadas, omitiendo criterio 2")

            # CRITERIO 3: Relación torso-piernas
            torso_height = abs(shoulder_avg_y - hip_avg_y)
            leg_segment = abs(hip_avg_y - knee_avg_y)
            if torso_height > 0 and leg_segment > 0:
                torso_leg_ratio = torso_height / leg_segment
                if torso_leg_ratio > PostureConfig.TORSO_LEG_RATIO_THRESHOLD:
                    sitting_score += 0.8
                    confidence_factors.append(f"torso_prominente: {torso_leg_ratio:.2f}")
                print(f"Torso-leg ratio: {torso_leg_ratio:.2f}")
            else:
                # Si no hay piernas, asumir torso prominente para sentado
                sitting_score += 0.6
                confidence_factors.append("sin_piernas_torso_prominente")
                print("Piernas no detectadas, asumiendo torso prominente")

            # CRITERIO 4: Compacidad corporal
            body_width = 0
            try:
                shoulder_width = abs(kpts[LEFT_SHOULDER][0] - kpts[RIGHT_SHOULDER][0])
                hip_width = abs(kpts[LEFT_HIP][0] - kpts[RIGHT_HIP][0])
                body_width = max(shoulder_width, hip_width)
            except:
                pass
            if body_width > 0 and person_height > 0:
                aspect_ratio = person_height / body_width
                if aspect_ratio < PostureConfig.ASPECT_RATIO_THRESHOLD:
                    sitting_score += 0.5
                    confidence_factors.append(f"cuerpo_compacto: {aspect_ratio:.2f}")
                print(f"Aspect ratio: {aspect_ratio:.2f}")

            # CRITERIO 5: Altura total con tobillos
            try:
                if (LEFT_ANKLE < len(kpts) and RIGHT_ANKLE < len(kpts) and
                    kpts[LEFT_ANKLE][1] != 0 and kpts[RIGHT_ANKLE][1] != 0):
                    ankle_avg_y = np.nanmean([kpts[LEFT_ANKLE][1], kpts[RIGHT_ANKLE][1]])
                    total_height = ankle_avg_y - shoulder_avg_y
                    if total_height > 0:
                        total_height_ratio = total_height / frame_height
                        if total_height_ratio < PostureConfig.TOTAL_HEIGHT_RATIO_THRESHOLD:
                            sitting_score += 0.6
                            confidence_factors.append(f"persona_completa_baja: {total_height_ratio:.2f}")
                        print(f"Total height ratio: {total_height_ratio:.2f}")
            except:
                pass

            # CRITERIO 6: Posición en el frame
            person_bottom = max((knee_avg_y if not np.isnan(knee_avg_y) else hip_avg_y), hip_avg_y)
            if not np.isnan(person_bottom) and frame_height > 0:
                relative_position = person_bottom / frame_height
                if relative_position < PostureConfig.RELATIVE_POSITION_THRESHOLD:
                    sitting_score += 0.4
                    confidence_factors.append(f"posicion_media_frame: {relative_position:.2f}")
                print(f"Relative position: {relative_position:.2f}")
            else:
                print("Posición inválida, omitiendo criterio 6")

            # Normalizar score
            max_possible_score = 3.9  # Ajustado por nuevos pesos
            normalized_score = max(min(sitting_score / max_possible_score, 1.0), 0.0)
            threshold = self.sitting_threshold
            if len(confidence_factors) < 2:
                threshold += 0.1

            posture = "sentado" if normalized_score >= threshold else "parado"
            print(f"Score: {normalized_score:.2f}, Threshold: {threshold:.2f}, Factores: {confidence_factors}, Postura: {posture}")

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