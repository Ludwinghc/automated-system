# Encapsular la logica de la deteccion de las posturas de las personas
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
        self.sitting_threshold = PostureConfig.SITTING_THRESHOLD
        self.confidence_threshold = PostureConfig.CONFIDENCE_THRESHOLD
        self.posture_history = defaultdict(lambda: deque(maxlen=PostureConfig.HISTORY_LENGTH))
        self.min_confidence_change = PostureConfig.MIN_CONFIDENCE_CHANGE

    def safe_divide(self, a, b, default=1.0):
        return a / b if abs(b) > 1e-6 else default

    def calculate_posture(self, kpts, frame_height):
        try:
            LEFT_HIP, RIGHT_HIP = 11, 12
            LEFT_KNEE, RIGHT_KNEE = 13, 14
            LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hip_avg = np.nanmean([kpts[LEFT_HIP][1], kpts[RIGHT_HIP][1]])
                knee_avg = np.nanmean([kpts[LEFT_KNEE][1], kpts[RIGHT_KNEE][1]])
                shoulder_avg = np.nanmean([kpts[LEFT_SHOULDER][1], kpts[RIGHT_SHOULDER][1]])

            sitting_score = 0
            leg_length = abs(hip_avg - knee_avg)
            if leg_length > 10:
                hip_knee_ratio = leg_length / frame_height
                if hip_knee_ratio < 0.2:
                    sitting_score += 0.5
            else:
                sitting_score += 0.3

            torso_height = abs(shoulder_avg - hip_avg)
            if torso_height > frame_height * 0.2:
                sitting_score += 0.3

            if leg_length > 10:
                torso_leg_ratio = self.safe_divide(torso_height, leg_length, 1.5)
                if torso_leg_ratio > 1.0:
                    sitting_score += 0.3

            posture = "sentado" if sitting_score >= PostureConfig.SITTING_THRESHOLD else "parado"
            return posture
        except Exception as e:
            print(f"Error al calcular postura: {e}")
            return "parado"

    def detect_postures(self, frame):
        try:
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            posture_counts = {"parado": 0, "sentado": 0}

            # Verificar si hay detecciones válidas
            if not results or not results[0].boxes or not results[0].keypoints:
                # print("No se detectaron personas en el frame")
                return posture_counts, frame

            for i, (box, keypoints) in enumerate(zip(results[0].boxes, results[0].keypoints)):
                if box.conf < self.confidence_threshold:
                    # print(f"Detección {i} descartada por baja confianza: {box.conf}")
                    continue

                # Verificar que keypoints sea válido
                if keypoints.xy is None or len(keypoints.xy) == 0:
                    # print(f"Detección {i} no tiene keypoints válidos")
                    continue

                kpts = keypoints.xy[0].cpu().numpy()
                current_posture = self.calculate_posture(kpts, frame.shape[0])

                self.posture_history[i].append(current_posture)

                if len(self.posture_history[i]) == self.posture_history[i].maxlen:
                    counts = {
                        "parado": list(self.posture_history[i]).count("parado"),
                        "sentado": list(self.posture_history[i]).count("sentado")
                    }
                    final_posture = max(counts, key=counts.get)
                    if counts[final_posture] / len(self.posture_history[i]) >= self.min_confidence_change:
                        posture = final_posture
                    else:
                        posture = current_posture
                else:
                    posture = current_posture

                posture_counts[posture] += 1

                color = (0, 0, 255) if posture == "sentado" else (0, 255, 0)
                x, y = int(box.xyxy[0][0]), int(box.xyxy[0][1])
                cv2.putText(frame, f"{posture.upper()}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            return posture_counts, frame
        except Exception as e:
            print(f"Error en detect_postures: {e}")
            return {"parado": 0, "sentado": 0}, frame