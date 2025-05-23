# Manejo de la captura del video y visualización de resultados
import cv2
import time
from .PostureDetector import PostureDetector
from. Config import PostureConfig
from src.integracion.Handler import Eventhandler

class VideoProcessor:
  def __init__(self):
    self.detector = PostureDetector()
    self.event_handler = Eventhandler()
    self.cap = cv2.VideoCapture(0)
    if not self.cap.isOpened():
      print("Error: No se pudo acceder a la camara")
      raise RuntimeError("Error: No se pudo acceder a la cámara.")
    # Verificacion de captura inicial
    success, _ = self.cap.read()
    if not success:
      print("Advertencia la camara se abrio pero no puede capturar frames")
    cv2.namedWindow('Deteccion de postura', cv2.WINDOW_NORMAL)
    self.last_standing_time = None
    self.alert_shown = False

  def process_frame(self):
    start_time = time.time()
    success, frame = self.cap.read()
    if not success:
      print("Error: al capturar el frame")
      return None, None, False, 1
    try:
      frame = cv2.resize(frame, PostureConfig.FRAME_SIZE)
    except Exception as e:
      print(f"Error al redimensionar el frame: {e}")
      return None, None, False, 1
    try:
      posture_counts, output_frame = self.detector.detect_postures(frame)
    except Exception as e:
      print(f"Error en detect_postures: {e}")
    total_people = posture_counts["parado"] + posture_counts["sentado"]

    stats_text =  f"Total: {total_people} | Paradas: {posture_counts['parado']} | Sentadas: {posture_counts['sentado']}"
    cv2.putText(output_frame, stats_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    current_time = time.time()
    alert_active = False

    if total_people > 0 and posture_counts['parado'] > total_people / 2:
      if self.last_standing_time is None:
        self.last_standing_time = current_time
      
      elapsed = current_time - self.last_standing_time
      countdown = max(0, PostureConfig.ALERT_DELAY_SECONDS - int(elapsed))
      cv2.putText(output_frame, f"Alerta en: {countdown} segundos", (20, 100),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
      if elapsed >= PostureConfig.ALERT_DELAY_SECONDS:
        alert_text = "Alerta! Los estudiantes se deben sentar!!"
        cv2.rectangle(output_frame, (10,130), (600,180), (0, 0, 255), -1)
        cv2.putText(output_frame, alert_text, (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 , (255,255,255), 2)
        alert_active = True
    else:
      if self.last_standing_time is not None:
        self.last_standing_time = None
      self.alert_shown = False
    
    self.event_handler.handler_alert(alert_active)
    processing_time = time.time() - start_time
    wait_time = max(1, int((1 / PostureConfig.TARGET_FPS - processing_time) * 1000))
    
    return output_frame, posture_counts, alert_active, wait_time
  
  def run(self):
    alert_active = False
    try:
      while(self.cap.isOpened()):
        frame, posture_counts, alert_active, wait_time = self.process_frame()
        if frame is None:
          print("Frame nulo, saliendo del bucle")
          break
        try:
          cv2.imshow('Detección de Postura', frame)
        except Exception as e:
          print(f"Error en cv2.imshow: {e}")
          break
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
          print("Tecla 'q' presionada, saliendo...")
          break
    finally:
      self.cap.release()
      cv2.destroyAllWindows()
      return alert_active
