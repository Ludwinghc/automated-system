# Configuraciones necesarias para la visión artificial, como parámetros y dimensiones
class PostureConfig:
  MODEL_PATH = 'yolov8s-pose.pt'
  SITTING_THRESHOLD = 0.6
  CONFIDENCE_THRESHOLD = 0.7
  HISTORY_LENGTH = 15
  MIN_CONFIDENCE_CHANGE = 0.6
  ALERT_DELAY_SECONDS = 5
  FRAME_SIZE = (1280,720)
  TARGET_FPS = 30
