# Configuraciones mejoradas para mejor detección de posturas
class PostureConfig:
    MODEL_PATH = 'yolov8s-pose.pt'
    
    # Thresholds optimizados
    SITTING_THRESHOLD = 0.5  # Aumentado para mayor precisión
    CONFIDENCE_THRESHOLD = 0.6  # Aumentado para detecciones más confiables
    
    # Historial y suavizado temporal
    HISTORY_LENGTH = 10  # Reducido para respuesta más rápida
    MIN_CONFIDENCE_CHANGE = 0.6  # Aumentado para mayor estabilidad
    
    # Parámetros de alerta
    ALERT_DELAY_SECONDS = 3  # Reducido para respuesta más rápida
    
    # Configuración de video
    FRAME_SIZE = (1280, 720)
    TARGET_FPS = 30
    
    # Nuevos parámetros para validación
    MIN_KEYPOINTS_VISIBLE = 4  # Mínimo de keypoints críticos visibles
    MIN_PERSON_HEIGHT_RATIO = 0.15  # Mínimo 15% de la altura del frame
    MAX_PERSON_HEIGHT_RATIO = 0.9   # Máximo 90% de la altura del frame