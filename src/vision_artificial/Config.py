# Configuraciones mejoradas para mejor detección de posturas
class PostureConfig:
    MODEL_PATH = 'yolov8s-pose.pt'
    
    # Thresholds optimizados
    SITTING_THRESHOLD = 0.3  # Reducido para facilitar "sentado"
    CONFIDENCE_THRESHOLD = 0.5  # Más permisivo
    
    # Umbrales para ratios
    HEIGHT_RATIO_THRESHOLD = 0.45  # Relajado para niños sentados
    VERTICAL_RATIO_THRESHOLD = 0.1  # Tolerante para rodillas no visibles
    TORSO_LEG_RATIO_THRESHOLD = 1.2  # Menos estricto
    ASPECT_RATIO_THRESHOLD = 2.5
    TOTAL_HEIGHT_RATIO_THRESHOLD = 0.55  # Ajustado para niños
    RELATIVE_POSITION_THRESHOLD = 0.8
    STANDING_HEIGHT_THRESHOLD = 0.6  # Para detectar "parado"
    
    # Historial y suavizado
    HISTORY_LENGTH = 5  # Reducido para transiciones rápidas
    MIN_CONFIDENCE_CHANGE = 0.5  # Más sensible a cambios
    
    # Parámetros de alerta
    ALERT_DELAY_SECONDS = 3
    
    # Configuración de video
    FRAME_SIZE = (1280, 720)
    TARGET_FPS = 30
    
    # Parámetros de validación
    MIN_KEYPOINTS_VISIBLE = 2  # Solo hombros/caderas
    MIN_PERSON_HEIGHT_RATIO = 0.1
    MAX_PERSON_HEIGHT_RATIO = 0.9