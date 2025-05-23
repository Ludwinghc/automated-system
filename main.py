from src.vision_artificial.VideoProcessor import VideoProcessor

if __name__ == "__main__":
    print("Iniciando Sistema automatizado con Visión artificial y Google home")
    try:
        processor = VideoProcessor()
        processor.run()
        print("Programa finalizado normalmente")
    except Exception as e:
        print(f"Error inesperado: {e}")