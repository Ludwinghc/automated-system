# Modulo para manejar la conexion con el google home nest
import pychromecast
from .Config import MusicPlayerConfig

class GoogleHomeController:
  def __init__(self):
    print("Iniciando el GoogleHome controller")
    try:
      print(f"Buscando dispositivo Chromecast: {MusicPlayerConfig.GOOGLE_HOME_NAME}")
      chromecasts, browser = pychromecast.discovery.discover_chromecasts()
      chromecasts, _ = pychromecast.get_listed_chromecasts(
        friendly_names=[MusicPlayerConfig.GOOGLE_HOME_NAME]
      )
      if not chromecasts:
        print(f"No se encontró el dispositivo {MusicPlayerConfig.GOOGLE_HOME_NAME}")
        raise RuntimeError("Dispositivo Chromecast no encontrado")
      self.device = chromecasts[0]
      self.device.wait()
      print(f"Dispositivo Chromecast encontrado: {self.device.name}")
    except Exception as e:
      print(f"Error al iniciar el google home: {e}")
      raise
  
  def get_device_name(self):
    return self.device.name
