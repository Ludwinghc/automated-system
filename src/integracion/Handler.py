# Modulo para integrar la vision artificial con google home
import time
from src.automatizacion_spotify.MusicPlayer import SpotifyPlayer
from src.automatizacion_spotify.GoogleHomeController import GoogleHomeController
from src.automatizacion_spotify.Config import MusicPlayerConfig

class Eventhandler:
  def __init__(self):
    print("Iniciando Handler")
    self.spotify_player = SpotifyPlayer()
    self.google_home_controller = GoogleHomeController()
    self.device_id = None
    self.is_playing = False
    self.last_alert_time = None
    self.initialize_device()
  
  def initialize_device(self):
    device_name = self.google_home_controller.get_device_name()
    self.device_id =  self.spotify_player.get_device_id(device_name)
    if not self.device_id:
      raise RuntimeError("No se pudo obtener el ID del dispositivo Spotify")
  
  def handler_alert(self, alert_active):
    current_time = time.time()
    if alert_active and not self.is_playing:
      print("Alerta activada, reproduciendo musica")
      try:
        self.spotify_player.play_track(
          self.device_id,
          MusicPlayerConfig.TRACK_URI
        )
        self.is_playing = True
        self.last_alert_time = None
      except Exception as e:
        print(f"Error al reproducir al musica: {e}")

    elif not alert_active and self.is_playing:
      if self.last_alert_time is None:
        self.last_alert_time = current_time
        print("Alerta desactivada, pausando música...")
      elapsed = current_time - self.last_alert_time
      countdown = max(0, MusicPlayerConfig.PAUSE_DELAY - int(elapsed))
      if elapsed >= MusicPlayerConfig.PAUSE_DELAY:
        try:
          self.spotify_player.sp.pause_playback(device_id=self.device_id)
          self.is_playing = False
          self.last_alert_time = None
        except Exception as e:
          print(f"Error al pausar música: {e}")
    elif alert_active and self.is_playing:
      self.last_alert_time = None