# Modulo para manejar la logica de spotify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
from.Config import MusicPlayerConfig

load_dotenv()

class SpotifyPlayer:
  def __init__(self):
    print("Iniciando spotify player")
    try:
      self.sp =  spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope=MusicPlayerConfig.SPOTIFY_SCOPE
      ))
    except Exception as e:
      print(f"Erro al iniciar spotify: {e}")
      raise
  
  def get_device_id(self, device_name):
    devices = self.sp.devices()
    for device in devices['devices']:
      print(f"- {device['name']} (id: {device['id']})")
      if device['name'] == device_name:
        return device['id']
    print(f"No se encontro el dispositivo spotify: {device_name}")
    return None
  
  def play_track(self, device_id, track_uri):
    try:
      self.sp.start_playback(device_id=device_id, uris=[track_uri])
    except Exception as e:
      print(f"Error al reproducir la pista: {e}")
      raise