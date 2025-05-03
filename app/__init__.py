# Crea y configura la instancia de la app de flask

from flask import Flask
from app.views.mainRoutes import main

def create_app():
  app = Flask(__name__)
  app.register_blueprint(main)
  return app