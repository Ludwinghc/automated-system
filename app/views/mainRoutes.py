from flask import Blueprint, render_template
from app.models.saludo import obtener_saludo

main = Blueprint('main', __name__)

@main.route('/')
def home():
  mensaje = obtener_saludo()
  return render_template('pages/home.html', mensaje=mensaje)