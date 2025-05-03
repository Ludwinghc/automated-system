# Este archivo es el punto de partida principal de la app
from app import create_app

app = create_app()

if __name__ == '__main__':
  app.run(debug=True)