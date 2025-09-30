from asgiref.wsgi import WsgiToAsgi
from app import app as flask_app

# Expose ASGI-compatible app for Uvicorn
app = WsgiToAsgi(flask_app)


