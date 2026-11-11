# Imagen local del prototipo. No se probó: en la máquina de desarrollo no hay Docker.
# Construir desde la raíz del repositorio:  docker build -f prototipo/Dockerfile -t mia14-prototipo .
# Ejecutar solo en esta máquina:            docker run -p 127.0.0.1:8501:8501 mia14-prototipo
# O con Compose:                            docker compose -f prototipo/compose.yaml up --build
# Dentro del contenedor la aplicación atiende en 0.0.0.0, por lo que el puerto se publica siempre con 127.0.0.1
# (nunca con -p 8501:8501 ni con -P, que lo abrirían a la red).
# El contexto es una lista blanca (.dockerignore de la raíz): no entran data/ con MIMIC-IV, modelos ni el .env.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PROTOTIPO_DATOS=/datos PROTOTIPO_CONTENEDOR=1
WORKDIR /app
COPY prototipo/requirements.txt prototipo/requirements.txt
RUN pip install --no-cache-dir -r prototipo/requirements.txt
COPY src/ src/
COPY config/config.yaml config/config.yaml
COPY data/catalogos/ data/catalogos/
COPY prototipo/ prototipo/

RUN useradd --create-home prototipo && mkdir -p /datos && chown prototipo /datos
USER prototipo
EXPOSE 8501
# Al primer arranque entrena el modelo de demostración si aún no está en /datos.
CMD ["sh", "-c", "test -f /datos/modelo_demo.joblib || python -m prototipo.modelo_demo; exec streamlit run prototipo/app.py --server.address=0.0.0.0"]
