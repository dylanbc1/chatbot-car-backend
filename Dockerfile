# Utiliza una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . /app/

# Instala las dependencias de sistema necesarias (gcc es comúnmente necesario para compilación de paquetes)
RUN apt-get update && apt-get install -y gcc

# Crea el entorno virtual
RUN python3 -m venv /opt/venv

# Activa el entorno virtual e instala las dependencias
RUN /opt/venv/bin/pip install --upgrade pip
RUN /opt/venv/bin/pip install -r requirements.txt

# Ejecuta el script de configuración personalizado (si es necesario)
RUN /opt/venv/bin/bash ./setup_reqs.sh

# Exponer el puerto 8000 para FastAPI (por defecto usa este puerto)
EXPOSE 8000

# Define el comando para ejecutar la API con uvicorn
CMD ["/opt/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
