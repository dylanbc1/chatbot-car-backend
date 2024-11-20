#!/bin/bash

# Activar el entorno virtual creado por Railway
source /opt/venv/bin/activate

# Instalar experta primero, lo que requiere frozendict==1.2
echo "Instalando experta..."
pip install experta

# Desinstalar la versión de frozendict instalada por experta
echo "Eliminando frozendict==1.2..."
pip uninstall -y frozendict

# Instalar la versión más reciente de frozendict requerida por yfinance
echo "Instalando frozendict>=2.3.4..."
pip install frozendict>=2.3.4

# Reinstalar yfinance para que sea compatible con la versión actual de frozendict
echo "Reinstalando yfinance..."
pip uninstall -y yfinance
pip install yfinance

# Instalar el resto de las dependencias
echo "Instalando dependencias del requirements.txt..."
pip install -r requirements.txt

# Mensaje final
echo "Instalación completada. Dependencias resueltas."
