#!/bin/bash

# Instalar experta, lo cual instala frozendict==1.2
pip install experta

# Desinstalar la versión incompatible de frozendict
pip uninstall -y frozendict

# Instalar la versión actualizada de frozendict requerida por yfinance
pip install --upgrade frozendict

# Desinstalar yfinance para evitar conflictos
pip uninstall -y yfinance

# Reinstalar yfinance para que funcione con la versión correcta de frozendict
pip install yfinance

# Instalar el resto de las dependencias del proyecto
pip install -r requirements.txt
