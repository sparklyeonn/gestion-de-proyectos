# Uso del paquete
Python de entrenamiento: 3.12.10
Desde esta carpeta, crear/activar un entorno con esa versión de Python.
Instalar: python -m pip install -r requirements.txt
Cargar solamente archivos joblib propios o de confianza.
Ejecutar: python predecir.py --entrada entrada_dashboard_demo.csv --modelo modelo_20261111T010106_22ef1d43.joblib --salida lotes --origen demo_submuestra_test_sintetico
El CSV actualizado queda en lotes/dashboard/predicciones_actuales.csv.
No necesita Kaggle ni reentrenamiento para esta predicción.
No mezclar este modelo con resultados de otra carpeta de ejecución.
Los CSV de demostración contienen datos sintéticos; no son evidencia de eficacia escolar.
