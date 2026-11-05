"""Inferencia independiente. Cargar únicamente modelos joblib de confianza."""
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import argparse, hashlib, json, platform
import numpy as np
import pandas as pd
import sklearn
import joblib

def validar(datos, modelo):
    features = modelo['features']
    faltan = set(features) - set(datos.columns)
    if faltan:
        raise ValueError(f'Faltan columnas: {sorted(faltan)}')
    if 'ID_Caso' not in datos:
        raise ValueError('Falta ID_Caso')
    ids = datos['ID_Caso'].astype('string').str.strip()
    if ids.isna().any() or ids.eq('').any() or ids.duplicated().any():
        raise ValueError('ID_Caso debe ser único, no vacío y no nulo por lote')
    x = datos[features].apply(pd.to_numeric, errors='raise')
    if x.empty or x.isna().all(axis=1).any() or np.isinf(x.to_numpy()).any():
        raise ValueError('Lote vacío, fila sin predictores o valores infinitos')
    for c, (lo, hi) in modelo['limites'].items():
        if not x[c].dropna().between(lo, hi).all():
            raise ValueError(f'{c} fuera del dominio del prototipo [{lo}, {hi}]. Revisar; no truncar automáticamente.')
    for c in modelo['binarias']:
        if not x[c].dropna().isin([0, 1]).all():
            raise ValueError(f'{c} debe ser 0 o 1')
    for c in ['Grade', 'FreeTime', 'GoOut']:
        if not (x[c].dropna() % 1 == 0).all():
            raise ValueError(f'{c} debe ser entero')
    return x, ids

def procesar_lote(ruta_csv, ruta_modelo, destino, origen='demostracion_sintetica'):
    m = joblib.load(ruta_modelo)
    actuales = {'python':platform.python_version(), 'numpy':np.__version__,
                'pandas':pd.__version__, 'sklearn':sklearn.__version__, 'joblib':joblib.__version__}
    for clave, version in m['versiones'].items():
        if clave in actuales and actuales[clave] != version:
            raise ValueError(f'Entorno incompatible: {clave} requiere {version}, instalado {actuales[clave]}')
    datos = pd.read_csv(ruta_csv, dtype={'ID_Caso':'string'})
    x, ids = validar(datos, m)
    prob = m['pipeline'].predict_proba(x)[:, 1]
    if not np.isfinite(prob).all() or not ((prob >= 0) & (prob <= 1)).all():
        raise ValueError('Probabilidades inválidas')
    alerta = (prob >= m['umbral']).astype(int)
    fecha = datetime.now(timezone.utc).isoformat()
    lote = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8]
    salida = pd.DataFrame({'ID_Caso':ids, 'Grade':x['Grade'],
        'Probabilidad_bajo_rendimiento':prob, 'Alerta':alerta,
        'Nivel':np.where(alerta == 1, 'Priorizar revisión', 'Seguimiento habitual'),
        'Fecha_UTC':fecha, 'ID_Lote':lote, 'Version_modelo':m['version'],
        'Modelo':m['nombre_modelo'], 'Umbral':m['umbral'],
        'Corte_Math':m['corte_math'], 'Origen':origen})
    destino = Path(destino); destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / f'predicciones_{lote}.csv'
    salida.to_csv(archivo, index=False, encoding='utf-8-sig')
    resumen = {'ID_Lote':lote, 'Fecha_UTC':fecha, 'Modelo':m['nombre_modelo'],
        'Version_modelo':m['version'], 'Umbral':m['umbral'], 'Filas':len(salida),
        'Alertas':int(alerta.sum()), 'Tasa_alertas':float(alerta.mean()),
        'Valores_imputados':int(x.isna().sum().sum()), 'Origen':origen,
        'SHA256_entrada':hashlib.sha256(Path(ruta_csv).read_bytes()).hexdigest(),
        'Archivo':archivo.name}
    (destino/f'resumen_{lote}.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
    historial = destino/'historial_ejecuciones.csv'
    pd.DataFrame([resumen]).to_csv(historial, mode='a', header=not historial.exists(), index=False)
    # Power BI lee este único archivo; el histórico queda separado.
    tablero = destino/'dashboard'; tablero.mkdir(exist_ok=True)
    temporal = tablero/'ultimo_lote.tmp'
    salida.to_csv(temporal, index=False, encoding='utf-8-sig')
    temporal.replace(tablero/'predicciones_actuales.csv')
    return salida, resumen

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrada', required=True)
    parser.add_argument('--modelo', required=True)
    parser.add_argument('--salida', required=True)
    parser.add_argument('--origen', default='demostracion_sintetica')
    args = parser.parse_args()
    _, resumen = procesar_lote(args.entrada, args.modelo, args.salida, args.origen)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
