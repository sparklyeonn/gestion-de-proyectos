"""Ejecutar con python -m streamlit run dashboard_matematicas.py"""
from contextlib import closing
from pathlib import Path
from io import BytesIO
import hashlib
import sqlite3

import pandas as pd

PROB = "Probabilidad_bajo_rendimiento"
MODELO = "HistGradientBoosting"
UMBRAL = 0.14


def leer_lote(archivo):
    datos = pd.read_csv(archivo, dtype={"ID_Caso": "string"}, float_precision="round_trip")
    requeridas = {"ID_Caso", "Grade", PROB, "Alerta", "Modelo", "Umbral"}
    faltantes = requeridas.difference(datos.columns)
    if faltantes:
        raise ValueError("Faltan columnas: " + ", ".join(sorted(faltantes)))
    datos["ID_Caso"] = datos["ID_Caso"].str.strip()
    ids = datos["ID_Caso"]
    if datos.empty or ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("El lote debe contener filas con identificadores únicos y completos.")
    if "ID_Lote" in datos and datos["ID_Lote"].nunique(dropna=False) != 1:
        raise ValueError("Carga un solo lote a la vez.")
    for campo in ["Grade", PROB, "Alerta", "Umbral"]:
        datos[campo] = pd.to_numeric(datos[campo], errors="raise")
    if not (datos["Modelo"].eq(MODELO) & datos["Umbral"].eq(UMBRAL)).all():
        raise ValueError("Usa el CSV del primer modelo HistGradientBoosting con umbral 0,14.")
    if not datos[PROB].between(0, 1).all() or not datos["Alerta"].eq(datos[PROB].ge(UMBRAL).astype(int)).all():
        raise ValueError("Las probabilidades o las alertas no corresponden al umbral 0,14.")
    if not datos["Grade"].isin([9, 10, 11, 12]).all():
        raise ValueError("Se esperaba un nivel entre 9 y 12 en Grade.")
    return datos


BASE = Path(__file__).resolve().parent
PERFILES = {"Docente A": [9, 10], "Docente B": [11, 12]}
NIVELES = {9: "1.º medio", 10: "2.º medio", 11: "3.º medio", 12: "4.º medio"}


def iniciar_base(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS lote_actual (
            id INTEGER PRIMARY KEY CHECK(id = 1), contenido BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS revisiones (
            lote TEXT NOT NULL, estudiante TEXT NOT NULL,
            PRIMARY KEY(lote, estudiante));
    """)


def lote_disponible(db, carpeta):
    registro = db.execute("SELECT contenido FROM lote_actual WHERE id = 1").fetchone()
    if registro:
        return registro[0]
    rutas = [carpeta / "lote_matematicas_1500.csv", carpeta / "predicciones_actuales.csv"]
    existentes = [ruta for ruta in rutas if ruta.is_file()]
    if existentes:
        return max(existentes, key=lambda ruta: ruta.stat().st_mtime).read_bytes()
    return None


def mis_cursos(datos, perfil):
    vista = datos.loc[datos["Grade"].isin(PERFILES[perfil])].copy()
    vista["Curso docente"] = vista["Grade"].map(NIVELES)
    if "Curso" in vista:
        seccion = vista["Curso"].astype("string").str.strip()
        tiene_seccion = seccion.fillna("").ne("")
        vista.loc[tiene_seccion, "Curso docente"] += " · " + seccion[tiene_seccion]
    return vista


def actualizar_revision(db, lote, estudiante, revisado):
    with db:
        if revisado:
            db.execute("INSERT OR IGNORE INTO revisiones VALUES (?, ?)", (lote, estudiante))
        else:
            db.execute("DELETE FROM revisiones WHERE lote=? AND estudiante=?", (lote, estudiante))


def administrar(st, db):
    st.title("Administración del lote")
    st.write("Carga el CSV de predicciones sintéticas que verán los perfiles docentes.")
    archivo = st.file_uploader("Archivo de predicciones", type="csv")
    if archivo is not None:
        contenido = archivo.getvalue()
        datos = leer_lote(BytesIO(contenido))
        st.write(f"{len(datos):,} estudiantes · {int(datos['Alerta'].sum())} con alerta")
        if st.button("Usar este lote", type="primary"):
            with db:
                db.execute("INSERT OR REPLACE INTO lote_actual VALUES (1, ?)", (contenido,))
            st.success("Lote guardado. Abre Mis cursos para revisar los resultados.")


def mostrar_resumen(st, vista, curso, alertas):
    if curso == "Todos mis cursos":
        st.subheader("Estudiantes con alerta por curso")
        resumen = vista.groupby("Curso docente")["Alerta"].agg(
            con_alerta="sum", evaluados="size", proporcion="mean"
        ).sort_values("proporcion", ascending=False, kind="stable")
        for nombre, fila in resumen.iterrows():
            porcentaje = f"{fila.proporcion:.1%}".replace(".", ",")
            etiqueta = (f"{nombre} · {int(fila.con_alerta)} de {int(fila.evaluados)} "
                        f"estudiantes con alerta ({porcentaje})")
            st.progress(float(fila.proporcion), text=etiqueta)
        return

    st.subheader("Avance de revisión del curso")
    total = len(alertas)
    if total == 0:
        st.caption("Este curso no tiene alertas que revisar en el lote actual.")
        return
    revisadas = int(alertas["Estado"].eq("Revisado").sum())
    porcentaje = f"{revisadas / total:.1%}".replace(".", ",")
    st.progress(revisadas / total, text=f"{revisadas} de {total} alertas revisadas ({porcentaje})")


def mostrar_cursos(st, db, contenido, perfil):
    datos = leer_lote(BytesIO(contenido))
    lote = hashlib.sha256(contenido).hexdigest()
    vista = mis_cursos(datos, perfil)
    if vista.empty:
        st.info("Este lote no contiene estudiantes de los niveles asignados al perfil.")
        return
    curso = st.selectbox("Curso", ["Todos mis cursos", *sorted(vista["Curso docente"].unique())],
                         key=f"curso_{perfil}")
    if curso != "Todos mis cursos":
        vista = vista.loc[vista["Curso docente"].eq(curso)]
    if "Fecha_UTC" in datos:
        fecha = pd.to_datetime(datos["Fecha_UTC"], utc=True, errors="coerce").max()
        if pd.notna(fecha):
            st.caption("Última actualización del lote · " +
                       fecha.tz_convert("America/Santiago").strftime("%d/%m/%Y a las %H:%M") +
                       " (hora de Chile)")
    revisados = {fila[0] for fila in db.execute(
        "SELECT estudiante FROM revisiones WHERE lote=?", (lote,))}
    alertas = vista.loc[vista.Alerta.eq(1)].sort_values(PROB, ascending=False).copy()
    alertas["Estado"] = alertas.ID_Caso.isin(revisados).map({True: "Revisado", False: "Pendiente"})
    a, b, c = st.columns(3)
    a.metric("Estudiantes evaluados", len(vista))
    b.metric("Con alerta", len(alertas), help=f"{100*len(alertas)/len(vista):.1f}% del grupo")
    c.metric("Pendientes de revisión", int(alertas.Estado.eq("Pendiente").sum()))
    mostrar_resumen(st, vista, curso, alertas)
    st.subheader("Estudiantes para revisar")
    st.caption("Revisa primero los casos con mayor probabilidad estimada de bajo puntaje.")
    filtro = st.radio("Estado de revisión", ["Pendiente", "Revisado", "Todos"], horizontal=True)
    tabla = alertas if filtro == "Todos" else alertas.loc[alertas.Estado.eq(filtro)]
    if tabla.empty:
        st.info("No hay estudiantes con alerta para esta selección.")
    else:
        tabla = tabla.assign(Probabilidad=(tabla[PROB] * 100).round(1)).rename(
            columns={"ID_Caso": "Estudiante", "Curso docente": "Curso"})
        st.dataframe(tabla[["Estudiante", "Curso", "Probabilidad", "Estado"]],
                     column_config={"Probabilidad": st.column_config.ProgressColumn(
                         "Probabilidad estimada", min_value=0, max_value=100, format="%.1f%%")},
                     hide_index=True, use_container_width=True)
        estudiante = st.selectbox("Estudiante para actualizar", tabla.Estudiante.tolist())
        marcar = estudiante not in revisados
        if st.button("Marcar como revisado" if marcar else "Volver a pendiente", type="primary"):
            actualizar_revision(db, lote, estudiante, marcar)
            st.rerun()
    with st.expander("Cómo se genera la alerta"):
        st.write("El modelo HistGradientBoosting marca para revisión las probabilidades "
                 "iguales o superiores al 14 % de obtener menos de 60 puntos sobre 100. "
                 "El docente contrasta la señal con los antecedentes del estudiante.")


def main():
    import streamlit as st

    st.set_page_config(page_title="Mis cursos · Matemáticas", page_icon="📘", layout="wide")
    st.sidebar.header("Acompañamiento en matemáticas")
    pagina = st.sidebar.radio("Vista", ["Mis cursos", "Administración"])
    st.sidebar.caption("Prototipo con perfiles simulados y datos sintéticos.")
    try:
        with closing(sqlite3.connect(BASE / "seguimiento_demo.sqlite3")) as db:
            iniciar_base(db)
            if pagina == "Administración":
                administrar(st, db)
                return
            perfil = st.sidebar.selectbox("Simular docente", list(PERFILES))
            st.title("Mis cursos")
            st.caption(f"{perfil} · Acompañamiento en matemáticas")
            contenido = lote_disponible(db, BASE)
            if contenido is None:
                st.info("Para comenzar, carga el CSV del cuaderno en Administración.")
            else:
                mostrar_cursos(st, db, contenido, perfil)
    except (ValueError, OSError, sqlite3.Error, UnicodeError) as error:
        st.error(f"No se pudo completar la operación. {error}")


if __name__ == "__main__":
    main()
