import streamlit as st
import pandas as pd
import plotly.express as px
from supabase_api import fetch_data_from_supabase

def stats_visuals_section():
    """
    Sección de Estadísticas y Visualización.
    """
    st.header("Estadísticas y Visualización")

    # Consultar datos desde Supabase
    try:
        data = fetch_data_from_supabase("ordencompra")
        if not data:
            st.warning("No hay datos disponibles en la tabla `ordencompra`.")
            return

        df = pd.DataFrame(data)
        st.write("Datos cargados desde Supabase:")
        st.dataframe(df)

        # Filtros
        st.subheader("Filtros")
        estados = st.multiselect("Seleccione Estados:", options=df["estado"].unique(), default=df["estado"].unique())
        tipos = st.multiselect("Seleccione Tipos de Compra:", options=df["tipo_compra"].unique(), default=df["tipo_compra"].unique())
        rango_fechas = st.date_input("Seleccione el Rango de Fechas:", [])
        
        # Aplicar filtros
        if estados:
            df = df[df["estado"].isin(estados)]
        if tipos:
            df = df[df["tipo_compra"].isin(tipos)]
        if rango_fechas and len(rango_fechas) == 2:
            df = df[(df["fecha_creacion_compra"] >= str(rango_fechas[0])) & (df["fecha_creacion_compra"] <= str(rango_fechas[1]))]

        # Estadísticas Descriptivas
        st.subheader("Estadísticas Descriptivas")
        st.write("Resumen de los datos filtrados:")
        st.write(df.describe())

        # Gráficos
        st.subheader("Visualización de Datos")

        # Distribución de Estados
        estado_count = df["estado"].value_counts().reset_index()
        estado_count.columns = ["estado", "conteo"]
        fig_estado = px.bar(estado_count, x="estado", y="conteo", title="Distribución de Órdenes por Estado")
        st.plotly_chart(fig_estado)

        # Evolución Temporal de Órdenes
        df["fecha_creacion_compra"] = pd.to_datetime(df["fecha_creacion_compra"], errors="coerce")
        df_time = df.groupby(df["fecha_creacion_compra"].dt.to_period("M")).size().reset_index(name="conteo")
        df_time["fecha_creacion_compra"] = df_time["fecha_creacion_compra"].astype(str)
        fig_tiempo = px.line(df_time, x="fecha_creacion_compra", y="conteo", title="Evolución Temporal de Órdenes")
        st.plotly_chart(fig_tiempo)

        # Distribución por Tipo de Compra
        tipo_count = df["tipo_compra"].value_counts().reset_index()
        tipo_count.columns = ["tipo_compra", "conteo"]
        fig_tipo = px.pie(tipo_count, names="tipo_compra", values="conteo", title="Distribución por Tipo de Compra")
        st.plotly_chart(fig_tipo)

    except Exception as e:
        st.error(f"Error al consultar o procesar datos desde Supabase: {e}")
