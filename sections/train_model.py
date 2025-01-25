import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from supabase_api import fetch_data_from_supabase

def train_model_section():
    st.header("Entrenamiento de Modelos")

    # Cargar datos desde Supabase
    try:
        table_name = "vista_analisis_compras4"
        data = fetch_data_from_supabase(table_name)
        df = pd.DataFrame(data)
        st.write("Datos cargados (vista previa):")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error al cargar datos desde Supabase: {e}")
        return

    # Seleccionar Tipo de Modelo
    tipo_modelo = st.radio("Seleccione el tipo de modelo:", ["Clasificación", "Segmentación"])

    if tipo_modelo == "Clasificación":
        st.subheader("Clasificación")

        # Seleccionar características y objetivo
        features = st.multiselect(
            "Seleccione las características para la clasificación:",
            options=df.columns,
            default=["cantidad", "precio_total", "centro_de_coste"]
        )
        target = st.selectbox("Seleccione la variable objetivo:", options=df.columns)

        if not features or not target:
            st.warning("Seleccione características y una variable objetivo para continuar.")
            return

        # Preprocesar datos
        X = preprocess_data(df, features)
        y = df[target].astype("category").cat.codes

        # Dividir en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar modelo
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Predicciones y métricas
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Precisión del modelo: {accuracy:.2%}")
        st.text(classification_report(y_test, y_pred))

    elif tipo_modelo == "Segmentación":
        st.subheader("Segmentación")

        # Seleccionar características
        features = st.multiselect(
            "Seleccione las características para la segmentación:",
            options=df.columns,
            default=["cantidad", "precio_total"]
        )
        n_clusters = st.slider("Número de Clusters:", min_value=2, max_value=10, value=3)

        if not features:
            st.warning("Seleccione al menos una característica para continuar.")
            return

        # Preprocesar datos
        X = preprocess_data(df, features)

        # Entrenar modelo
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)

        # Agregar clusters al DataFrame
        df["Cluster"] = clusters
        st.write("Segmentación completada:")
        st.dataframe(df)

        # Visualización
        if len(features) >= 2:
            fig = px.scatter(df, x=features[0], y=features[1], color="Cluster", title="Segmentación de Compras")
            st.plotly_chart(fig)

def preprocess_data(df, selected_features):
    """
    Convierte columnas categóricas a numéricas automáticamente.
    """
    X = df[selected_features].copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes
    return X
