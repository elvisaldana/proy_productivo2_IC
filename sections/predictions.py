import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from supabase_api import fetch_data_from_supabase

def predictions_section():
    st.header("Predicciones de Compras")

    # Submenú de predicciones
    prediction_type = st.radio(
        "Seleccione el tipo de predicción:",
        ["Demanda Futura", "Tiempos de Entrega", "Compras Atípicas", "Evolución Temporal de Demandas"]
    )

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

    # Codificar columnas categóricas automáticamente
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    st.write(f"Columnas categóricas detectadas: {categorical_columns}")
    df = encode_categorical_columns(df, categorical_columns)

    # Predicción de Demanda Futura
    if prediction_type == "Demanda Futura":
        st.subheader("Predicción de Demanda Futura")
        features = ["categoria", "tipo_compra", "cantidad", "precio_total"]
        if validate_features(df, features):
            X, y = prepare_regression_data(df, features, target="cantidad")
            predict_future_demand(X, y)

    # Predicción de Tiempos de Entrega
    elif prediction_type == "Tiempos de Entrega":
        st.subheader("Predicción de Tiempos de Entrega")
        features = ["nombre_proveedor", "producto_tipo", "cantidad"]
        if validate_features(df, features):
            X, y = prepare_regression_data(df, features, target="tiempo_entrega")
            predict_delivery_times(X, y)

    # Clasificación de Compras Atípicas
    elif prediction_type == "Compras Atípicas":
        st.subheader("Clasificación de Compras Atípicas")
        features = ["precio_total", "cantidad", "categoria"]
        if validate_features(df, features):
            try:
                X, y = prepare_classification_data(df, features, target="atipica")
                classify_purchases(X, y)
            except Exception as e:
                st.error(f"Error durante la preparación de los datos o el entrenamiento: {e}")

    # Evolución Temporal de Demandas
    elif prediction_type == "Evolución Temporal de Demandas":
        st.subheader("Evolución Temporal de Demandas")
        if "fecha_pedido_compra" in df.columns and "cantidad" in df.columns:
            temporal_demand_evolution(df)

# Codificar columnas categóricas
def encode_categorical_columns(df, categorical_columns):
    """
    Codifica las columnas categóricas en un DataFrame.
    :param df: DataFrame a procesar.
    :param categorical_columns: Lista de columnas categóricas a codificar.
    :return: DataFrame con las columnas categóricas codificadas.
    """
    df = df.copy()
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

# Validar características
def validate_features(df, features):
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        st.error(f"Faltan las siguientes características: {', '.join(missing)}")
        return False
    return True

# Preprocesar datos para regresión
def prepare_regression_data(df, features, target):
    X = df[features]
    y = df[target]
    return X, y

# Predecir demanda futura
def predict_future_demand(X, y):
    try:
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        st.success("Modelo entrenado correctamente para la predicción de demanda futura.")
        st.write(f"MAE: {mean_absolute_error(y, model.predict(X))}")
        st.plotly_chart(px.line(x=range(len(y)), y=model.predict(X), title="Demanda Futura"))
    except Exception as e:
        st.error(f"Error en la predicción de demanda futura: {e}")

# Predecir tiempos de entrega
def predict_delivery_times(X, y):
    try:
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        st.success("Modelo entrenado correctamente para la predicción de tiempos de entrega.")
        st.write(f"MAE: {mean_absolute_error(y, model.predict(X))}")
        st.plotly_chart(px.scatter(x=y, y=model.predict(X), title="Tiempos de Entrega"))
    except Exception as e:
        st.error(f"Error en la predicción de tiempos de entrega: {e}")

# Clasificar compras atípicas
def classify_purchases(X, y):
    """
    Entrena un modelo para clasificar compras atípicas.
    """
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        st.success("Modelo entrenado correctamente para la clasificación de compras atípicas.")
        st.write(f"Precisión: {model.score(X, y)}")
        st.plotly_chart(px.bar(x=range(len(y)), y=model.predict(X), title="Clasificación de Compras Atípicas"))
    except Exception as e:
        st.error(f"Error en la clasificación de compras atípicas: {e}")
# Evolución temporal
def temporal_demand_evolution(df):
    try:
        df["fecha_pedido_compra"] = pd.to_datetime(df["fecha_pedido_compra"])
        temporal_df = df.groupby("fecha_pedido_compra")["cantidad"].sum().reset_index()
        st.plotly_chart(px.line(temporal_df, x="fecha_pedido_compra", y="cantidad", title="Evolución Temporal"))
    except Exception as e:
        st.error(f"Error en la evolución temporal de demandas: {e}")


# Función auxiliar: Preprocesar datos para clasificación
def prepare_classification_data(df, features, target):
    """
    Prepara los datos para un modelo de clasificación.
    """
    try:
        # Verificar si la columna objetivo existe
        if target not in df.columns:
            st.warning(f"La columna objetivo '{target}' no está presente. Generando columna basada en reglas...")
            # Generar columna objetivo basada en reglas, por ejemplo:
            df[target] = (df["precio_total"] > df["precio_total"].quantile(0.95)).astype(int)

        # Validar que todas las características estén presentes
        missing_features = [feature for feature in features if feature not in df.columns]
        if missing_features:
            raise ValueError(f"Las siguientes características no están presentes: {', '.join(missing_features)}")

        # Verificar valores nulos
        if df[features].isnull().any().any() or df[target].isnull().any():
            raise ValueError("Los datos contienen valores nulos. Limpia los datos antes de entrenar.")

        # Separar características y objetivo
        X = df[features]
        y = df[target]
        return X, y

    except Exception as e:
        raise ValueError(f"Error al preparar los datos de clasificación: {e}")
def encode_categorical_columns(df, categorical_columns):
    """
    Codifica las columnas categóricas en un DataFrame.
    :param df: DataFrame a procesar.
    :param categorical_columns: Lista de columnas categóricas a codificar.
    :return: DataFrame con las columnas categóricas codificadas.
    """
    df = df.copy()
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

