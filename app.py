import pandas as pd
import joblib
import streamlit as st
import os

# Configurar el título de la aplicación
st.title('Predicción de Aprobación de Curso')

# --- Cargar Recursos ---
# Asegúrate de que estos archivos estén en el directorio correcto accesibles desde donde ejecutas la app
try:
    # Load the dataset
    file_path = 'Aprobacion curso 2019 fut.xlsx'
    df = pd.read_excel(file_path, sheet_name=1) # Second sheet is index 1

    # Load the encoder, scaler, and model
    onehot_encoder = joblib.load('onehot_encoder.joblib')
    minmax_scaler = joblib.load('minmax_scaler.joblib')
    model = joblib.load('best_stacking_model.joblib')

    st.success("Dataset, encoder, scaler, and model loaded successfully!")

except FileNotFoundError:
    st.error("Error: Asegúrate de que los archivos 'Aprobacion curso 2019 fut.xlsx', 'onehot_encoder.joblib', 'minmax_scaler.joblib' y 'best_stacking_model.joblib' estén en el directorio correcto.")
    st.stop() # Detiene la ejecución si no se encuentran los archivos


# --- Interfaz de Usuario para Predicción Individual ---
st.subheader('Realizar una Predicción Individual')

# Input fields for user
# Ensure the options for selectbox match the categories the encoder was trained on
if hasattr(onehot_encoder, 'categories_'):
    felder_options = onehot_encoder.categories_[0]
    felder_input = st.selectbox('Selecciona el tipo de Felder:', felder_options)
else:
    st.error("Error: No se pudieron obtener las categorías del encoder. Asegúrate de que el archivo 'onehot_encoder.joblib' es válido.")
    st.stop()

examen_input = st.number_input('Introduce la nota del Examen de Admisión de la Universidad:', min_value=0.0, max_value=10.0, step=0.01) # Adjusted max_value for typical university scales

# --- Lógica de Preprocesamiento y Predicción ---
if st.button('Predecir'):
    # Create a DataFrame from user input
    input_data = pd.DataFrame({'Felder': [felder_input], 'Examen_admisión_Universidad': [examen_input]})

    # Apply one-hot encoding to the 'Felder' input
    felder_encoded_input = onehot_encoder.transform(input_data[['Felder']])
    encoded_df_input = pd.DataFrame(felder_encoded_input, columns=onehot_encoder.get_feature_names_out(['Felder']))

    # Apply min-max scaling to the 'Examen_admisión_Universidad' input
    # Handle potential errors if input is outside the original scaler range
    try:
        examen_scaled_input = minmax_scaler.transform(input_data[['Examen_admisión_Universidad']])
    except ValueError as e:
        st.warning(f"Advertencia de escalado: {e}. El valor de 'Examen_admisión_Universidad' puede estar fuera del rango visto durante el entrenamiento del scaler.")
        # Optionally, clamp the value or handle as appropriate for your model
        st.stop() # Stop for now, but you might want a different behavior


    # Create the processed input DataFrame with the correct column order
    # Ensure all Felder categories are present, even if 0, to match model expectations
    all_felder_cols = onehot_encoder.get_feature_names_out(['Felder'])
    processed_input = pd.DataFrame(0.0, index=[0], columns=['Examen_admisión_Universidad_scaled'] + list(all_felder_cols))

    # Fill in the scaled examen score
    processed_input['Examen_admisión_Universidad_scaled'] = examen_scaled_input[0][0]

    # Fill in the one-hot encoded Felder value
    for col in encoded_df_input.columns:
        if col in processed_input.columns:
             processed_input[col] = encoded_df_input[col].iloc[0]

    # Ensure the order of columns in processed_input matches the order expected by the model
    # This is crucial for correct prediction. You might need to verify the exact order
    # from your model training process if the order below is not correct.
    expected_columns = ['Examen_admisión_Universidad_scaled', 'Felder_activo', 'Felder_equilibrio',
                        'Felder_intuitivo', 'Felder_reflexivo', 'Felder_secuencial',
                        'Felder_sensorial', 'Felder_verbal', 'Felder_visual'] # Example order, verify with your model
    processed_input = processed_input[expected_columns]


    # Make prediction on the processed input
    try:
        prediction_input = model.predict(processed_input)
        st.subheader('Resultado de la Predicción para tu Entrada:')
        st.write(prediction_input[0])
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")


