import streamlit as st
import pandas as pd
import joblib

# Load the saved artifacts
try:
    onehot_encoder = joblib.load('onehot_encoder.joblib')
    minmax_scaler = joblib.load('minmax_scaler.joblib')
    model = joblib.load('best_stacking_model.joblib')
except FileNotFoundError:
    st.error("Error: Make sure 'onehot_encoder.joblib', 'minmax_scaler.joblib', and 'best_stacking_model.joblib' are in the same directory.")
    st.stop()

# Set up the Streamlit app structure
st.title('Student Course Approval Prediction')
st.write('This application predicts student course approval based on their Felder style and university entrance exam score.')

# Create input fields
felder_style = st.selectbox(
    'Select your Felder style:',
    ('activo', 'visual', 'equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal')
)

examen_score = st.number_input(
    'Enter your University Entrance Exam Score:',
    min_value=0.0,
    max_value=6.0,
    value=3.0,
    step=0.01
)

# Preprocess the input data
if st.button('Predict'):
    # Create a DataFrame from the inputs
    input_df = pd.DataFrame([[felder_style, examen_score]], columns=['Felder', 'Examen_admisión_Universidad'])

    # Apply one-hot encoding to 'Felder'
    # We need to reshape the column to be a 2D array as required by the encoder
    felder_encoded = onehot_encoder.transform(input_df[['Felder']])

    # Convert the encoded output to a DataFrame
    # We need to get the feature names from the encoder to name the new columns
    encoded_df = pd.DataFrame(felder_encoded, columns=onehot_encoder.get_feature_names_out(['Felder']))

    # Drop the original 'Felder' column from the input DataFrame
    input_df = input_df.drop('Felder', axis=1)

    # Concatenate the original DataFrame with the encoded DataFrame
    input_df_processed = pd.concat([input_df, encoded_df], axis=1)

    # Apply the scaler to the 'Examen_admisión_Universidad' column
    # We need to reshape the column to be a 2D array as required by the scaler
    input_df_processed['Examen_admisión_Universidad_scaled'] = minmax_scaler.transform(input_df_processed[['Examen_admisión_Universidad']])

    # Drop the original 'Examen_admisión_Universidad' column
    input_df_processed = input_df_processed.drop('Examen_admisión_Universidad', axis=1)

    # Ensure the columns are in the same order as the training data used by the model
    # The df_encoded DataFrame from the notebook has the correct order
    # Assuming df_encoded was the final processed DataFrame before training
    # Recreate df_encoded column order based on the existing kernel variable
    expected_columns = ['Felder_activo', 'Felder_equilibrio', 'Felder_intuitivo', 'Felder_reflexivo', 'Felder_secuencial', 'Felder_sensorial', 'Felder_verbal', 'Felder_visual', 'Examen_admisión_Universidad_scaled']

    # Reindex the processed input DataFrame to match the expected columns
    try:
        input_df_final = input_df_processed[expected_columns]
    except KeyError as e:
        st.error(f"Column mismatch after preprocessing: {e}")
        st.stop()

    # Make prediction
    prediction = model.predict(input_df_final)

    # Display the prediction
    st.subheader('Prediction:')
    if prediction[0] == 'si':
        st.success('The student is predicted to approve the course.')
    else:
        st.error('The student is predicted not to approve the course.')

# Instructions on how to run the application:
# 1. Save this file as app.py.
# 2. Open a terminal or command prompt.
# 3. Navigate to the directory where app.py, onehot_encoder.joblib, minmax_scaler.joblib, and best_stacking_model.joblib are saved.
# 4. Run the command: streamlit run app.py
