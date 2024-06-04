import streamlit as st
import numpy as np
from joblib import load

# Cache the model loading process with a spinner display
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    # Load the pre-trained model pipeline from the specified path
    pipe = load('model/pipe.joblib')
    return pipe

# Cache the prediction function with a spinner display
@st.cache_data(show_spinner="Making a prediction...")
def make_prediction(_pipe, X_pred):
    # Extract features from the prediction input and reshape them for prediction
    features = [each[0] for each in X_pred]
    features = np.array(features).reshape(1, -1)
    # Make a prediction using the loaded model pipeline
    pred = _pipe.predict(features)
    return pred[0]

# Main function to run the Streamlit app
if __name__ == "__main__":
    st.title("Mushroom classifier 🍄")
    
    st.subheader("Step 1: Select the values for prediction")

    # Create three columns for user inputs
    col1, col2, col3 = st.columns(3)

    # Column 1: Inputs for odor, stalk surface above ring, and stalk color below ring
    with col1:
        odor = st.selectbox('Odor', ('a - almond', 'l - anisel', 'c - creosote', 'y - fishy', 'f - foul', 'm - musty', 'n - none', 'p - pungent', 's - spicy'))
        stalk_surface_above_ring = st.selectbox('Stalk surface above ring', ('f - fibrous', 'y - scaly', 'k - silky', 's - smooth'))
        stalk_color_below_ring = st.selectbox('Stalk color below ring', ('n - brown', 'b - buff', 'c - cinnamon', 'g - gray', 'o - orange', 'p - pink', 'e - red', 'w - white', 'y - yellow'))

    # Column 2: Inputs for gill size, stalk surface below ring, and ring type
    with col2:
        gill_size = st.selectbox('Gill size', ('b - broad', 'n - narrow'))
        stalk_surface_below_ring = st.selectbox('Stalk surface below ring', ('f - fibrous', 'y - scaly', 'k - silky', 's - smooth'))
        ring_type = st.selectbox('Ring type', ('e - evanescente', 'f - flaring', 'l - large', 'n - none', 'p - pendant', 's - sheathing', 'z - zone'))

    # Column 3: Inputs for gill color, stalk color above ring, and spore print color
    with col3:
        gill_color = st.selectbox('Gill color', ('k - black', 'n - brown', 'b - buff', 'h - chocolate', 'g - gray', 'r - green', 'o - orange', 'p - pink', 'u - purple', 'e - red', 'w - white', 'y - yellow'))
        stalk_color_above_ring = st.selectbox('Stalk color above ring', ('n - brown', 'b - buff', 'c - cinnamon', 'g - gray', 'o - orange', 'p - pink', 'e - red', 'w - white', 'y - yellow'))
        spore_print_color = st.selectbox('Spore print color', ('k - black', 'n - brown', 'b - buff', 'h - chocolate', 'r - green', 'o - orange', 'u - purple', 'w - white', 'y - yellow'))

    st.subheader("Step 2: Ask the model for a prediction")

    # Button to trigger the prediction
    pred_btn = st.button("Predict", type="primary")

    # If the prediction button is pressed
    if pred_btn:
        # Load the model pipeline
        pipe = load_model()

        # Create a list of the selected input features
        x_pred = [odor, 
                  gill_size, 
                  gill_color, 
                  stalk_surface_above_ring, 
                  stalk_surface_below_ring, 
                  stalk_color_above_ring, 
                  stalk_color_below_ring, 
                  ring_type, 
                  spore_print_color]
        
        # Make a prediction based on the selected features
        pred = make_prediction(pipe, x_pred)

        # Convert the prediction result to a user-friendly message
        nice_pred = "The mushroom is poisonous 🤢" if pred == 'p' else "The mushroom is edible 🍴"

        # Display the prediction result
        st.write(nice_pred)
