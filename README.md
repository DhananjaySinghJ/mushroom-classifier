# Mushroom Classifier 🍄

This repository contains a Streamlit application that classifies mushrooms as either edible or poisonous based on various features. The project consists of two main files: `main.py` and `model.py`.

## Files

- **`main.py`**: This file contains the code for the Streamlit application. It allows users to input various features of a mushroom and uses a pre-trained machine learning model to predict whether the mushroom is edible or poisonous.
- **`model.py`**: This file contains the code to train and save the machine learning model. It uses a dataset of mushrooms, processes the data, and trains a Gradient Boosting Classifier.

## Getting Started

### Prerequisites

- Python 3.7 or later
- Required Python packages listed in `requirements.txt`

### Installing

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/mushroom-classifier.git
    cd mushroom-classifier
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

1. **Train the model and save the pipeline**:
    ```sh
    python model.py
    ```

2. **Run the Streamlit app**:
    ```sh
    streamlit run main.py
    ```

3. **Open your browser** and go to `http://localhost:8501` to see the app.

### Usage

1. **Select the values for prediction**: In the Streamlit app, select the values for each feature of the mushroom (e.g., odor, gill size, gill color, etc.).
2. **Ask the model for a prediction**: Click the "Predict" button to get the classification result.
3. **View the result**: The app will display whether the mushroom is edible or poisonous.

## Dataset

The dataset used for training the model is the Mushroom Dataset from [Kaggle](https://www.kaggle.com/uciml/mushroom-classification). It contains various features of mushrooms and their classification as edible or poisonous.

## Model

The machine learning model used in this project is a Gradient Boosting Classifier. The model pipeline includes an Ordinal Encoder to handle categorical features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset is provided by the UCI Machine Learning Repository.
- The project structure and model training approach are inspired by various online resources and tutorials on machine learning and Streamlit.
