import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier

# URL to the dataset
URL = "https://raw.githubusercontent.com/marcopeix/MachineLearningModelDeploymentwithStreamlit/master/17_caching_capstone/data/mushrooms.csv"

# Columns to be used in the dataset
COLS = ['class', 'odor', 'gill-size', 'gill-color', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'ring-type', 'spore-print-color']

# Read data from the URL
df = pd.read_csv(URL)
# Select only the specified columns
df = df[COLS]

# Create a pipeline with an ordinal encoder and a gradient boosting classifier
pipe = Pipeline([
    ('encoder', OrdinalEncoder()),  # Step 1: Encode categorical features as ordinal integers
    ('gbc', GradientBoostingClassifier(max_depth=5, random_state=42))  # Step 2: Apply a gradient boosting classifier
])

# Separate features (X) and target (y) from the dataset
X = df.drop(['class'], axis=1)  # Features: all columns except 'class'
y = df['class']  # Target: the 'class' column

# Fit the pipeline to the data
pipe.fit(X, y)

# Save the fitted pipeline to a file
dump(pipe, 'model/pipe.joblib')
