import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
data = pd.read_csv('data.csv')

# Split features and labels
X = data[['feature1', 'feature2']]
y = data['label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'phone_classifier_model.pkl')

print("Model trained and saved.")
