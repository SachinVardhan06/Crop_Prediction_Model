import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('crop_recommendation.csv')

# Split features and labels
X = df.drop('label', axis=1)
y = df['label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Save the trained model
joblib.dump(model, 'crop_recommendation_model.pkl')
