import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data
# ...
iris= pd.read_csv("C:\\Users\\vyshn\\OneDrive\\Desktop\\fom\\iris.csv")
print(iris.head())

x = iris.drop("species", axis=1)
y = iris["species"]
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = [
    ("Logistic Regression", LogisticRegression()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("SVM", SVC())
]

# Evaluate models
for name, model in classifiers:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Model: {name}\nAccuracy: {accuracy}\nClassification Report:\n{report}\n")

# Cross-validation
for name, model in classifiers:
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Model: {name}\nCross-Validation Scores: {scores}\nMean CV Score: {np.mean(scores)}\n")