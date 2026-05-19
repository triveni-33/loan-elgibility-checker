import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('loan.csv')

# Factorize categorical columns
data['Gender'], _ = pd.factorize(data['Gender'])
data['Married'], _ = pd.factorize(data['Married'])
data['Dependents'], _ = pd.factorize(data['Dependents'])
data['Education'], _ = pd.factorize(data['Education'])
data['Self_Employed'], _ = pd.factorize(data['Self_Employed'])
data['Property_Area'], _ = pd.factorize(data['Property_Area'])
data['Loan_Status'], _ = pd.factorize(data['Loan_Status'])

# Prepare features and target
X = data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
y = data['Loan_Status']
    
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Evaluate the model
predictions = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy using Random Forest Classifier:", accuracy)

# Save the model
joblib.dump(rf_classifier, "mymodel_rfc.h5")