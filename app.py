from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)

# Load and preprocess the data
data = pd.read_csv('loan.csv')

# Factorize categorical columns
data['Gender'], _ = pd.factorize(data['Gender'])
data['Married'], _ = pd.factorize(data['Married'])
data['Dependents'], _ = pd.factorize(data['Dependents'])
data['Education'], _ = pd.factorize(data['Education'])
data['Self_Employed'], _ = pd.factorize(data['Self_Employed'])
data['Property_Area'], _ = pd.factorize(data['Property_Area'])
data['Loan_Status'], _ = pd.factorize(data['Loan_Status'])

# Fit the TF-IDF vectorizer on Loan_ID and the RandomForest model
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['Loan_ID'])
y = data['Loan_Status']

model = RandomForestRegressor()
model.fit(X, y)

# Save the trained model and TF-IDF vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(model, 'model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Loan = request.form['loan_id']
    
    # Load the TF-IDF vectorizer and the model
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('model.pkl')
    
    Loan_tfidf = tfidf.transform([Loan])
    prediction = model.predict(Loan_tfidf)
    
    if prediction[0] > 0:
        result = "You are not eligible for a loan"
    else:
        result = "You are eligible for a loan"
    
    return render_template('result.html', loan_id=Loan, result=result)

if __name__ == '__main__':
    app.run(debug=True)
