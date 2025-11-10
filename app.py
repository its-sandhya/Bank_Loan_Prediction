from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
model = pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        ApplicantIncome = float(request.form['ApplicantIncome'])
        Credit_History = float(request.form['Credit_History'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        
        # Prepare input for prediction (no scaler needed)
        input_features = np.array([[ApplicantIncome, Credit_History, LoanAmount, Loan_Amount_Term]])
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        result = 'Approved' if prediction == 1 else 'Not Approved'
        
        return render_template('index.html', prediction_text=f'Loan Status: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
