import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('D:/telco churn predictor/randomforest.pkl', 'rb'))
columns = ['tenure', 'PhoneService', 'Contract',
           'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges']

def predict_churn(input_data):
    # Preprocess the input data
    input_df = pd.DataFrame([input_data], columns=columns)

    # Make predictions using the loaded model
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    return prediction[0], probability[0]

def main():

    st.title("Customer Churn Predictor")
    st.write("Enter customer details to predict churn.")

    # Create input fields for user input
    monthly_charges = st.number_input("Monthly Charges ($)")
    tenure = st.slider("Tenure (months)", 0, 100, 1)
    phone_service = st.radio("Has Phone Service?", ["Yes", "No"])
    paperless_billing = st.radio("Paperless Billing?", ["Yes", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Bank transfer", "Credit card", "Electronic check", "Mailed check"])

    # Map input to numeric values
    phone_service = 1 if phone_service == "Yes" else 0
    contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    contract = contract_mapping[contract]
    paperless_billing = 1 if paperless_billing == "Yes" else 0
    payment_method_mapping = {
        "Bank transfer": 0,
        "Credit card": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }
    payment_method = payment_method_mapping[payment_method]

    # Create a dictionary to store the user input
    input_data = {
        'tenure': tenure,
        'PhoneService': phone_service,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges
    }

    # Predict churn based on user input
    churn_probability = predict_churn(input_data)
    churn_prediction = churn_probability[1]

    # Display the prediction
    st.subheader("Churn Prediction")
    if churn_prediction >= 0.4:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")

    # Display the churn probability
    st.subheader("Churn Probability")
    st.write("The probability of churn is:", churn_probability)

# Assuming the rest of your code remains the same




# Run the Streamlit app
if __name__ == '__main__':
    main()