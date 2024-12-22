import streamlit as st
import pandas as pd
import xgboost as xgb
from customer_data_cleaning import clean_data

# Load pre-trained XGBoost model
model = xgb.XGBClassifier()
model.load_model('churn_predict_model.json')

# Set the title for the Streamlit app
st.title("Welcome to Customer Churn Probability Calculator")

# Sidebar with options for input mode selection
with st.sidebar:
    add_radio = st.radio(
        "Choose mode",
        ("Single", "List")  # Options for single customer input or list of customers via file upload
    )

# If the user selects "List" mode to upload a CSV file
if add_radio == 'List':
    file = st.file_uploader("Upload file (in CSV format)", type="csv")
    if st.button('Predict'):
        user_df = pd.read_csv(file)# Read the uploaded CSV file
        input_data = clean_data(user_df, model)
        pred = model.predict_proba(input_data)[:, 1]# Predict churn for the data in the CSV file 
        # Round the predictions to 2 decimal places
        pred = [round(p, 2) for p in pred]
        user_df['ChurnProbability'] = pred
   
        # Display predictions in a table format
        st.dataframe(user_df[['customerID', 'ChurnProbability']])
        
        # Function to convert the DataFrame to CSV format for download
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode("utf-8")
        
        csv = convert_df(user_df[['customerID', 'ChurnProbability']])

        # Button to download the predictions as a CSV file
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="churn_data.csv",
            mime="text/csv",
        )

# If the user selects "Single" mode to input data for one customer
else:
    left_column, right_column = st.columns(2)  # Create two columns for input fields

    with left_column:
        # Collecting input data from the user in the left column
        customer_id = st.text_input("Enter Customer ID")
        tenure = st.number_input("Enter Tenure", step=1)
        monthly_charges = st.number_input("Enter Monthly Charges", step=10)
        senior_citizen = st.radio("Senior Citizen", ("Yes", "No"))
        partner = st.radio("Partner", ("Yes", "No"))
        phone_service = st.radio("Phone Service", ("Yes", "No"))
        multiple_line = st.radio("Multiple Lines", ("Yes", "No", "No Phone Service"))
        internet_service = st.radio("Internet Service", ("Yes", "No", "Fiber Optic"))
        contract = st.radio("Contract", ("Month-to-Month", "One Year", "Two Year"))
        paperless_billing = st.radio("Paperless Billing", ("Yes", "No"))

    with right_column:
        # Collecting more input data from the user in the right column
        gender = st.radio("Gender", ("Male", "Female"))
        total_charges = st.number_input("Enter Total Charges", step=100)
        dependent = st.radio("Dependent", ("Yes", "No"))
        online_security = st.radio("Online Security", ("Yes", "No", "No Internet Service"))
        online_backup = st.radio("Online Backup", ("Yes", "No", "No Internet Service"))
        device_protection = st.radio("Device Protection", ("Yes", "No", "No Internet Service"))
        tech_support = st.radio("Tech Support", ("Yes", "No", "No Internet Service"))
        streaming_tv = st.radio("Streaming TV", ("Yes", "No", "No Internet Service"))
        streaming_movies = st.radio("Streaming Movies", ("Yes", "No", "No Internet Service"))
        payment_method = st.radio("Payment Method", 
            ("Electronic Check", "Mailed Check", "Bank Transfer (Automatic)", "Credit Card (Automatic)"))

    if st.button('Predict'):
        # Combine the user inputs into a single row for prediction
        data = [[customer_id, gender, senior_citizen, partner, dependent, tenure, 
                phone_service, multiple_line, internet_service, online_security,
                online_backup, device_protection, tech_support, streaming_tv, streaming_movies,
                contract, paperless_billing, payment_method, monthly_charges, total_charges]]
        
        # Column names matching the input data
        columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                   'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                   'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
        
        churn_df = pd.DataFrame(data, columns=columns)  # Create DataFrame with the input data

        input_data = clean_data(churn_df, model)  # Clean the input data for prediction

        pred = model.predict_proba(input_data)[0, 1]  # Predict churn probability
        st.write(f"{churn_df['customerID'].to_string(index=False)} has a {pred:.2f} probability of churn.")  # Display the result

