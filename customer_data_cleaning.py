import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def clean_data(customer_df, model):
    try:
        # Drop customer ID as it's not needed for prediction
        user_df = customer_df.drop(columns=['customerID'])
    except KeyError:
        raise KeyError("The input DataFrame does not contain a 'customerID' column.")

    # Get expected features from the model
    try:
        expected_features = model.get_booster().feature_names
    except AttributeError:
        raise AttributeError("Model does not have the 'get_booster' method or 'feature_names'. Ensure that the correct model is passed.")

    # Convert TotalCharges to numeric, handle errors and missing values
    try:
        user_df['TotalCharges'] = pd.to_numeric(user_df['TotalCharges'], errors='coerce').fillna(0)
    except KeyError:
        raise KeyError("The input DataFrame does not contain a 'TotalCharges' column.")
    except Exception as e:
        raise ValueError(f"An error occurred while converting 'TotalCharges' to numeric: {e}")

    # Specify columns to be one-hot encoded
    columns_to_encode = ['SeniorCitizen', 'gender', 'Partner', 'Dependents', 'PhoneService',
                         'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                         'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                         'Contract', 'PaperlessBilling', 'PaymentMethod']

    try:
        # Initialize OneHotEncoder and apply it to the categorical columns
        encoder = OneHotEncoder(sparse_output=False)
        dummy_df = encoder.fit_transform(user_df[columns_to_encode])
        one_hot_df = pd.DataFrame(dummy_df, columns=encoder.get_feature_names_out(columns_to_encode))
    except KeyError as e:
        missing_columns = set(columns_to_encode) - set(user_df.columns)
        raise KeyError(f"The following columns are missing from the input DataFrame: {missing_columns}")
    except Exception as e:
        raise ValueError(f"An error occurred during one-hot encoding: {e}")

    try:
        # Combine the encoded features with the original DataFrame and clean up
        df_encoded = pd.concat([user_df, one_hot_df], axis=1)
        df_encoded = df_encoded.drop(columns_to_encode, axis=1).replace({' ': 0}).astype(float)
        df_encoded = df_encoded.reindex(columns=expected_features, fill_value=0)
        df_encoded = df_encoded.replace({' ': 0})
    except Exception as e:
        raise ValueError(f"An error occurred while merging or processing the DataFrame: {e}")

    return df_encoded
