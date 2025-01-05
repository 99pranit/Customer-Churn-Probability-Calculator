# Customer Churn Prediction Calculator  

A machine learning project that predicts customer churn for telecom companies using the [Kaggle Telecom Customer Churn Prediction Dataset](https://www.kaggle.com). This project helps businesses identify at-risk customers and implement retention strategies to minimize churn.  

---

## Introduction  

The **Customer Churn Prediction Calculator** uses advanced machine learning techniques to analyze customer data and predict the likelihood of churn. By understanding the factors driving churn, telecom companies can design targeted strategies to retain customers and improve satisfaction.  

---

## Dataset  

- **Source**: [Kaggle Telecom Customer Churn Prediction Dataset](https://www.kaggle.com)  
- **Size**: Includes detailed records of telecom customers, including demographics, service usage, and churn status.  
- **Key Features**:  
  - `customerID`: Unique identifier for each customer.  
  - `tenure`: Duration of the customer's stay with the telecom service provider.  
  - `MonthlyCharges`: Monthly amount charged to the customer.  
  - `TotalCharges`: Total amount charged during tenure.  
  - `Churn`: Indicates whether the customer churned (`Yes` or `No`).  

---

## Features  

- **Data Preprocessing**:  
  - Handled missing values and corrected data inconsistencies.  
  - Standardized numerical features for uniform scaling.  
- **Feature Engineering**:  
  - Extracted insights such as average monthly spend per tenure.  
  - One-hot encoded categorical features like `Contract`, `PaymentMethod`, etc.  
- **Machine Learning Models**:  
  - Trained classifiers like Random Forest, and Gradient Boosting.  
  - Hyperparameter tuning with GridSearchCV for optimal performance.  

---

## Technologies Used  

- **Programming Language**: Python  
- **Libraries and Frameworks**:  
  - Scikit-learn for model training and evaluation  
  - Pandas and NumPy for data handling  
  - Matplotlib and Seaborn for visualization  

---

## Future Work

- Integrate additional features like customer feedback for sentiment analysis.
- Test deep learning models for sentimental effects on churn patterns.
