import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

def generate_customer_data(n_customers=5000):
    """Generate realistic customer dataset for churn prediction"""
    
    customers = []
    
    for i in range(n_customers):
        # Basic demographics
        customer_id = f"CUST_{i+1:05d}"
        age = np.random.normal(45, 15)
        age = max(18, min(80, int(age)))  # Ensure realistic age range
        
        gender = random.choice(['Male', 'Female'])
        
        # Account information
        tenure_months = np.random.exponential(24)  # Average 24 months
        tenure_months = max(1, min(120, int(tenure_months)))  # 1-120 months
        
        # Service usage patterns
        monthly_charges = np.random.normal(65, 25)
        monthly_charges = max(15, min(150, monthly_charges))  # $15-$150 range
        
        total_charges = monthly_charges * tenure_months + np.random.normal(0, 100)
        total_charges = max(monthly_charges, total_charges)
        
        # Contract and services
        contract_type = random.choices(
            ['Month-to-month', 'One year', 'Two year'],
            weights=[0.5, 0.3, 0.2]  # Month-to-month more likely to churn
        )[0]
        
        # Service features
        internet_service = random.choices(
            ['DSL', 'Fiber optic', 'No'],
            weights=[0.4, 0.4, 0.2]
        )[0]
        
        online_security = random.choice(['Yes', 'No']) if internet_service != 'No' else 'No internet service'
        online_backup = random.choice(['Yes', 'No']) if internet_service != 'No' else 'No internet service'
        tech_support = random.choice(['Yes', 'No']) if internet_service != 'No' else 'No internet service'
        
        # Additional services
        phone_service = random.choice(['Yes', 'No'])
        multiple_lines = random.choice(['Yes', 'No']) if phone_service == 'Yes' else 'No phone service'
        
        streaming_tv = random.choice(['Yes', 'No']) if internet_service != 'No' else 'No internet service'
        streaming_movies = random.choice(['Yes', 'No']) if internet_service != 'No' else 'No internet service'
        
        # Payment and billing
        paperless_billing = random.choice(['Yes', 'No'])
        payment_method = random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        
        # Customer service interactions
        support_calls = np.random.poisson(2)  # Average 2 calls
        support_calls = min(20, support_calls)  # Cap at 20
        
        # Usage metrics
        avg_monthly_gb = np.random.exponential(50) if internet_service != 'No' else 0
        avg_monthly_gb = min(500, avg_monthly_gb)  # Cap at 500GB
        
        # Calculate churn probability based on features
        churn_prob = 0.1  # Base probability
        
        # Risk factors that increase churn probability
        if contract_type == 'Month-to-month':
            churn_prob += 0.3
        elif contract_type == 'One year':
            churn_prob += 0.1
            
        if tenure_months < 6:
            churn_prob += 0.4
        elif tenure_months < 12:
            churn_prob += 0.2
            
        if monthly_charges > 80:
            churn_prob += 0.2
            
        if support_calls > 5:
            churn_prob += 0.3
            
        if online_security == 'No' and internet_service != 'No':
            churn_prob += 0.1
            
        if payment_method == 'Electronic check':
            churn_prob += 0.1
            
        if age < 30:
            churn_prob += 0.1
            
        # Protective factors that decrease churn probability
        if tech_support == 'Yes':
            churn_prob -= 0.1
            
        if streaming_tv == 'Yes' and streaming_movies == 'Yes':
            churn_prob -= 0.15
            
        if paperless_billing == 'Yes':
            churn_prob -= 0.05
            
        # Ensure probability is between 0 and 1
        churn_prob = max(0.05, min(0.85, churn_prob))
        
        # Determine actual churn based on probability
        churned = 'Yes' if random.random() < churn_prob else 'No'
        
        customer = {
            'customerID': customer_id,
            'gender': gender,
            'SeniorCitizen': 1 if age >= 65 else 0,
            'Partner': random.choice(['Yes', 'No']),
            'Dependents': random.choice(['Yes', 'No']),
            'tenure': tenure_months,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': random.choice(['Yes', 'No']) if internet_service != 'No' else 'No internet service',
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract_type,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': round(monthly_charges, 2),
            'TotalCharges': round(total_charges, 2),
            'SupportCalls': support_calls,
            'AvgMonthlyGB': round(avg_monthly_gb, 1),
            'Age': age,
            'Churn': churned
        }
        
        customers.append(customer)
    
    return pd.DataFrame(customers)

# Generate the dataset
print("Generating customer dataset...")
df = generate_customer_data(5000)

# Display basic information
print(f"\nDataset created with {len(df)} customers")
print(f"Churn rate: {df['Churn'].value_counts()['Yes'] / len(df) * 100:.1f}%")
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# Save to CSV
df.to_csv('customer_churn_data.csv', index=False)
print("\n✅ Dataset saved as 'customer_churn_data.csv'")

# Display some statistics
print("\n📊 Dataset Summary:")
print(f"Total Customers: {len(df)}")
print(f"Churned Customers: {df['Churn'].value_counts()['Yes']}")
print(f"Active Customers: {df['Churn'].value_counts()['No']}")
print(f"Average Tenure: {df['tenure'].mean():.1f} months")
print(f"Average Monthly Charges: ${df['MonthlyCharges'].mean():.2f}")
print(f"Contract Distribution:")
print(df['Contract'].value_counts())