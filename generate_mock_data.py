import pandas as pd
import numpy as np
import os

def generate_mock_data():
    print("Generating mock dataset...")
    os.makedirs('dataset', exist_ok=True)
    
    np.random.seed(42)
    n_samples = 1000
    
    # 1. Generate Application Records
    ids = np.random.choice(np.arange(5000000, 6000000), n_samples, replace=False)
    
    genders = np.random.choice(['M', 'F'], n_samples)
    own_car = np.random.choice(['Y', 'N'], n_samples)
    own_property = np.random.choice(['Y', 'N'], n_samples)
    children = np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.08, 0.02])
    
    # Incomes centered around $180,000
    income = np.random.lognormal(mean=12.0, sigma=0.5, size=n_samples)
    
    education = np.random.choice([
        'Secondary / secondary special', 'Higher education', 
        'Incomplete higher', 'Lower secondary', 'Academic degree'
    ], n_samples, p=[0.6, 0.3, 0.07, 0.02, 0.01])
    
    family_status = np.random.choice([
        'Married', 'Single / not married', 'Civil marriage', 'Divorced', 'Widow'
    ], n_samples)
    
    housing_type = np.random.choice([
        'House / apartment', 'With parents', 'Rented apartment', 
        'Municipal apartment', 'Office apartment', 'Co-op apartment'
    ], n_samples, p=[0.88, 0.05, 0.03, 0.02, 0.01, 0.01])
    
    # Age: between 21 and 65 (represented as DAYS_BIRTH, negative)
    ages = np.random.randint(21, 65, n_samples)
    days_birth = -ages * 365
    
    # Employed: negative days. Some unemployed (represented as positive value 365243 in original Kaggle dataset)
    employed_years = np.random.randint(0, 30, n_samples)
    days_employed = -employed_years * 365
    # Let's make ~10% unemployed
    unemployed_idx = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
    days_employed[unemployed_idx] = 365243
    
    work_phone = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    phone = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    email = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    occupations = np.random.choice([
        'Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 
        'High skill tech staff', 'Accountants', 'Medicine staff', 
        'Security staff', 'Cooking staff', 'Cleaning staff', 
        'Private service staff', 'Low-skill Laborers', 'Waiters/Barmen staff', 
        'Secretaries', 'HR staff', 'Realty agents', 'IT staff', None
    ], n_samples)
    
    # Family members count
    family_members = children + np.random.choice([1, 2], n_samples, p=[0.3, 0.7])
    
    app_df = pd.DataFrame({
        'ID': ids,
        'CODE_GENDER': genders,
        'FLAG_OWN_CAR': own_car,
        'FLAG_OWN_REALTY': own_property,
        'CNT_CHILDREN': children,
        'AMT_INCOME_TOTAL': income,
        'NAME_EDUCATION_TYPE': education,
        'NAME_FAMILY_STATUS': family_status,
        'NAME_HOUSING_TYPE': housing_type,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'FLAG_WORK_PHONE': work_phone,
        'FLAG_PHONE': phone,
        'FLAG_EMAIL': email,
        'OCCUPATION_TYPE': occupations,
        'CNT_FAM_MEMBERS': family_members
    })
    
    app_df.to_csv('dataset/application_record.csv', index=False)
    print(f"Saved dataset/application_record.csv: {app_df.shape}")
    
    # 2. Generate Credit Records
    # For each application, generate 1 to 24 monthly records
    credit_records = []
    for i in ids:
        months = np.random.randint(1, 25)
        for m in range(months):
            # 'C' means paid off, 'X' means no loan, '0' means 1-29 days overdue, etc.
            status = np.random.choice(['0', '1', '2', '3', '4', '5', 'C', 'X'], p=[0.4, 0.05, 0.01, 0.01, 0.01, 0.01, 0.4, 0.11])
            credit_records.append({
                'ID': i,
                'MONTHS_BALANCE': -m,
                'STATUS': status
            })
            
    credit_df = pd.DataFrame(credit_records)
    credit_df.to_csv('dataset/credit_record.csv', index=False)
    print(f"Saved dataset/credit_record.csv: {credit_df.shape}")
    
    print("Mock data generated successfully!")

if __name__ == "__main__":
    generate_mock_data()
