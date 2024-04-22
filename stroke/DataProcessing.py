from sklearn.model_selection import train_test_split
import pandas as pd
from zenml.steps import step, Output
from typing import Tuple


@step
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Define dictionaries to map categorical values to numerical representations
    gender_dict = {'Male': 2, 'Female': 1, 'Other': 3}
    ever_married_dict = {'Yes': 1, 'No': 0}
    work_type_dict = {'Private': 1, 'Self-employed': 2, 'Govt_job': 3, 'children': 4, 'Never_worked': 5}
    Residence_type_dict = {'Urban': 1, 'Rural': 2}
    smoking_status_dict = {'formerly smoked': 1, 'never smoked': 2, 'smokes': 3, 'Unknown': 4}

    # Map categorical values to numerical representations using the defined dictionaries
    df['gender'] = df['gender'].map(gender_dict)
    df['ever_married'] = df['ever_married'].map(ever_married_dict)
    df['work_type'] = df['work_type'].map(work_type_dict)
    df['residence_type'] = df['Residence_type'].map(Residence_type_dict)
    df['smoking_status'] = df['smoking_status'].map(smoking_status_dict)

    # Fill missing values in the 'bmi' column with the mean value of the column
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)

    # Drop the 'id' column from the DataFrame
    # axis=1 indicates that we are dropping a column (0 would indicate dropping a row)
    df.drop('id', inplace=True, axis=1)

    # Split data into features and target
    X = df.drop('stroke', axis=1)
    y = df['stroke']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
