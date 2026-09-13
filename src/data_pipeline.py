import pandas as pd
import numpy as np

def clean_and_variable_data(df: pd.DataFrame):
    """
    Cleans raw car dataset and validates strict boundary conditions.
    """
    # 1. Edge Case: Empty Dataframe
    if df.empty:
        raise ValueError("Ingested dataset contains zero records.")

    # 2. Drop the missin values before extracting features and target
    df = df.dropna(subset=['horsepower', 'price'])

    # 3. Edge Case: Data Type Validations
    if pd.api.types.is_bool_dtype(df['horsepower']) or pd.api.types.is_bool_dtype(df['price']):
        raise TypeError("Horsepower and price columns must be numeric.")

    if not pd.api.types.is_numeric_dtype(df['horsepower']) or not pd.api.types.is_numeric_dtype(df['price']):
        raise TypeError("Horsepower and price columns must be numeric.")

    # 4. Edge Case: Boundary Values (N negative or zero horsepower)
    if (df['price'] <= 0).any():
        raise ValueError("Price cannot be negative or zero.")
    if (df['horsepower'] <= 0).any():
        raise ValueError("Horsepower must be greater than zero.")

    # 5. Return features (x) and target (y)
    x = df[['horsepower']]
    y = df['price']

    return x, y     