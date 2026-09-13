import pytest
import pandas as pd
from src.data_pipeline import clean_and_variable_data

# TEST 1: Verify standard, perfectly clean data returns correct outputs
def test_standard_valid_data():
    # 1. ARRANGE: Create a valid DataFrame
    valid_df = pd.DataFrame({
        'horsepower': [100, 150, 200],
        'price': [12000, 18000, 25000]
    })

    # 2. ACT: Call the cleaning function
    x, y = clean_and_variable_data(valid_df)

    # 3. ASSERT: Verify row count match and columns are correct
    assert len(x) == 3
    assert len(y) == 3
    assert 'horsepower' in x.columns

# TEST 2: Test the lowest possible valid horsepower boundary (e.g., exactly 1)
def test_minimum_horsepower_boundary():
    # 1. ARRANGE
    min_hp_df = pd.DataFrame({
        'horsepower': [1],
        'price': [500]
    })

    # 2. ACT
    x, y = clean_and_variable_data(min_hp_df)

    # 3. ASSERT
    assert len(x) == 1
    # the x.iloc[0]['horsepower'] is the first row of the 'horsepower' column in the x DataFrame
    assert x.iloc[0]['horsepower'] == 1

# TEST 3: Test the lowest possible valid price boundary (e.g., exactly 1)
def test_minimum_price_boundary():
    # 1. ARRANGE
    min_price_df = pd.DataFrame({
        'horsepower': [500],
        'price': [1]
    })

    # 2. ACT
    x, y = clean_and_variable_data(min_price_df)

    # 3. ASSERT
    assert len(y) == 1
    assert y.iloc[0] == 1

# TEST 4: Verify rows with NaN (null) values are successfully dropped.
def test_drop_rows_with_null_values():
    # 1. ARRANGE: Create an empty Dataframe
    null_df = pd.DataFrame({
        # The null values are added randomly to test the dropping of rows with NaN values
        'horsepower': [100, None, 200], # The second row has a null value for horsepower
        'price': [12000, 18000, None] # The third row has a null value for price
    })

    # 2. ACT: Call the cleaning function
    x, y = clean_and_variable_data(null_df)

    # 3. ASSERT: Verify that the rows with null values are dropped
    assert len(x) == 1  # Only the first row should remain
    assert len(y) == 1 # Only the first row should remain

# TEST 5: Ensure negative horsepower triggers a value error
def test_negative_horsepower_error():
    # 1. ARRANGE: Create a Dataframe with negative horsepower
    neg_hp_df = pd.DataFrame({
        'horsepower': [-1],
        'price': [1000]
    })

    # 2. ACT: Expect a ValueError to be raised when calling the cleaning function
    x, y = None, None
    # excinfo is a context manager that captures the exception raised within its block. It allows us to assert that the correct exception is raised and to inspect the exception message.
    with pytest.raises(ValueError) as excinfo:
        x, y = clean_and_variable_data(neg_hp_df)

    # 3. ASSERT: Verify the error message
    assert str(excinfo.value) == "Horsepower must be greater than zero."

# TEST 6: Ensure exactly zero horsepower triggers a value error
def test_zero_horsepower_error():
    # 1. ARRANGE: Create a Dataframe with ZERO horsepower
    zero_hp_df = pd.DataFrame({
        'horsepower': [0],
        'price': [100]
    })

    # 2. ACT: Expect a ValueError to be raised when calling the cleaning function
    x, y = None, None
    with pytest.raises(ValueError, match="Horsepower must be greater than zero.") as excinfo:
        x, y = clean_and_variable_data(zero_hp_df)

    # 3. ASSERT: Verify the error message
    assert str(excinfo.value) == "Horsepower must be greater than zero."

# TEST 7: Ensure a negative price triggers a value error
def test_negative_price_error():
    # 1. ARRANGE: Create a dataframe with a negative price
    neg_price_df = pd.DataFrame({
        'horsepower': [100],
        'price': [-100]
    })

    # 2. ACT: Expect a ValueError to be raised when calling the cleaning function
    x, y = None, None
    with pytest.raises(ValueError, match="Price cannot be negative or zero.") as excinfo:
        x, y = clean_and_variable_data(neg_price_df)

    # 3. ASSERT: Verify the error message
    assert str(excinfo.value) == "Price cannot be negative or zero."

# TEST 8: Ensure exactly zero price triggers a value error
def test_zero_price_error():
    # 1. ARRANGE: Create a dataframe with an empty price
    zero_price_df = pd.DataFrame({
        'horsepower': [100],
        'price': [0]
    })

    # 2. ACT: Expect a ValueError to be raised when calling the cleaning function
    x, y = None, None
    with pytest.raises(ValueError, match="Price cannot be negative or zero.") as excinfo:
        x, y = clean_and_variable_data(zero_price_df)

    # 3. ASSERT: Verify the error message
    assert str(excinfo.value) == "Price cannot be negative or zero."

# TEST 9: Ensure an empty dataframe triggers a value error
def test_empty_dataframe_error():
    # 1. ARRANGE: create an empty dataframe
    empty_df = pd.DataFrame({
        'horsepower': [],
        'price': []
    })

    # 2. ACT: Expect a ValueError to be raised when calling the cleaning function
    x, y = None, None
    with pytest.raises(ValueError, match="Ingested dataset contains zero records.") as excinfo:
        x, y = clean_and_variable_data(empty_df)

    # 3. ASSERT: Verify the error message
    assert str(excinfo.value) == "Ingested dataset contains zero records."

# TEST 10: Catch a string (e.g., One Hundred") placed in the horsepower column
def test_string_in_horsepower_column():
    # 1. ARRANGE: Create a DataFrame with a string in the horsepower column
    hp_price_df = pd.DataFrame({
        'horsepower': ['One Hundred'],
        'price': [100000]
    })

    # 2. ACT: Expect a TypeError to be raised when calling the cleaning function
    x, y = None, None
    with pytest.raises(TypeError, match="Horsepower and price columns must be numeric.") as excinfo:
        x, y = clean_and_variable_data(hp_price_df)

    # 3. ASSERT: Verify the error message
    assert str(excinfo.value) == "Horsepower and price columns must be numeric."

# TEST 11: Catch a string (e.g., Ten Thousand) placed in the price column
def test_string_in_price_column():
    # 1. ARRANGE: Create a dataframe with a string in the price column
    string_price_df = pd.DataFrame({
        'horsepower': [100],
        'price': ['Ten Thousand']
    })

    # 2. ACT: Expect a TypeError to be raised when calling the cleaning function
    x, y = None, None
    with pytest.raises(TypeError, match="Horsepower and price columns must be numeric.") as excinfo:
        x, y = clean_and_variable_data(string_price_df)

    # 3. ASSERT: Verify the error message
    assert str(excinfo.value) == "Horsepower and price columns must be numeric."

# TEST 12: Catch a boolean (e.g., True) placed in the horsepower column
def test_boolean_in_horsepower_column():
    # 1. ARRANGE: Create a dataframe with a boolean value in the horsepower column
    bool_hp_df = pd.DataFrame({
        'horsepower': [True],
        'price': [1000]
    })

    # 2. ACT: Expect a TypeError to be raised when calling the cleaning function
    x, y = None, None
    with pytest.raises(TypeError, match="Horsepower and price columns must be numeric.") as excinfo:
        x, y = clean_and_variable_data(bool_hp_df)

    # 3. ASSERT: Verify the error message
    assert str(excinfo.value) == "Horsepower and price columns must be numeric."

# TEST 13: Catch a boolean (e.g., False) placed in the price column
def test_boolean_in_price_column():
    # 1. ARRANGE: Create a dataframe with a boolean value in the price column
    bool_price_df = pd.DataFrame({
        'horsepower': [100],
        'price': [False]
    })

    # 2. ACT: Expect a TypeError to be raised when calling the cleaning function
    x, y = None, None
    with pytest.raises(TypeError, match="Horsepower and price columns must be numeric.") as excinfo:
        x, y = clean_and_variable_data(bool_price_df)

    # 3. ASSERT: Verify the error message
    assert str(excinfo.value) == "Horsepower and price columns must be numeric."

# TEST 14: Catch a boolean (e.g., False) placed in the horsepower column
def test_boolean_in_horsepower():
    # 1. ARRANGE: Create a dataframe with a boolean value in the horsepower column
    bool_hp_df = pd.DataFrame({
        'horsepower': [False],
        'price': [1000]
    })

    # 2. ACT: Expect a TypeError to be raised when calling the cleaning function
    x, y = None, None
    with pytest.raises(TypeError, match="Horsepower and price columns must be numeric.") as excinfo:
        x, y = clean_and_variable_data(bool_hp_df)

    # 3. ASSERT: Verify the error message
    assert str(excinfo.value) == "Horsepower and price columns must be numeric."
