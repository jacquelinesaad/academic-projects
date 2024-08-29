"""
    Name: Jacqueline Saad
    Email: jacqueline.saad05@myhunter.cuny.edu
    Resources: 10 minutes to Pandas, A Gentle Introduction to Pandas, Geeks for Geeks
"""
import pandas as pd

def import_data(file_name):
    '''Reads the CSV file into a DataFrame.'''
    df = pd.read_csv(file_name)

    # Select only the specified columns.
    columns_to_keep = ['dbn', 'school_name', 'NTA', 'graduation_rate', 'pct_stu_safe',
                       'attendance_rate', 'college_career_rate', 'language_classes', 
                       'advancedplacement_courses', 'method1', 'overview_paragraph']
    df = df[columns_to_keep]

    # Data cleaning; drop rows with missing values in the 'graduation_rate' column.
    df.dropna(subset=['graduation_rate'], inplace=True)

    return df

def impute_numeric_cols(df):
    '''Fill missing values in numeric columns with the median of the respective column.'''
    numeric_cols = ['pct_stu_safe', 'attendance_rate', 'college_career_rate']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

def compute_count_col(df, col):
    '''Split and count the comma-separated values in each entry and return the resulting series.'''
    counts = df[col].str.split(',').apply(len)
    return counts

def encode_categorical_col(col):
    '''Use pandas get_dummies function for categorical encoding and sort columns.'''
    encoded_df = pd.get_dummies(col, prefix='', prefix_sep='')
    # Convert boolean values to integers(binary encoding 0s and 1s).
    encoded_df = encoded_df.astype(int)
    encoded_df = encoded_df.reindex(sorted(encoded_df.columns), axis=1)
    encoded_df = encoded_df.iloc[:, :-1]
    return encoded_df

def split_test_train(df, xes_col_names, y_col_name, frac=0.25, random_state=922):
    '''Create a test set using panda's sample function'''
    df_test = df.sample(frac=frac, random_state=random_state)

    # Create a copy of the DataFrame for training.
    df_train = df.copy()

    # Drop the rows in df_test from df_train.
    df_train = df_train.drop(df_test.index)

    # Split the DataFrame into x (independent variables) and y (dependent variables).
    x_train = df_train[xes_col_names]
    x_test = df_test[xes_col_names]
    y_train = df_train[y_col_name]
    y_test = df_test[y_col_name]

    return x_train, x_test, y_train, y_test

def compute_lin_reg(xes, yes):
    '''Linear Regression Model; computes slope and y-intercept of the linear regression line.'''
    # Calculate the standard deviation of the xes and yes lists and store them in sd_x and sd_y.
    sd_x = (sum((x - sum(xes) / len(xes))**2 for x in xes) / (len(xes) - 1))**0.5
    sd_y = (sum((y - sum(yes) / len(yes))**2 for y in yes) / (len(yes) - 1))**0.5

    # Compute the correlation coefficient, r, between xes and yes.
    mean_x = sum(xes) / len(xes)
    mean_y = sum(yes) / len(yes)
    r = sum((x - mean_x) * (y - mean_y) for x, y in zip(xes, yes)) / ((len(xes) - 1) * sd_x * sd_y)

    # Compute the slope, theta_1, as theta_1 = r*sd_y/sd_x.
    # Compute the y-intercept, theta_0, as theta_0 = average(yes) - theta_1 * average(xes).
    theta_1 = r * sd_y / sd_x
    theta_0 = mean_y - theta_1 * mean_x

    return theta_0, theta_1

def predict(xes, theta_0, theta_1):
    '''Predicts the dependent values based on the linear regression model.'''

    # Initialize an empty list to store the predicted values.
    predictions = []

    # Calculate the predicted values for each x in xes.
    for x in xes:
        y_predicted = theta_0 + theta_1 * x
        predictions.append(y_predicted)

    return predictions

def mse_loss(y_actual, y_estimate):
    '''Calculate the mean squared error loss between two series.'''

    # Check if the input series have the same length.
    if len(y_actual) != len(y_estimate):
        raise ValueError("Series must be the same length")

    # Calculate the mean squared error.
    n = len(y_actual)
    squared_errors = [(y_actual[i] - y_estimate[i]) ** 2 for i in range(n)]
    mean_squared_error = sum(squared_errors) / (n - 1)  # Use (n - 1) for sample mean squared error.

    return mean_squared_error

def rmse_loss(y_actual, y_estimate):
    '''Calculate the square root mean squared error using the mse_loss function.'''
    mean_squared_error = mse_loss(y_actual, y_estimate)

    # Calculate the square root of the mean squared error to get RMSE.
    root_mean_squared_error = (mean_squared_error) ** 0.5

    return root_mean_squared_error

def compute_error(y_actual, y_estimate, loss_fnc=mse_loss):
    '''Calculate the error using the specified loss function.'''
    error = loss_fnc(y_actual, y_estimate)

    return error

def test_compute_count_col(compute_fnc=compute_count_col):
    '''Test function, compute_fnc returns True if correct (e.g. computes the the count column).'''
    # Sample data for testing.
    data = {'Column1': ['A, B, C', 'B, C, D, E', 'A, E, F']}

    # Expected result.
    expected_result = pd.Series([3, 4, 3])

    # Create a DataFrame from the sample data.
    df = pd.DataFrame(data)

    # Call the compute_fnc with the DataFrame and the column name.
    result = compute_fnc(df, 'Column1')

    # Check if the actual result matches the expected result.
    if result.equals(expected_result):
        return True
    else:
        return False

# Test the compute_count_col function.
if test_compute_count_col():
    print("compute_count_col test passed!")
else:
    print("compute_count_col test failed!")

def test_predict(predict_fnc=predict):
    '''Test function, predict_fnc returns True if correct (e.g. predicts the correct values).'''
    # Sample data for testing.
    x_values = pd.Series([1, 2, 3, 4, 5])
    theta_0 = 2.0
    theta_1 = 1.5

    # Expected result.
    expected_result = pd.Series([3.5, 5.0, 6.5, 8.0, 9.5])

    # Call the predict_fnc with the sample data and parameters.
    result = predict_fnc(x_values, theta_0, theta_1)

    # Check if the actual result matches the expected result.
    if result.equals(expected_result):
        return True
    else:
        return False

# Test the predict function.
if test_predict():
    print("predict test passed!")
else:
    print("predict test failed!")

def test_mse_loss(loss_fnc=mse_loss):
    '''Test function, loss_fnc returns True if correct (e.g. computes MSA).'''
    # Sample data for testing.
    y_actual = pd.Series([3, 5, 7, 9, 11])
    y_estimate = pd.Series([2.8, 4.9, 7.2, 8.8, 11.2])

    # Expected result.
    expected_result = 0.134

    # Call the loss_fnc with the sample data.
    result = loss_fnc(y_actual, y_estimate)

    # Check if the actual result is approximately equal to the expected result.
    tolerance = 0.001  # Define a small tolerance for floating-point comparisons.
    if abs(result - expected_result) < tolerance:
        return True
    else:
        return False

# Test the mse_loss function.
if test_mse_loss():
    print("mse_loss test passed!")
else:
    print("mse_loss test failed!")
