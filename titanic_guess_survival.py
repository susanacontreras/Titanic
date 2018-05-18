import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np


def basic_forest_no_categories(data):
    """Use RandomForestRegression on numerical features only"""
    # exclude non-numerical data
    original_data = data.select_dtypes(exclude='object')
    # print(original_data.columns)

    y = original_data[['Survived']]
    # predictors=data[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
    predictors = original_data.drop(['Survived'], axis=1)
    # print(predictors.shape)

    # split data into training and validation data, for both predictors and target
    train_X, test_X, train_y, test_y = train_test_split(predictors, y, test_size=0.1, random_state = 0)

    # exclude missing values
    cols_with_missing = [col for col in train_X.columns
                                     if train_X[col].isnull().any()]
    reduced_X_train = train_X.drop(cols_with_missing, axis=1)
    reduced_X_test = test_X.drop(cols_with_missing, axis=1)

    titanic_model_split = RandomForestRegressor()
    print(titanic_model_split)
    titanic_model_split.fit(reduced_X_train, np.ravel(train_y))
    error_tree = mean_absolute_error(test_y, titanic_model_split.predict(reduced_X_test))
    return error_tree


def basic_forest_add_categorical(data, use_cabin=False, impute=False, test_dataset=False):
    """Use RandomForestRegression on all relevant features, substitute object into one-hot encodings. Decide if use
    cabin data (first letter only), or not"""

    y = data[['Survived']]
    if use_cabin:
        predictors = data.drop(['Survived', 'Name', 'Ticket'], axis=1)
        # of cabin info, only take the level (letter) and not the number of the cabin
        mycopy = predictors.copy()  # copy issue, does not update
        for idx, el in enumerate(mycopy.Cabin):
            if not pd.isna(el):
                mycopy.Cabin[idx] = str(el)[0]
        # function to substitute object into one-hot encodings
        one_hot_encoded_training_predictors = pd.get_dummies(mycopy)

    else:
        predictors = data.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
        # function to substitute object into one-hot encodings
        one_hot_encoded_training_predictors = pd.get_dummies(predictors)

    # split data into training and validation data, for both predictors and target
    print(one_hot_encoded_training_predictors.columns)
    train_X, test_X, train_y, test_y = train_test_split(one_hot_encoded_training_predictors, y, test_size=0.1,
                                                        random_state=0)

    # exclude missing values
    if impute:
        # 1 - impute
        my_imputer = Imputer()
        reduced_X_train = my_imputer.fit_transform(train_X)
        reduced_X_test = my_imputer.fit_transform(test_X)

    else:
        # 2 - remove missing values
        cols_with_missing = [col for col in one_hot_encoded_training_predictors.columns
                                         if one_hot_encoded_training_predictors[col].isnull().any()]
        reduced_X_train = train_X.drop(cols_with_missing, axis=1)
        reduced_X_test = test_X.drop(cols_with_missing, axis=1)

    titanic_model_split = RandomForestRegressor()
    titanic_model_split.fit(reduced_X_train, np.ravel(train_y))

    error_tree = mean_absolute_error(test_y, titanic_model_split.predict(reduced_X_test))


    if not test_dataset:
        return error_tree
    else:
        # Read the test data
        test_filename = 'test.csv'
        test_data = pd.read_csv(test_filename)

        # Treat the test data in the same way as training data. In this case, pull same columns.
        test_X = test_data[col_predictiors]
        predicted_prices = titanic_model_split.predict(test_X)
        print(predicted_prices)
        # submit your results!
        my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
        # you could use any filename. We choose submission here
        my_submission.to_csv('submission.csv', index=False)



if __name__ == '__main__':

    train_filename = 'train.csv'
    data = pd.read_csv(train_filename)
    # NOTE: problem with categorical variables in plotting
    # NOTE: to add new columns to the DataFrame, add them like dic, e.g. mydata['sex'] = sex
    # NOTE: use chisquare to test categorical variables dependence
    # print(data.describe())

    error_basic = basic_forest_no_categories(data)
    error_no_cabin = basic_forest_add_categorical(data, use_cabin=False)
    error_cabin = basic_forest_add_categorical(data, use_cabin=True)
    print('Errors: basic', error_basic, 'no cabin:', error_no_cabin, 'cabin:', error_cabin)