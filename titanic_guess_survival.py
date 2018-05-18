import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np


def basic_forest_no_categories(data):

    # exclude objects
    original_data = data.select_dtypes(exclude='object')  # exclude non-numerical data
    print(original_data.columns)

    y = original_data[['Survived']]
    # predictors=data[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
    predictors = original_data.drop(['Survived'], axis=1)
    print(predictors.shape)

    # split data into training and validation data, for both predictors and target
    # The split is based on a random number generator. Supplying a numeric value to
    # the random_state argument guarantees we get the same split every time we
    # run this script.
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
    print(error_tree)


def basic_forest_add_categorical(data):

    y = data[['Survived']]
    predictors = data.drop(['Survived','Name','Ticket','Cabin'], axis=1)
    # cols_with_missing = [col for col in predictors.columns
    #                                  if predictors[col].isnull().any()]
    # reduced_predictors = predictors.drop(cols_with_missing, axis=1)
    # print(reduced_predictors.dtypes)

    # of cabin info, only take the level (letter) and not the number of the cabin
    # print(predictors.Cabin, 'kioi')
    # auy = [str(el)[0] for el in reduced_predictors.Cabin]
    # print(auy[:10])

    # split data into training and validation data, for both predictors and target
    # The split is based on a random number generator. Supplying a numeric value to
    # the random_state argument guarantees we get the same split every time we
    # run this script.
    one_hot_encoded_training_predictors = pd.get_dummies(predictors)  # function to substitute object into one-hot encodings
    print(one_hot_encoded_training_predictors.columns)
    train_X, test_X, train_y, test_y = train_test_split(one_hot_encoded_training_predictors, y, test_size=0.1, random_state = 0)

    # exclude missing values
    cols_with_missing = [col for col in one_hot_encoded_training_predictors.columns
                                     if one_hot_encoded_training_predictors[col].isnull().any()]
    reduced_X_train = train_X.drop(cols_with_missing, axis=1)
    reduced_X_test = test_X.drop(cols_with_missing, axis=1)

    titanic_model_split = RandomForestRegressor()
    titanic_model_split.fit(reduced_X_train, np.ravel(train_y))

    error_tree = mean_absolute_error(test_y, titanic_model_split.predict(reduced_X_test))
    print(error_tree)



if __name__ == '__main__':

    train_filename = 'train.csv'
    test_filename = 'test.csv'
    data = pd.read_csv(train_filename)
    # NOTE: problem with categorical variables in plotting

    # print(data.describe())

    # basic_forest_no_categories(data)
    basic_forest_add_categorical(data)