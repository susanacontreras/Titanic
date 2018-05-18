## exec(open("./main_susi.py").read())
import pandas as pd

# Load data
titanic_data = pd.read_csv('train.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

from copy import copy
import numpy as np
### Defining mean absolute error of random forest
def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50),
                                X, y,
                                scoring = 'neg_mean_absolute_error').mean()


## Separating target data from predictor features
titanic_target = titanic_data.Survived
titanic_predictors = titanic_data.drop(['Survived'], axis=1)

###### Data manipulation 1 - Only Non categorical features ######
## Only non object values
titanic_numeric_predictors = titanic_predictors.select_dtypes(exclude=['object'])
## Imputing the data: Removing nans and replacing them with the average of the feature
my_imputer = Imputer()
data_with_imputed_values = my_imputer.fit_transform(titanic_numeric_predictors)
# converting again to cell array (imputer outputs a numpy array)
data_with_imputed_values_cell=pd.DataFrame(data_with_imputed_values)
# to restore the column names
data_with_imputed_values_cell.columns=titanic_numeric_predictors.columns
#Separating test from trainning data
y=titanic_target
X=data_with_imputed_values_cell
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0,test_size=0.25)
#Testing the algorithm
mae_without_categoricals = get_mae(train_X, train_y)

###### Data manipulation 2 - Categorical and non Categorical features ######
## All values
titanic_predictors_withFloatAsCategory = pd.get_dummies(titanic_predictors)
## Imputing the data: Removing nans and replacing them with the average of the feature
my_imputer = Imputer()
data_with_imputed_values = my_imputer.fit_transform(titanic_predictors_withFloatAsCategory)
# converting again to cell array (imputer outputs a numpy array)
data_with_imputed_values_cell=pd.DataFrame(data_with_imputed_values)
# to restore the column names
data_with_imputed_values_cell.columns=titanic_predictors_withFloatAsCategory.columns
#Separating test from trainning data
y=titanic_target
X=data_with_imputed_values_cell
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0,test_size=0.25)
#Testing the algorithm
mae_with_categoricals = get_mae(train_X, train_y)

###### Data manipulation 3 - Categorical and non Categorical features plus feature engineering ######
## Select features that have too many categories
categorical_features = titanic_predictors.select_dtypes(include=['object'])
relevant_categorical_features_1=[]

for i_feature in categorical_features.columns:
    if categorical_features[i_feature].describe()[1]<=5:
        relevant_categorical_features_1.append(i_feature)

## All values
titanic_predictors_withFloatAsCategory_relevant1 = pd.get_dummies(titanic_predictors[relevant_categorical_features_1+list(titanic_numeric_predictors.columns)])
## Imputing the data: Removing nans and replacing them with the average of the feature
my_imputer = Imputer()
data_with_imputed_values = my_imputer.fit_transform(titanic_predictors_withFloatAsCategory_relevant1)
# converting again to cell array (imputer outputs a numpy array)
data_with_imputed_values_cell=pd.DataFrame(data_with_imputed_values)
# to restore the column names
data_with_imputed_values_cell.columns=titanic_predictors_withFloatAsCategory_relevant1.columns
#Separating test from trainning data
y=titanic_target
X=data_with_imputed_values_cell
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0,test_size=0.25)
#Testing the algorithm
mae_with_categoricals_relevant1 = get_mae(train_X, train_y)

###### Data manipulation 4 - Categorical and non Categorical features plus feature engineering ######
## Add features based on engineering of features

categorical_features_plusEng = copy(titanic_predictors.select_dtypes(include=['object']))
### Adding cabin class feature
vv_cabin=[]
is_nan_cabin=categorical_features_plusEng.Cabin.isnull()
c_cabin=0
for ii_cabin in categorical_features_plusEng.Cabin:
    cabin=copy(ii_cabin)
    if is_nan_cabin[c_cabin]:
        vv_cabin.append(cabin)
    else:
        ### Adding just a b or c
        st=str(cabin)
        vv_cabin.append(st[0])
    c_cabin+=1

titanic_predictors_plusEng=copy(titanic_predictors)
titanic_predictors_plusEng['Cabin_Class']=vv_cabin
## All values
titanic_predictors_withFloatAsCategory_relevant2 = pd.get_dummies(titanic_predictors_plusEng[list(categorical_features_plusEng.columns)+list(titanic_numeric_predictors.columns)])
## Imputing the data: Removing nans and replacing them with the average of the feature
my_imputer = Imputer()
data_with_imputed_values = my_imputer.fit_transform(titanic_predictors_withFloatAsCategory_relevant2)
# converting again to cell array (imputer outputs a numpy array)
data_with_imputed_values_cell=pd.DataFrame(data_with_imputed_values)
# to restore the column names
data_with_imputed_values_cell.columns=titanic_predictors_withFloatAsCategory_relevant2.columns
#Separating test from trainning data
y=titanic_target
X=data_with_imputed_values_cell
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0,test_size=0.25)
#Testing the algorithm
mae_with_categoricals_relevant2 = get_mae(train_X, train_y)


print('Absolute error without categorical features: '+str(mae_without_categoricals))
print('Absolute error with categorical features: '+str(mae_with_categoricals))
print('Absolute error with categorical features only with "relevant" categories: '+str(mae_with_categoricals_relevant1))
print('Absolute error with categorical features only with "relevant" categories + class:  '+str(mae_with_categoricals_relevant2))
