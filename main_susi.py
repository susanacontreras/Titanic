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

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50),
                                X, y,
                                scoring = 'neg_mean_absolute_error').mean()


## Separating target data from predictor features
titanic_target = titanic_data.Survived
titanic_predictors = titanic_data.drop(['Survived'], axis=1)

###### Data manipulation 1 ######
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
