import pandas as pd
import matplotlib.pyplot as plt
import math as math
import numpy as np
from scipy.stats import chisquare
from scipy import stats


# import training data
filepath = 'train.csv'
data = pd.read_csv(filepath)
print(data.describe())

# look into sex feature
survived_sex = pd.crosstab(index=data["Survived"],
                           columns=data["Sex"], margins=True)

print(survived_sex)
# print chisquared test of independence between sex and survival
print(chisquare(f_obs = [81, 468, 233, 109], f_exp = [193.47, 355.53, 120.5, 221.47]))

# look into fare feature
fares_survivors = data.Fare[data.Survived == True]
fares_died = data.Fare[data.Survived == False]
print(stats.ttest_ind(fares_survivors,fares_died, equal_var=False))
print(fares_survivors.mean())
print(fares_died.mean())

# plotting boxplots of age, pclass, sibsp and fare
plt.figure()
boxplot = data.boxplot(column = ['Age', 'Pclass', 'SibSp', 'Fare'])

# passengers with a higher fare were more likely to survive

# could also look into name and extract titles
print(data.Name)

# feature engineering: make a new variable of total family members
familymembers = data.SibSp + data.Parch
print(familymembers)

data_copied = data.copy()
data_copied['fammembers'] = familymembers

plt.figure()
boxplot = data_copied.boxplot(column = ['fammembers'])

survivor_fammembers = data_copied.fammembers[data_copied.Survived == True]
died_fammembers = data_copied.fammembers[data_copied.Survived == False]


# plot family members of surviving passengers
df_fammembers_surv = pd.DataFrame(data = survivor_fammembers)
plt.figure()
boxplot = df_fammembers_surv.boxplot(column = ['fammembers'])

# plot family members of dead passengers
df_fammembers_died = pd.DataFrame(data = died_fammembers)
plt.figure()
boxplot = df_fammembers_died.boxplot(column = ['fammembers'])

print(stats.ttest_ind(survivor_fammembers,died_fammembers, equal_var=True))
print(survivor_fammembers.mean())
print(died_fammembers.mean())

# add 1 (minimum value) to be able to take logs later
survivor_fammembers_ed = survivor_fammembers + 1
died_fammembers_ed = died_fammembers + 1

# add logarithm of family members to survival dataframe
df_fammembers_surv['logfammembers'] = np.log(np.array(survivor_fammembers_ed))

print(df_fammembers_surv.head())

# add logarithm of family members to dead dataframe

df_fammembers_died['logfammembers'] = np.log(np.array(died_fammembers_ed))

print(df_fammembers_died.head())

# t test between the logarithms
print(stats.ttest_ind(df_fammembers_surv['logfammembers'],df_fammembers_died['logfammembers'], equal_var=False))


# plot logs of family members
plt.figure()
boxplot = df_fammembers_surv.boxplot(column = ['logfammembers'])

#df_fammembers_died = pd.DataFrame(data = died_fammembers)

plt.figure()
boxplot = df_fammembers_died.boxplot(column = ['logfammembers'])

print(survivor_fammembers_ed.mean())
print(died_fammembers_ed.mean())
plt.show()



#print(df_fammembers)