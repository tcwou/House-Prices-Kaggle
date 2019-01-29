# noinspection PyUnresolvedReferences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

pd.options.mode.chained_assignment = None  # default='warn'

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#explore data
#data.info()
#There is an id column and several variables with many missing values

#drop irrelevant id columns
test_id = test['Id']
data = data.drop(columns=['Id'])
test = test.drop(columns=['Id'])
#Look for any columns with only one unique value
for column in data:
    if len(data[column].unique()) <= 1:
        print(column)
#No column names returned, so all columns have more than one unique value

#Examine correlation matrix to identify highly correlated variables
"""
correlations = data.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(correlations, annot=True)
plt.show()
"""


#0.88 between GarageCars and GarageArea. Remove Garagecars
#0.83 between YearBuilt and GarageYrBlt. Remove GarageYrBlt.
#0.78 between GrLivArea and TotRmsAbvGrd. Remove TotRmsAbvGrd.


data = data.drop(columns=['GarageCars', 'GarageYrBlt', 'TotRmsAbvGrd'])
test = test.drop(columns=['GarageCars', 'GarageYrBlt', 'TotRmsAbvGrd'])

#identify any outliers in numerical features
numeric_data = data.select_dtypes(include=[np.number])
print(numeric_data.columns)

"""
f, ax = plt.subplots(6, 6, figsize=(100,50))
for row in range(6):
    for col in range(6):
        idx = col+(6*row)
        if idx < 34:
            x=numeric_data.iloc[:, idx]
            y=numeric_data['SalePrice']
            ax[row, col].scatter(x, y, s=5)
            ax[row, col].set_title(numeric_data.columns[idx])
            ax[row, col].set_yticklabels([])
f.delaxes(ax[5, 3])
f.delaxes(ax[5, 4])
f.delaxes(ax[5, 5])
plt.subplots_adjust(hspace=0.8)
plt.show()
"""

outlier_features = ['LotFrontage', 'LotArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea']
outlier_data = data.copy()

zscore_cols=[]
# Calculate Z-score for each feature of interest
for col in outlier_features:
    col_zscore = col + '_zscore'
    zscore_cols += [col_zscore]
    outlier_data[col_zscore] = (outlier_data[col] - outlier_data[col].mean())/outlier_data[col].std(ddof=0)

# Get index values of points with z-score >= 5 for any feature



#The plots of numeric features show possible outliers in LotFrontage, LotArea, BsmtFinSF1 TotalBsmtSF, 1stFlrSF, GrLivArea
#Isolate these points and investigate these points further

lotfrontage_outliers = numeric_data[numeric_data['LotFrontage']>200]
#print(lotfrontage_outliers)
#row 934, 1298
lotarea_outliers = numeric_data[numeric_data['LotArea']>100000]
#print(lotarea_outliers)
#row 249, 313, 335, 706
bsmtfinsf1_outliers = numeric_data[numeric_data['BsmtFinSF1']>4000]
#print(bsmtfinsf1_outliers)
#1298
totalbsmtsf_outliers = numeric_data[numeric_data['TotalBsmtSF']>6000]
#print(totalbsmtsf_outliers)
#1298
firstflrsf_outliers = numeric_data[numeric_data['1stFlrSF']>4000]
#print(firstflrsf_outliers)
#1298
grlivarea_outliers = numeric_data[numeric_data['GrLivArea']>4000]
#print(grlivarea_outliers)
#523, 691, 1182, 1298

#The outlier in row 1298 appears in five of the six examined features. The validity of this feature is dubious.
#With further investigation of the data source the author states that there are there are three true partial-sale outliers.
#The author suggests removing all GrLivArea > 4000 sqft, which includes row #1298 that has questionable validity.

#Drop GrLivArea > 4000 and replot graphs
numeric_data = numeric_data[numeric_data['GrLivArea']<=4000]
"""
f, ax = plt.subplots(6, 6, figsize=(100,50))
for row in range(6):
    for col in range(6):
        idx = col+(6*row)
        if idx < 34:
            x=numeric_data.iloc[:, idx]
            y=numeric_data['SalePrice']
            ax[row, col].scatter(x, y, s=5)
            ax[row, col].set_title(numeric_data.columns[idx])
            ax[row, col].set_yticklabels([])
f.delaxes(ax[5, 3])
f.delaxes(ax[5, 4])
f.delaxes(ax[5, 5])
plt.subplots_adjust(hspace=0.8)
plt.show()
"""
#The replotted graphs show a tighter distribution for BsmtFinSF1, TotalBsmtSF 1stFlrSF and GrLivArea
#There is still one anomalous point for LotFrontage, however without the ability to scrutinize the source of the data, we assume it is valid
data = data[data['GrLivArea']<4000]

#Plot the distribution of all numeric features to look for required data transformations
"""
f, ax = plt.subplots(6, 6, figsize=(100,50))
for row in range(6):
    for col in range(6):
        idx = col+(6*row)
        if idx < 34:
            x=numeric_data.iloc[:, idx]
            ax[row, col].hist(x, bins=25)
            ax[row, col].set_title(numeric_data.columns[idx])
            #ax[row, col].set_yticklabels([])
f.delaxes(ax[5, 3])
f.delaxes(ax[5, 4])
f.delaxes(ax[5, 5])
plt.subplots_adjust(hspace=0.8)
plt.show()
"""

#Missing Values
#count missing values
"""
for col in data:
    num_missing = len(test) - data[col].count()
    if num_missing > 0:
        print(col, num_missing, str(round(num_missing/len(data)*100))+ '%')

print()
print()

for col in test:
    num_missing = len(test) - test[col].count()
    if num_missing > 0:
        print(col, num_missing, str(round(num_missing/len(test)*100))+ '%')
"""

#impute missing values

#LotFrontage 259 18.0% - continuous (Linear feet of street connected to property)
#MasVnrType 8 1.0% - categorical
#MasVnrArea 8 1.0% - continuous (masonary veneer area)
#BsmtQual 37 3.0% - categorical numeric bins (consider converting to mean of bin)
#BsmtCond 37 3.0% - categorical (NA = not present)
#BsmtExposure 38 3.0% - categorical (NA = not present)
#BsmtFinType1 37 3.0% - categorical (NA = not present)
#BsmtFinType2 38 3.0% - categorical (NA = not present)
#Electrical 1 0.0% - categorical (use mode)
#FireplaceQu 690 47.0% - categorical (NA = not present)
#GarageType 81 6.0% - categorical (NA = not present)
#GarageFinish 81 6.0% - categorical (NA = not present)
#GarageQual 81 6.0% 0 - categorical (NA = not present)
#GarageCond 81 6.0% - categorical (NA = not present)
#PoolQC 1451 100.0% - categorical (NA = not present)
#Fence 1176 81.0% - categorical (NA = not present)
#MiscFeature 1402 96.0% - categorical (NA = not present)
#Alley 1365 94.0% - categorical (NA = not present)

#Upon inspection of the data, for most categorical features, NA does not indicate missing data, but rather that the feature is not present
#For example PoolQC = NA means that the house does not have a pool.
#For these categorical columns change NA to "noFeature"

for feature in ['Alley', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:
    data[feature] = data[feature].fillna('NotPresent')
    test[feature] = test[feature].fillna('NotPresent')


#For remaining categorical data use mode value
for feature in ['MasVnrType', 'Electrical', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']:
    value = data[feature].value_counts().keys()[0]
    data[feature] = data[feature].fillna(value)
    test[feature] = test[feature].fillna(value)

#For MasVnrArea since the missing rows have MasVnrType as None, the Area should be zero
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)

#For BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath if there is no basment, these missing values should be zero
for feature in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    data.loc[data['BsmtFinType1'] == 'NotPresent', feature] = 0
    test.loc[test['BsmtFinType1']=='NotPresent', feature] = 0

#For GaragaArea if there is no garage the missing value should be zero
data.loc[data['GarageFinish'] == 'NotPresent', 'GarageArea'] = 0
test.loc[test['GarageFinish']== 'NotPresent', 'GarageArea'] = 0

#For LotFrontage, impute variables based on linear regression of most correlated variables
#From correlation matrix, LotArea and 1stFlrSF are the most correlated variables

lotdata = data[['LotArea', '1stFlrSF', 'LotFrontage']]
lotdata = lotdata.dropna()

x = lotdata[['LotArea', '1stFlrSF']]
y = lotdata['LotFrontage']

#x = x.values.reshape(-1, 1)
y = y.values.reshape(-1, 1)
all_x = data[['LotArea', '1stFlrSF']]
test_all_x = test[['LotArea', '1stFlrSF']]

linear_lotfrontage = LinearRegression()
linear_lotfrontage.fit(x, y)

linear_pred = linear_lotfrontage.predict(all_x)
test_linear_pred = linear_lotfrontage.predict(test_all_x)
#print(linear_lotfrontage.intercept_, linear_lotfrontage.coef_)

#Residual plot
#plt.scatter(data['LotFrontage'], linear_pred,  color='black')
#plt.scatter(data['LotArea'], data['LotFrontage'],  color='black')
data['LotFrontagePred'] = linear_pred
test['LotFrontagePred'] = test_linear_pred

residuals_linear = data['LotFrontagePred'] - data['LotFrontage']
#print(np.mean(residuals_linear))
#print(residuals_linear.corr(data['LotFrontage']))
"""
f, ax = plt.subplots(2, 2, figsize=(100,50))
ax[0, 0].scatter(lotdata['LotArea'], lotdata['LotFrontage'])
ax[0, 1].scatter(lotdata['1stFlrSF'], lotdata['LotFrontage'])
ax[1, 1].scatter(data['LotFrontage'], residuals_linear)
plt.show()
"""
#-0.66 correlation with lotArea and 1stFlrSF
#Replace missing values with predicted values
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontagePred'])
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontagePred'])
data = data.drop(columns=['LotFrontagePred'])
test = test.drop(columns=['LotFrontagePred'])

#Create new features
#Total baths is the sum of full baths and half baths
#Total house SF is sum of basement SF and above ground living SF
data['TotalBaths'] = data['BsmtFullBath'] + data['FullBath'] + 0.5*(data['BsmtHalfBath'] + data['HalfBath'])
test['TotalBaths'] = test['BsmtFullBath'] + test['FullBath'] + 0.5*(test['BsmtHalfBath'] + test['HalfBath'])

data['TotalSF'] = data['TotalBsmtSF'] + data['GrLivArea']
test['TotalSF'] = test['TotalBsmtSF'] + test['GrLivArea']

data = data.drop(columns=['BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath'])
test = test.drop(columns=['BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath'])

#print(data.groupby(['MoSold'])['SalePrice'].mean())


data = data.replace({'MoSold': {1: 'Q1', 2: 'Q1',3: 'Q1',
4: 'Q2',5: 'Q2',6: 'Q2',7: 'Q3',8: 'Q3',9: 'Q3',10: 'Q4',
11: 'Q4',12: 'Q4'}})

#print(data.groupby(['MoSold'])['SalePrice'].mean())

test = test.replace({'MoSold': {1: 'Q1', 2: 'Q1',3: 'Q1',
4: 'Q2',5: 'Q2',6: 'Q2',7: 'Q3',8: 'Q3',9: 'Q3',10: 'Q4',
11: 'Q4',12: 'Q4'}})


data = data.replace({'MSSubClass': {20: 'class20', 30: 'class30',40: 'class40',
45: 'class45',50: 'class50',60: 'class60',70: 'class70',
75: 'class75',80: 'class80',85: 'class85',90: 'class90',
120: 'class120',150: 'class150',160: 'class160',180: 'class180',
190: 'class190'}})

test = test.replace({'MSSubClass': {20: 'class20', 30: 'class30',40: 'class40',
45: 'class45',50: 'class50',60: 'class60',70: 'class70',
75: 'class75',80: 'class80',85: 'class85',90: 'class90',
120: 'class120',150: 'class150',160: 'class160',180: 'class180',
190: 'class190'}})

data['AgeSold'] = data['YrSold'] - data['YearBuilt']
test['AgeSold'] = test['YrSold'] - test['YearBuilt']

data['Remodel'] = data['YearRemodAdd'] - data['YearBuilt']
test['Remodel'] = test['YearRemodAdd'] - data['YearBuilt']

data.loc[data['Remodel'] > 0, 'Remodel'] = 'Yes'
data.loc[data['Remodel'] == 0, 'Remodel'] = 'No'
test.loc[test['Remodel'] > 0, 'Remodel'] = 'Yes'
test.loc[test['Remodel'] == 0, 'Remodel'] = 'No'

data.loc[data['AgeSold'] == -1, 'AgeSold'] = 0
test.loc[test['AgeSold'] == -1, 'AgeSold'] = 0

data['YrSold'] = data.YrSold.astype(str)
test['YrSold'] = test.YrSold.astype(str)

data = data.drop(columns=['YearBuilt', 'YearRemodAdd'])
test = test.drop(columns=['YearBuilt', 'YearRemodAdd'])
#-0.01293957945616051 AgeSold addition
"""
f, ax = plt.subplots(2, 2, figsize=(100,50))
ax[0, 0].scatter(data['AgeSold'], data['SalePrice'])
ax[0, 1].scatter(data['YearBuilt'], data['SalePrice'])
ax[1, 1].scatter(data['YrSold'], data['SalePrice'])
plt.show()
"""


#Add quadratic terms to correlated numeric features
numeric_data = data.select_dtypes(include=[np.number])
numeric_corr_dict = {}
for col in numeric_data.columns:
    numeric_corr_dict[col] = numeric_data[col].corr(data['SalePrice'])

numeric_corr_dict_sorted = sorted(numeric_corr_dict.items(), key=lambda kv: kv[1], reverse=True)
#for corr in numeric_corr_dict_sorted:
   #print(corr)

numeric_quadratics =  [k for k,v in numeric_corr_dict.items() if abs(v) >= 0.3]

numeric_quadratics.remove('SalePrice')
quadratic_columns = []
for col in numeric_quadratics:
    quadratic_columns += [col+"**2"]
    data[col+"**2"] = data[col]**2
    test[col + "**2"] = test[col] ** 2


numeric_data = data.select_dtypes(include=[np.number])
#print(numeric_data.columns)
#It appears that many features are skewed. We can reduce the skew by applying log transform. We first + 1 in order to deal with points with value 0
#As a rule of thumb if absolute value of skewness is > 0.5, then we should apply the transform.

skewed_table = abs(numeric_data.skew(axis = 0))
skewed_numeric = skewed_table[skewed_table > 0.5].index
#print(skewed_table)
numeric_data[skewed_numeric] = np.log1p(numeric_data[skewed_numeric])

skewed_after = abs(numeric_data.skew(axis = 0))
#print(skewed_after)

#After transformation we see that the skew has been reduced for many features, although some features still show high levels of skew

data[skewed_numeric] = np.log1p(data[skewed_numeric])
#print(skewed_numeric)
skewed_numeric_test = skewed_numeric.drop('SalePrice')
#print(skewed_numeric_test)
test[skewed_numeric_test] = np.log1p(test[skewed_numeric_test])


"""
f, ax = plt.subplots(7, 7, figsize=(100,50))
for row in range(7):
    for col in range(7):
        idx = col+(7*row)
        if idx < 42:
            x=data[skewed_numeric].iloc[:, idx]
            y=data['SalePrice']
            ax[row, col].scatter(x, y, s=5)
            ax[row, col].set_title(data[skewed_numeric].columns[idx])
            ax[row, col].set_yticklabels([])
#f.delaxes(ax[6, 3])
#f.delaxes(ax[6, 4])
#f.delaxes(ax[6, 5])
plt.subplots_adjust(hspace=0.8)
plt.show()
"""



#Inspect all categorical variables to look for ordinal relationships

cat_data = data.select_dtypes(include=['object', 'category'])
cat_data['SalePrice'] = data['SalePrice'].copy()


"""
f, ax = plt.subplots(7, 7, figsize=(100,50))
for row in range(7):
    for col in range(7):
        idx = col+(7*row)
        if idx < 44:
            x=cat_data.iloc[:, idx]
            y=cat_data['SalePrice']
            ax[row, col].scatter(x, y, s=5)
            ax[row, col].set_title(cat_data.columns[idx])
            ax[row, col].set_yticklabels([])
#f.delaxes(ax[6, 3])
#f.delaxes(ax[6, 4])
#f.delaxes(ax[6, 5])
plt.subplots_adjust(hspace=0.8)
plt.show()
"""

#After inspecting the categorical features the following features are ordinal in nature:
ordinal = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
#In order to preserve the ordering information we can code the levels of the feature to a numeric scale


ordinal_1=['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
ordinal_2 = ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'Fence']
def ordinal_coding(df):
    ord_data = df[ordinal]
    for col in ordinal_1:
        try:
            ord_data.loc[ord_data[col] == 'Ex', col] = 5
            ord_data.loc[ord_data[col] == 'Gd', col] = 4
            ord_data.loc[ord_data[col] == 'TA', col] = 3
            ord_data.loc[ord_data[col] == 'Fa', col] = 2
            ord_data.loc[ord_data[col] == 'Po', col] = 1
            ord_data.loc[ord_data[col] == 'NotPresent', col] = 0
        except:
            pass

    ord_data.loc[ord_data['BsmtExposure'] == 'Gd','BsmtExposure'] = 5
    ord_data.loc[ord_data['BsmtExposure'] == 'Av','BsmtExposure'] = 4
    ord_data.loc[ord_data['BsmtExposure'] == 'Mn','BsmtExposure'] = 3
    ord_data.loc[ord_data['BsmtExposure'] == 'No','BsmtExposure'] = 2
    ord_data.loc[ord_data['BsmtExposure'] == 'NotPresent','BsmtExposure'] = 1

    ord_data.loc[ord_data['BsmtFinType1'] == 'GLQ','BsmtFinType1'] = 6
    ord_data.loc[ord_data['BsmtFinType1'] == 'ALQ','BsmtFinType1'] = 5
    ord_data.loc[ord_data['BsmtFinType1'] == 'BLQ','BsmtFinType1'] = 4
    ord_data.loc[ord_data['BsmtFinType1'] == 'Rec','BsmtFinType1'] = 3
    ord_data.loc[ord_data['BsmtFinType1'] == 'LwQ','BsmtFinType1'] = 2
    ord_data.loc[ord_data['BsmtFinType1'] == 'Unf','BsmtFinType1'] = 1
    ord_data.loc[ord_data['BsmtFinType1'] == 'NotPresent','BsmtFinType1'] = 0

    ord_data.loc[ord_data['BsmtFinType2'] == 'GLQ','BsmtFinType2'] = 6
    ord_data.loc[ord_data['BsmtFinType2'] == 'ALQ','BsmtFinType2'] = 5
    ord_data.loc[ord_data['BsmtFinType2'] == 'BLQ','BsmtFinType2'] = 4
    ord_data.loc[ord_data['BsmtFinType2'] == 'Rec','BsmtFinType2'] = 3
    ord_data.loc[ord_data['BsmtFinType2'] == 'LwQ','BsmtFinType2'] = 2
    ord_data.loc[ord_data['BsmtFinType2'] == 'Unf','BsmtFinType2'] = 1
    ord_data.loc[ord_data['BsmtFinType2'] == 'NotPresent','BsmtFinType2'] = 0

    ord_data.loc[ord_data['Functional'] == 'Typ','Functional'] = 7
    ord_data.loc[ord_data['Functional'] == 'Min1','Functional'] = 6
    ord_data.loc[ord_data['Functional'] == 'Min2','Functional'] = 5
    ord_data.loc[ord_data['Functional'] == 'Mod','Functional'] = 4
    ord_data.loc[ord_data['Functional'] == 'Maj1','Functional'] = 3
    ord_data.loc[ord_data['Functional'] == 'Maj2','Functional'] = 2
    ord_data.loc[ord_data['Functional'] == 'Sev','Functional'] = 1
#ord_data.loc[ord_data['Functional'] == 'Sal','Functional'] = 0

    ord_data.loc[ord_data['GarageFinish'] == 'Fin','GarageFinish'] = 3
    ord_data.loc[ord_data['GarageFinish'] == 'RFn','GarageFinish'] = 2
    ord_data.loc[ord_data['GarageFinish'] == 'Unf','GarageFinish'] = 1
    ord_data.loc[ord_data['GarageFinish'] == 'NotPresent','GarageFinish'] = 0

    ord_data.loc[ord_data['Fence'] == 'GdPrv','Fence'] = 4
    ord_data.loc[ord_data['Fence'] == 'MnPrv','Fence'] = 3
    ord_data.loc[ord_data['Fence'] == 'GdWo','Fence'] = 2
    ord_data.loc[ord_data['Fence'] == 'MnWw','Fence'] = 1
    ord_data.loc[ord_data['Fence'] == 'NotPresent','Fence'] = 0
    return ord_data
#Upon inspecting the correlation of the ordinal features to the SalePrice, we choose to keep only those with at least weak correlation (>0.3 correlation), all others will be dummy coded along with other categorical variables
ord_data = ordinal_coding(data)
test_ord_data = ordinal_coding(test)

ord_data['SalePrice'] = data['SalePrice'].copy()
ordinal_corr_dict = {}
for col in ordinal:
    ordinal_corr_dict[col] = ord_data[col].corr(ord_data['SalePrice'])

"""
ordinal_corr_dict_sorted = sorted(ordinal_corr_dict.items(), key=lambda kv: kv[1], reverse=True)
for corr in ordinal_corr_dict_sorted:
    print(corr)

f, ax = plt.subplots(4, 4, figsize=(100,50))
for row in range(4):
    for col in range(4):
        idx = col+(4*row)
        if idx < 17:
            x=ord_data.iloc[:, idx]
            y=ord_data['SalePrice']
            ax[row, col].scatter(x, y, s=5)
            ax[row, col].set_title(ord_data.columns[idx])
            ax[row, col].set_yticklabels([])

#f.delaxes(ax[6, 3])
#f.delaxes(ax[6, 4])
#f.delaxes(ax[6, 5])
plt.subplots_adjust(hspace=0.8)
plt.show()
"""

ordinal_keep =  [k for k,v in ordinal_corr_dict.items() if abs(v) >= 0.3]

#Implement numeric coding of ordinal features
data[ordinal_keep] = ord_data[ordinal_keep]
test[ordinal_keep] = test_ord_data[ordinal_keep]

#Add interaction terms
numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.drop(columns=['SalePrice'])
numeric_data = numeric_data.drop(columns=quadratic_columns)

interactions_lasso = ['OverallQual:OverallCond', 'GrLivArea:TotalSF', 'LotArea:GrLivArea', 'OverallCond:ExterQual', 'ExterQual:Fireplaces', 'OverallCond:TotalSF', 'KitchenQual:TotalBaths', 'OverallQual:TotalBaths', '1stFlrSF:GrLivArea', 'KitchenQual:GarageCond', 'OverallQual:GarageQual', 'KitchenAbvGr:AgeSold', 'HeatingQC:TotalBaths', 'BsmtQual:TotalBaths', 'BsmtExposure:FireplaceQu']
interactions_xgboost = ['OverallCond:TotalSF', 'OverallCond:AgeSold', 'ExterQual:AgeSold', 'LotArea:OverallCond', 'LotArea:GrLivArea', '1stFlrSF:AgeSold', 'OverallCond:TotalSF', 'LotFrontage:ScreenPorch', 'BsmtQual:OpenPorchSF', '1stFlrSF:GrLivArea', 'TotalBaths:AgeSold', 'OverallCond:GrLivArea', 'BsmtFinType1:BsmtFinSF1', 'HeatingQC:TotalSF', 'LotArea:TotalSF']
interactions = list(set(interactions_lasso) | set(interactions_xgboost))

for i in range(len(numeric_data.columns)):
    i_col = numeric_data.columns[i]
    for j in range(i + 1, len(numeric_data.columns)):
        j_col = numeric_data.columns[j]
        col_name = str(numeric_data.columns[i]) + ':' + numeric_data.columns[j]
        data[col_name] = data[i_col] * data[j_col]
        test[col_name] = test[i_col] * test[j_col]
        if col_name not in interactions:
            data = data.drop(columns=col_name)
            test = test.drop(columns=col_name)

print(data.shape)

#There are 666 interaction terms, including this many terms may lead to overfitting, need to select only the most important factors
# BL: -0.012992945370987613, -0.009495158470982073
# ridge: all interactions: -0.025953560965334136, -0.02166142757630759
# ridge: lasso interactions -0.012613704715810713, -0.009285922796767725
# ridge: lasso and xgboost interactions -0.012480040798350288, -0.009133599989098148
# EN: {'alpha': 0.001, 'l1_ratio': 0.5} -0.011918440615919257, {'alpha': 0.0012, 'l1_ratio': 0.23333} -0.008811714804093606
# EN: {'alpha': 0.001, 'l1_ratio': 0.6333333333333333} -0.011864634627334846, {'alpha': 0.0008, 'l1_ratio': 0.36} -0.008670316153710377
# {'alpha': 0.0006, 'l1_ratio': 0.47555555555555556}
"""
interaction = PolynomialFeatures(interaction_only=True,include_bias = False)
interaction_df = interaction.fit_transform(numeric_data)


numeric_data_interactions = pd.DataFrame(interaction_df)
print(numeric_data.head())
print(numeric_data_interactions)
"""

#Upon inspection of the data, the remaining skew is coming from the large number of zero values in many of the numeric features
#By creating dummy variables we can indicate the presence or non-presence of the variable


zero_dummy_cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'GarageArea**2', 'MasVnrArea**2', 'BsmtFinSF1**2','OpenPorchSF**2','TotalBsmtSF**2','WoodDeckSF**2']
for col in zero_dummy_cols:
    data['NonZero' + col] = (data[col] > 0).astype(int)
    test['NonZero' + col] = (test[col] > 0).astype(int)
    print(data['NonZero' + col].value_counts())

#Remove variables with high number of zeros['PoolArea', '3SsnPorch', 'LowQualFinSF']


#Dummy code categorical columns
#print(data.shape)
cat_cols = [col for col in cat_data.columns if col not in ordinal_keep]
cat_cols.remove('SalePrice')



dummy_df = pd.get_dummies(data[cat_cols])
test_dummy_df = pd.get_dummies(test[cat_cols])


data = pd.concat([data, dummy_df], axis=1)
test = pd.concat([test, test_dummy_df], axis=1)

data = data.drop(columns=cat_cols, axis=1)
test = test.drop(columns=cat_cols, axis=1)

missing_cols = set(data.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0

test = test[data.columns]
test = test.drop(columns=['SalePrice'])



#print(data.shape)

#Feature selection
#Currently we have 256 features after dummy coding. Too many features will lead to over-fitting
#We can assess feature importance using various methods such as RFE, Gradient Boosting, Chi-Squared, Regularization
#We need a multivariate feature selection method so it can take into account of unknown relationships between features

"""
fs_y = data['SalePrice'].copy()
fs_x = data.copy().drop(columns='SalePrice')

#print(fs_y.shape)
#print(fs_x.shape)

model = XGBRegressor()
model.fit(fs_x, fs_y)
print(model.feature_importances_)



plot_importance(model, max_num_features=63)
plt.show()
"""

#Obtain best threshold value for minimizing RMSE
"""
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(fs_x, fs_y , test_size=0.33, random_state=7)
# fit model on all training data
model = XGBRegressor()
model.fit(X_train, y_train)
# make predictions for test data and evaluate
y_pred = model.predict(X_test)
RMSE = (np.sum((y_pred - y_test)**2))**0.5
print("RMSE: %.2f" % (RMSE))
# Fit model using each importance as a threshold
thresholds = np.unique(np.sort(model.feature_importances_))
print(thresholds)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    selection_model = XGBRegressor()
    selection_model.fit(select_X_train, y_train)
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    RMSE = (np.sum((np.expm1(y_test) - np.expm1(y_pred)) ** 2)/len(X_test)) ** 0.5
    print("Thresh=%.3f, n=%d, RMSE: %.5f" % (thresh, select_X_train.shape[1], RMSE))

"""
#The lowest RMSE is obtained when threshold is = 0.007 the most important features are selected which are:
"""
from sklearn.feature_selection import SelectFromModel
selection = SelectFromModel(model, threshold=0.004, prefit=True)
feature_idx = selection.get_support()
feature_x = fs_x.columns[feature_idx]
print(feature_x)


from sklearn.model_selection import GridSearchCV, train_test_split


X = data[feature_x]

print(X.shape)
Y = data['SalePrice']


# A parameter grid for XGBoost
#params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
#'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}


params = {
 'max_depth':[4],
 'min_child_weight':[2],
 'gamma': [0],
 'subsample': [0.8],
 'colsample_bytree': [0.8],
 #'reg_alpha':[0.01],
 'scale_pos_weight': [1],
 'random_state': [0],
 'learning_rate': [0.01],
 'n_estimators':[1000]
}

# Initialize XGB and GridSearch
# thres 0.004 max depth 4, min child weight 2 -0.013009625664150901, -0.0006374438533263484
# thres 0.0035 max depth 4, min child weight 2 -0.013582197980543808, -0.0006164275755484212

xgb = XGBRegressor()

grid = GridSearchCV(xgb, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X, Y)

y_pred_en = grid.best_estimator_.predict(X)
y_test = Y
best_params = grid.best_params_
best_score = grid.best_score_
best_knn = grid.best_estimator_
print(grid.cv_results_)
print(best_params)
print(best_score)
print(best_knn)

X_test = test[feature_x]
test_ensemble_xgb = grid.best_estimator_.predict(X_test)


X_test = test[feature_x]
xgboost_pred = np.expm1(grid.best_estimator_.predict(X_test))
prediction = pd.DataFrame(test_id, columns=['Id'])
prediction['SalePrice'] = xgboost_pred
prediction.to_csv('submission.csv', index=False)
"""
"""
Y = data['SalePrice']
X = data.drop(columns=['SalePrice'])

KR = KernelRidge()

params = {'alpha': np.linspace(100, 300, 10),
          'kernel': ['polynomial'],
          'coef0': np.linspace(10, 300, 5),
          'degree': [2]}

grid = GridSearchCV(KR, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X, Y)

y_pred_en = grid.best_estimator_.predict(X)
y_test = Y
best_params = grid.best_params_
best_score = grid.best_score_
best_knn = grid.best_estimator_
print(grid.cv_results_)
print(best_params)
print(best_score)
print(best_knn)

X_test = test
test_ensemble_kr = grid.best_estimator_.predict(X_test)
"""
"""
kr_pred = np.expm1(grid.best_estimator_.predict(X_test))
prediction = pd.DataFrame(test_id, columns=['Id'])
prediction['SalePrice'] = kr_pred
prediction.to_csv('submission.csv', index=False)
"""

#RIDGE
"""
Y = data['SalePrice']
X = data.drop(columns=['SalePrice'])

#min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(X)


rr_model = Ridge(normalize=False)
params = {'alpha': np.linspace(7, 20, 30)}

grid = GridSearchCV(rr_model, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X, Y)
best_params = grid.best_params_
best_score = grid.best_score_
best_knn = grid.best_estimator_
print(grid.cv_results_)
print(best_params)
print(best_score)
#print(list(zip(grid.best_estimator_.coef_, X.columns)))
#en_coefs = pd.Series(grid.best_estimator_.coef_,index=X.columns)
#print(en_coefs.sort_values(ascending=False))

y_pred_en = grid.best_estimator_.predict(X)

X_test = test
#X_test = min_max_scaler.transform(X_test)

rr_pred = np.expm1(grid.best_estimator_.predict(X_test))
prediction = pd.DataFrame(test_id, columns=['Id'])
prediction['SalePrice'] = rr_pred
prediction.to_csv('submission.csv', index=False)
"""

#LASSO/ELASTIC NET


Y = data['SalePrice']
X = data.drop(columns=['SalePrice'])


EN_model = ElasticNet(normalize=False)
params = {'alpha': np.linspace(0.0008, 0.0012, 3),
          'l1_ratio': np.linspace(0.58, 0.66, 4)}

grid = GridSearchCV(EN_model, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X, Y)
best_params = grid.best_params_
best_score = grid.best_score_
best_knn = grid.best_estimator_
print(grid.cv_results_)
print(best_params)
print(best_score)
print(list(zip(grid.best_estimator_.coef_, X.columns)))
en_coefs = pd.Series(grid.best_estimator_.coef_,index=X.columns)
print(en_coefs.sort_values(ascending=False))

y_pred_en = grid.best_estimator_.predict(X)


##LOOK AT RESSIDUAL PLOT
"""
y_test = list(Y)

mean_X = sum(y_pred_en) / len(y_pred_en)
mean_Y = sum(y_test) / len(y_test)


beta1 = sum([(y_pred_en[i] - mean_X)*(y_test[i] - mean_Y) for i in range(len(y_pred_en))]) / sum([(y_pred_en[i] - mean_X)**2 for i in range(len(y_pred_en))])
beta0 = mean_Y - beta1 * mean_X

y_hat = [beta0 + beta1*y_pred_en[i] for i in range(len(y_pred_en))]

residuals = [y_test[i] - y_hat[i] for i in range(len(y_test))]


Var_e = sum([(y_test[i] - y_hat[i])**2 for i in range(len(y_test)) ]) / (len(y_test) -2) # change 2 to number of parameters
SE_regression = Var_e**0.5
studentized_residuals = [residuals[i]/SE_regression for i in range(len(residuals))]
"""


residuals_train = y_pred_en - y_test
print(np.mean(residuals_train))

data['residual'] = residuals_train

data = data[abs(data['residual'])< 0.35]

plt.scatter(y_pred_en, residuals_train)
plt.show()

Y = data['SalePrice']
X = data.drop(columns=['SalePrice', 'residual'])


#EN

EN_model = ElasticNet(normalize=False)
params = {'alpha': np.linspace(0.0002, 0.0020, 10),
          'l1_ratio': np.linspace(0.1, 0.5, 10)}

grid = GridSearchCV(EN_model, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X, Y)
best_params = grid.best_params_
best_score = grid.best_score_
best_knn = grid.best_estimator_
print(grid.cv_results_)
print(best_params)
print(best_score)
print(list(zip(grid.best_estimator_.coef_, X.columns)))
en_coefs = pd.Series(grid.best_estimator_.coef_,index=X.columns)
print(en_coefs.sort_values(ascending=False))

y_pred_en = grid.best_estimator_.predict(X)

X_test = test
#X_test = min_max_scaler.transform(X_test)

rr_pred = np.expm1(grid.best_estimator_.predict(X_test))
prediction = pd.DataFrame(test_id, columns=['Id'])
prediction['SalePrice'] = rr_pred
prediction.to_csv('submission.csv', index=False)

y_test = Y
residuals_train = y_pred_en - y_test
print(np.mean(residuals_train))

data['residual'] = residuals_train

plt.scatter(y_pred_en, residuals_train)
plt.show()
#End EN

"""
#KRR
KR = KernelRidge()

params = {'alpha': np.linspace(100, 300, 10),
          'kernel': ['polynomial'],
          'coef0': np.linspace(100, 500, 5),
          'degree': [2]}

grid = GridSearchCV(KR, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X, Y)

y_pred_en = grid.best_estimator_.predict(X)
y_test = Y
best_params = grid.best_params_
best_score = grid.best_score_
best_knn = grid.best_estimator_
print(grid.cv_results_)
print(best_params)
print(best_score)
print(best_knn)

X_test = test
test_ensemble_kr = grid.best_estimator_.predict(X_test)

rr_pred = np.expm1(grid.best_estimator_.predict(X_test))
prediction = pd.DataFrame(test_id, columns=['Id'])
prediction['SalePrice'] = rr_pred
prediction.to_csv('submission.csv', index=False)

y_test = Y
residuals_train = y_pred_en - y_test
print(np.mean(residuals_train))

data['residual'] = residuals_train

plt.scatter(y_pred_en, residuals_train)
plt.show()
"""

"""
#XGB
X = X[feature_x]
xgb = XGBRegressor()

grid = GridSearchCV(xgb, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X, Y)

y_pred_en = grid.best_estimator_.predict(X)
y_test = Y
best_params = grid.best_params_
best_score = grid.best_score_
best_knn = grid.best_estimator_
print(grid.cv_results_)
print(best_params)
print(best_score)
print(best_knn)

X_test = test[feature_x]
test_ensemble_xgb = grid.best_estimator_.predict(X_test)


X_test = test[feature_x]
xgboost_pred = np.expm1(grid.best_estimator_.predict(X_test))
prediction = pd.DataFrame(test_id, columns=['Id'])
prediction['SalePrice'] = xgboost_pred
prediction.to_csv('submission.csv', index=False)

y_test = Y
residuals_train = y_pred_en - y_test
print(np.mean(residuals_train))

data['residual'] = residuals_train

plt.scatter(y_pred_en, residuals_train)
plt.show()
#end XGB

#min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(X)
"""
"""
rr_model = Ridge(normalize=False)
params = {'alpha': np.linspace(7, 20, 30)}

grid = GridSearchCV(rr_model, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X, Y)
best_params = grid.best_params_
best_score = grid.best_score_
best_knn = grid.best_estimator_
print(grid.cv_results_)
print(best_params)
print(best_score)
#print(list(zip(grid.best_estimator_.coef_, X.columns)))
#en_coefs = pd.Series(grid.best_estimator_.coef_,index=X.columns)
#print(en_coefs.sort_values(ascending=False))

y_pred_en = grid.best_estimator_.predict(X)

X_test = test
#X_test = min_max_scaler.transform(X_test)

rr_pred = np.expm1(grid.best_estimator_.predict(X_test))
prediction = pd.DataFrame(test_id, columns=['Id'])
prediction['SalePrice'] = rr_pred
prediction.to_csv('submission.csv', index=False)

y_test = Y
residuals_train = y_pred_en - y_test
print(np.mean(residuals_train))

data['residual'] = residuals_train

plt.scatter(y_pred_en, residuals_train)
plt.show()
"""
#EN outliers: [30, 88, 410, 462, 495, 588, 632, 968, 970, 1324, 1432, 1453]
#RR outliers: [30, 410, 462, 495, 588, 632, 968, 970, 1324, 1432, 1453]

#test_ensemble_en = grid.best_estimator_.predict(X_test)


#Create an ensemble model from all XGBoost, Kernel Ridge and Elastic net models
"""
ensemble = pd.DataFrame(data['SalePrice'], columns=['SalePrice'])


ensemble['SalePrice'] = data['SalePrice']
ensemble['XGB'] = y_pred_xgb
ensemble['kr'] = y_pred_kr
ensemble['en'] = y_pred_en

ensemble_Y = ensemble['SalePrice']
ensemble_X = ensemble[['kr', 'en']]

ensemble_EN = ElasticNet(normalize=True)
params = {'alpha': [1e-7],
          'l1_ratio': [0.25, 0.3, 0.35]}

grid = GridSearchCV(ensemble_EN, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(ensemble_X, ensemble_Y)
best_params = grid.best_params_
best_score = grid.best_score_
best_knn = grid.best_estimator_
print(grid.cv_results_)
print(best_params)
print(best_score)
print(list(zip(grid.best_estimator_.coef_, ['kr', 'en'])))

test_ensemble = pd.DataFrame(test_ensemble_xgb, columns=['XGB'])
test_ensemble['XGB'] = test_ensemble_xgb
test_ensemble['kr'] = test_ensemble_kr
test_ensemble['en'] = test_ensemble_en

test_ensemble_X = test_ensemble[['kr', 'en']]

print(test_ensemble.head())

ensemble_pred = np.expm1(grid.best_estimator_.predict(test_ensemble_X))
prediction = pd.DataFrame(test_id, columns=['Id'])
prediction['SalePrice'] = ensemble_pred
prediction.to_csv('submission.csv', index=False)
"""
"""
EN_pred = np.expm1(grid.best_estimator_.predict(X_test))
prediction = pd.DataFrame(test_id, columns=['Id'])
prediction['SalePrice'] = EN_pred
prediction.to_csv('submission.csv', index=False)


"""

"""
#SVR
Y = data['SalePrice']
X = data.drop(columns=['SalePrice'])

svr_model = SVR()

params = {'C':[1],
              'kernel':['poly','rbf','sigmoid']}
grid = GridSearchCV(svr_model, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X, Y)
best_params = grid.best_params_
best_score = grid.best_score_
best_knn = grid.best_estimator_
print(grid.cv_results_)
print(best_params)
print(best_score)

"""



"""
#-0.013878054892817572 thresh 0.005 XGB gamma 0, max_depth 4, min_child_weight 2, colsample_bytree 0.8, subsample:0.7
#-0.012595452819610175 EN {'alpha': 0.0001, 'l1_ratio': 0.99}

#with quadratic columns
#-0.0122375 EN {'alpha': 0.0001, 'l1_ratio': 0.99} with quadratic numerical columns threshold 0.3 (no skew corr)
#-0.0125825 EN {'alpha': 0.0001, 'l1_ratio': 0.95 with quadratic numerical columns threshold 0.1

#with qc + new variables:
#-0.013137182554602458 XGBthresh 0.001 XGB gamma 0, max_depth 4, min_child_weight 2, colsample_bytree 0.8, subsample:0.8
#-0.012888155481127268 XGBthresh 0.001 KernelRidge(alpha=0.03, coef0=50, degree=1, gamma=None, kernel='polynomial',
      kernel_params=None)
#-0.012257447435523235 EN {'alpha': 0.0001, 'l1_ratio': 0.99}
#-0.01309022 rr alpha: {'alpha': 10}
"""