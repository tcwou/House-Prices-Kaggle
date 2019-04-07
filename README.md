# Housing Prices Prediction - Kaggle Competition

In this project I utilized Linear Regression models to predict housing prices as part of the 'House Prices: Advanced Regression Techniques' Kaggle competition.

The notebook will discuss the process and results for the initial data exploration, missing value imputation, feature engineering, feature selection, and regression modelling. The models used to fit the data include Ridge, Lasso, Elastic Net and Gradient Boosting (XGBoost).

By comparing standardized elastic net coefficients, the most important categorical features for high sale price were related to neighborhood, zoning type, and home functionality. The most important numerical features were related to property age and size.

The RMLSE achieved by the top performing Elastic Net model is 0.11501, placing it in the top 7% of scores as of January 2019.

## Notebook
* [Notebook](https://nbviewer.jupyter.org/github/tcwou/House-Prices-Kaggle/blob/master/Kaggle%20Housing%20Project%20Notebook.ipynb)

### Methods Used
* Machine Learning
* Data Visualization
* Feature Engineering
* Regularized Linear Regression
* Gradient Boosting

### Technologies
* Python
* Pandas
* Scikit-learn
