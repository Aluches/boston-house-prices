# Importing necessary libraries
import numpy as np              # For numerical operations
import pandas as pd             # For data manipulation and analysis
import seaborn as sns           # For statistical data visualization
import matplotlib.pyplot as plt # For plotting graphs
import statsmodels.api as sm
from scipy import stats

# Importing sklearn libraries
from sklearn.model_selection import train_test_split # To split data into training and testing sets
from sklearn.linear_model import LinearRegression    # For performing linear regression
from sklearn.metrics import mean_squared_error       # To evaluate the model's performance
from sklearn.preprocessing import StandardScaler     # For feature scaling


import warnings
warnings.filterwarnings("ignore")

#Loading Datasets
file_path = "Boston.csv"
boston_df = pd.read_csv(file_path)

#OBSERVATIONS
# Load the dataset
data = pd.read_csv("Boston.csv")  # Loading the dataset from a CSV file

# Display the first few rows of the dataset to understand its structure
print(data.head())  # Displaying the first five rows of the dataset

# Display the summary statistics of the dataset
print("\nDataset Info:")
print(data.describe())  # Generating descriptive statistics of the dataset

# Check for missing values in the dataset
print("\nMissing Values Per Column:")
print(data.isnull().sum())  # Summarizing the count of missing values for each column

#SANITY CHECK FOR THE DATA
# Check the data types of each column to ensure they are appropriate
data.dtypes  # Displaying the data types of each column

# Check for duplicate rows in the dataset
data.duplicated().sum()  # Counting the number of duplicate rows

# Display the correlation matrix to understand relationships between variables
plt.figure(figsize=(10, 8))  # Setting the figure size for better readability
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')  # Plotting the correlation matrix with a heatmap
plt.title('Correlation Matrix')  # Adding a title to the plot
plt.show()  # Displaying the plot


#Exploratory Data Analysis (EDA)
# Distribution of 'MEDV'
plt.figure(figsize=(8, 6))  # Setting the figure size
sns.histplot(data['MEDV'], kde=True, bins=30)  # Plotting the histogram of MEDV with a kernel density estimate
plt.title('Distribution of MEDV')  # Adding a title to the plot
plt.xlabel('MEDV (Median value of owner-occupied homes in $1000)')  # Labeling the x-axis
plt.ylabel('Frequency')  # Labeling the y-axis
plt.show()  # Displaying the plot

# Univariate analysis for different variables
univariate_plots = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']  # List of variables for univariate analysis
for variable in univariate_plots:
    plt.figure(figsize=(8, 6))  # Setting the figure size for each plot
    sns.histplot(data[variable], kde=True, bins=30)  # Plotting the histogram of each variable with a kernel density estimate
    plt.title(f'Distribution of {variable}')  # Adding a title to each plot
    plt.xlabel(variable)  # Labeling the x-axis for each plot
    plt.ylabel('Frequency')  # Labeling the y-axis for each plot
    plt.show()  # Displaying each plot

# Bivariate analysis for additional pairs of features with significant correlations
# Define the pairs of variables for bivariate analysis
bivariate_pairs = [
        ('RM', 'MEDV'),  # Pair 1: RM and MEDV
        ('LSTAT', 'MEDV'),  # Pair 2: LSTAT and MEDV
        ('TAX', 'RAD'),  # Pair 3: TAX and RAD
        ('INDUS', 'NOX'),  # Pair 4: INDUS and NOX
        ('INDUS', 'DIS'),  # Pair 5: INDUS and DIS
        ('INDUS', 'TAX'),  # Pair 6: INDUS and TAX
        ('NOX', 'AGE'),  # Pair 7: NOX and AGE
        ('AGE', 'DIS'),  # Pair 8: AGE and DIS
        ('DIS', 'NOX')  # Pair 9: DIS and NOX
    ]

# Plotting each pair of variables to visualize their relationships
for x_var, y_var in bivariate_pairs:
        plt.figure(figsize=(8, 6))  # Setting the figure size for each plot
        sns.scatterplot(x=data[x_var], y=data[y_var])  # Creating a scatter plot for each pair of variables
        plt.title(f'Relationship between {x_var} and {y_var}')  # Adding a title to each scatter plot
        plt.xlabel(x_var)  # Labeling the x-axis for each scatter plot
        plt.ylabel(y_var)  # Labeling the y-axis for each scatter plot
        plt.show()  # Displaying each scatter plot

#Let's check the outliers

#RAD vs TAX
# Import from spicy.stats
from scipy.stats import pearsonr

# Remove the outliers
boston_df1 = boston_df[boston_df['TAX'] < 600]

print('The correlation between TAX and RAD is', pearsonr(boston_df1['TAX'], boston_df1['RAD'])[0])


#INDUS vs TAX
# Import from spicy.stats
from scipy.stats import pearsonr

# Remove the outliers
boston_df1 = boston_df[boston_df['TAX'] < 500]
boston_df1 = boston_df[boston_df['INDUS'] < 25]

print('The correlation between TAX and INDUS is', pearsonr(boston_df1['TAX'], boston_df1['INDUS'])[0])


#Data Preprocessing
#Missing Value Treatment Identify and handle any missing values in the dataset.
#Missing value treatment
# Check for missing values in the dataset
missing_values = boston_df.isnull().sum()  # Summarize the count of missing values for each column
print("Missing values in each column:\n", missing_values)

#duplicated entries
num_duplicates = boston_df.duplicated().sum()  # Counting the number of duplicate rows
print(f'Number of duplicate entries: {num_duplicates}')  # Printing the number of duplicate entries

#Log Transformation of Dependent Variable if Skewed
#Transforming the Skewed 'MEDV' with log transformation
boston_df['MEDV_log'] = np.log(boston_df['MEDV'])  # Applying log transformation
boston_df1['MEDV_log'] = np.log(boston_df['MEDV'])  # Applying log transformation

# Check the distribution of the transformed 'MEDV' column
plt.figure(figsize=(8, 6))  # Setting the figure size
sns.histplot(boston_df['MEDV_log'], kde=True, bins=30)  # Plotting the histogram of log-transformed MEDV with a kernel density estimate
plt.title('Distribution of Log-Transformed MEDV')  # Adding a title to the plot
plt.xlabel('Log(MEDV) (Log of median value of owner-occupied homes)')  # Labeling the x-axis
plt.ylabel('Frequency')  # Labeling the y-axis
plt.show()  # Displaying the plot

#Feature Engineering Let's create a new feature that might be useful for our model.
# Creating a new feature: Ratio of rooms to age, this can make a relation of room acrros the time.
boston_df['RM_AGE_ratio'] = boston_df['RM'] / boston_df['AGE']  # Creating a new feature as the ratio of rooms to age

#Outlier Detection and Treatment (if needed) We will detect outliers using the IQR method and handle them.
# Outlier detection using IQR for the 'CRIM' column as an example
Q1 = boston_df1['CRIM'].quantile(0.25)  # First quartile
Q3 = boston_df1['CRIM'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile range

# Define the lower and upper bounds for detecting outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out the outliers
boston_df = boston_df1[(boston_df1['CRIM'] >= lower_bound) & (boston_df['CRIM'] <= upper_bound)]

#Preparing Data for Modeling
# Define the features (X) and the target (y)
X = boston_df1.drop(['MEDV', 'MEDV_log'], axis=1)  # Dropping the original and log-transformed target variables
y = boston_df1['MEDV_log']  # Using the log-transformed target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  # 80-20 split

#Standardizing the features if they vary greatly in scale.
#tester method
from sklearn.preprocessing import StandardScaler

# Standardizing the features
scaler = StandardScaler()  # Initializing the scaler
X_train_scaled = scaler.fit_transform(X_train)  # Fitting and transforming the training data
X_test_scaled = scaler.transform(X_test)  # Transforming the test data


#Model Building - Linear Regression
#Checking for Multicollinearity (Remove the highest VIF values and check again)
#We will use the Variance Inflation Factor (VIF) to detect multicollinearity.
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to calculate VIF
def calculate_vif(df):
    vif = pd.DataFrame()
    vif["Feature"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif

# Calculate VIF for the initial set of features
vif_df = calculate_vif(X_train)
print("Initial VIF values:\n", vif_df)

# Iteratively remove features with the highest VIF values until all VIFs are below 5
while vif_df['VIF'].max() > 5:
    feature_to_remove = vif_df.loc[vif_df['VIF'].idxmax(), 'Feature']
    X_train = X_train.drop(columns=[feature_to_remove])
    X_test = X_test.drop(columns=[feature_to_remove])
    vif_df = calculate_vif(X_train)
    print(f"Removed {feature_to_remove} with VIF {vif_df['VIF'].max()}\n")
    print("Updated VIF values:\n", vif_df)

# Create the Linear Regression Model
from sklearn.linear_model import LinearRegression

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Check VIF again to ensure multicollinearity is addressed
vif_df = calculate_vif(pd.DataFrame(X_train, columns=X_train.columns))
print("Final VIF values:\n", vif_df)

#Examining the Significance of the Model
#We will use statsmodels to check the significance of regression coefficients.
import statsmodels.api as sm

# Adding a constant for the intercept
X_train_const = sm.add_constant(X_train)

# Fit the model using statsmodels
model_sm = sm.OLS(y_train, X_train_const).fit()

# Print the summary to check the significance of coefficients
print(model_sm.summary())

#Model Performance Check
#We will use R-squared, RMSE, MAE, and MAPE to evaluate the model performance.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Performance on train data
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

# Performance on test data
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

print(f'Train RMSE: {train_rmse}, Train MAE: {train_mae}, Train R^2: {train_r2}, Train MAPE: {train_mape}')
print(f'Test RMSE: {test_rmse}, Test MAE: {test_mae}, Test R^2: {test_r2}, Test MAPE: {test_mape}')

#We will recalculate VIF values to ensure multicollinearity is handled.
# Check VIF to ensure multicollinearity is addressed
vif_df = calculate_vif(pd.DataFrame(X_train, columns=X_train.columns))
print("Final VIF values:\n", vif_df)

#We will apply cross-validation and evaluate the performance
from sklearn.model_selection import cross_val_score

# Cross-validation
cv_rmse = np.sqrt(-cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error'))
cv_mae = -cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_r2 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

print(f'Cross-Validation RMSE: {cv_rmse.mean()}')
print(f'Cross-Validation MAE: {cv_mae.mean()}')
print(f'Cross-Validation R^2: {cv_r2.mean()}')

#Checking Linear Regression Assumptions
# Mean of residuals
residuals = y_train - model.predict(X_train)
mean_residuals = np.mean(residuals)
print(f"Mean of residuals: {mean_residuals}")

# No Heteroscedasticity
plt.figure(figsize=(8, 6))
plt.scatter(model.predict(X_train), residuals)
plt.title('Heteroscedasticity Check')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.show()

# Linearity of variables
plt.figure(figsize=(8, 6))
sns.regplot(x=model.predict(X_train), y=y_train, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.title('Linearity Check')
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.show()

# Normality of error terms
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title('Normality of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

#Q-Q plot of residuals
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.show()

stat, pvalue = stats.shapiro(residuals)
print(f'Shapiro-Wilk test statistic\nw:{stat:2.4f}\npvalue: {pvalue:0.4f}')

#Get Model Coefficients
# Get model coefficients
coefs = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefs': model.coef_
})

print(coefs)

#FINAL MODEL
print(model_sm.summary())
