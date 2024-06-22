# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
titanic_data = pd.read_csv('/home/develijah/Data Analysis Project/Titanic Data Analysis Project/archive/Titanic-Dataset.csv')  

# Display the entire dataset (be cautious with large datasets)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print("Entire dataset:")
print(titanic_data)

# Reset display options to default
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(titanic_data.head())

# Display the last few rows of the dataset
print("Last few rows of the dataset:")
print(titanic_data.tail())

# Display summary statistics
print("\nSummary statistics of the dataset:")
print(titanic_data.describe())

# Check for missing values
print("\nMissing values in each column:")
print(titanic_data.isnull().sum())

# Check data types
print("\nData types of each column:")
print(titanic_data.dtypes)

# Handle missing values (example: fill with median or drop rows/columns)
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data.drop(columns=['Cabin'], inplace=True)  # Dropping column with too many missing values
titanic_data.dropna(subset=['Embarked'], inplace=True)  # Drop rows with missing 'Embarked'

# Verify cleaning
print("\nMissing values after cleaning:")
print(titanic_data.isnull().sum())

# Descriptive statistics
print("\nUpdated summary statistics of the dataset:")
print(titanic_data.describe())

# Visualize survival rate by gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', hue='Sex', data=titanic_data)
plt.title('Survival by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Visualize survival rate by class
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', hue='Pclass', data=titanic_data)
plt.title('Survival by Class')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Histogram of ages
plt.figure(figsize=(10, 6))
plt.hist(titanic_data['Age'], bins=30, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = titanic_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Survival rate by age
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic_data, x='Age', hue='Survived', multiple='stack')
plt.title('Survival Rate by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Additional analysis: Survival rate by fare
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic_data, x='Fare', hue='Survived', multiple='stack', bins=30)
plt.title('Survival Rate by Fare')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# Additional analysis: Survival rate by embarked location
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', hue='Embarked', data=titanic_data)
plt.title('Survival by Embarked Location')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Additional analysis: Boxplot of age by survival status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Age', data=titanic_data)
plt.title('Age Distribution by Survival Status')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.show()
