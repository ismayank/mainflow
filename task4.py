import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'USvideos.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Distribution of numerical variables
numerical_columns = ['views', 'likes', 'dislikes', 'comment_count']
data[numerical_columns].hist(bins=50, figsize=(20, 15))
plt.show()

# Boxplot to identify outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[numerical_columns])
plt.show()

# Correlation matrix
corr_matrix = data[numerical_columns].corr()
print(corr_matrix)

# Heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Distribution of categorical variables
categorical_columns = ['category_id', 'comments_disabled', 'ratings_disabled', 'video_error_or_removed']
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=column)
    plt.show()
