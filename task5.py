import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('heart.csv')

# Initial data exploration
print(data.head())
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Feature engineering
# Example: Creating a new feature for BMI (Body Mass Index) if weight and height are available
# data['BMI'] = data['weight'] / (data['height']/100)**2

# Transform existing features if necessary
# Example: Normalize or scale features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop('target', axis=1))

# Split the data into training and testing sets
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PCA for feature selection
pca = PCA(n_components=5)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a RandomForest model to get feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")

# Plot the feature importances
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Train a model with selected features
selected_features = X.columns[indices[:5]]  # Select top 5 features based on importance
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

# Predict and evaluate the model
y_pred = rf_selected.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy with selected features: {accuracy}")

# Save the preprocessed data and model if needed
X_train_selected.to_csv('X_train_selected.csv', index=False)
X_test_selected.to_csv('X_test_selected.csv', index=False)

#output
# mayanksingh@MAYANKs-MacBook-Air task2 % python3 -u "/Users/mayanksingh/Desktop/Main/task2/task5.py"
#    age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal  target
# 0   52    1   0       125   212    0        1      168      0      1.0      2   2     3       0
# 1   53    1   0       140   203    1        0      155      1      3.1      0   0     3       0
# 2   70    1   0       145   174    0        1      125      1      2.6      0   0     3       0
# 3   61    1   0       148   203    0        1      161      0      0.0      2   1     3       0
# 4   62    0   0       138   294    1        1      106      0      1.9      1   3     2       0
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1025 entries, 0 to 1024
# Data columns (total 14 columns):
#  #   Column    Non-Null Count  Dtype  
# ---  ------    --------------  -----  
#  0   age       1025 non-null   int64  
#  1   sex       1025 non-null   int64  
#  2   cp        1025 non-null   int64  
#  3   trestbps  1025 non-null   int64  
#  4   chol      1025 non-null   int64  
#  5   fbs       1025 non-null   int64  
#  6   restecg   1025 non-null   int64  
#  7   thalach   1025 non-null   int64  
#  8   exang     1025 non-null   int64  
#  9   oldpeak   1025 non-null   float64
#  10  slope     1025 non-null   int64  
#  11  ca        1025 non-null   int64  
#  12  thal      1025 non-null   int64  
#  13  target    1025 non-null   int64  
# dtypes: float64(1), int64(13)
# memory usage: 112.2 KB
# None
#                age          sex           cp     trestbps        chol  ...      oldpeak        slope           ca         thal       target
# count  1025.000000  1025.000000  1025.000000  1025.000000  1025.00000  ...  1025.000000  1025.000000  1025.000000  1025.000000  1025.000000
# mean     54.434146     0.695610     0.942439   131.611707   246.00000  ...     1.071512     1.385366     0.754146     2.323902     0.513171
# std       9.072290     0.460373     1.029641    17.516718    51.59251  ...     1.175053     0.617755     1.030798     0.620660     0.500070
# min      29.000000     0.000000     0.000000    94.000000   126.00000  ...     0.000000     0.000000     0.000000     0.000000     0.000000
# 25%      48.000000     0.000000     0.000000   120.000000   211.00000  ...     0.000000     1.000000     0.000000     2.000000     0.000000
# 50%      56.000000     1.000000     1.000000   130.000000   240.00000  ...     0.800000     1.000000     0.000000     2.000000     1.000000
# 75%      61.000000     1.000000     2.000000   140.000000   275.00000  ...     1.800000     2.000000     1.000000     3.000000     1.000000
# max      77.000000     1.000000     3.000000   200.000000   564.00000  ...     6.200000     2.000000     4.000000     3.000000     1.000000

# [8 rows x 14 columns]
# age         0
# sex         0
# cp          0
# trestbps    0
# chol        0
# fbs         0
# restecg     0
# thalach     0
# exang       0
# oldpeak     0
# slope       0
# ca          0
# thal        0
# target      0
# dtype: int64
# Feature ranking:
# 1. feature 2 (0.13507197453533204)
# 2. feature 11 (0.1273270473259574)
# 3. feature 7 (0.12216864549468961)
# 4. feature 9 (0.121904718442662)
# 5. feature 12 (0.11051814687257032)
# 6. feature 0 (0.07790831761825727)
# 7. feature 4 (0.07482220268882704)
# 8. feature 3 (0.07117087529075947)
# 9. feature 8 (0.05759449379975219)
# 10. feature 10 (0.04578200610925984)
# 11. feature 1 (0.028731063614836953)
# 12. feature 6 (0.018556906142942732)
# 13. feature 5 (0.008443602064153139)