import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('01.Data Cleaning and Preprocessing.csv', skipinitialspace=True)

# Display basic information about the DataFrame
print("Original DataFrame Info:")
print(df.info())

# Display the first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:")
print(df.head())

# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')

# Display updated column names
print("\nUpdated column names:")
print(df.columns)

# Separate 'Observation' column
observation_column = df['Observation']
df = df.drop('Observation', axis=1)

# Convert remaining columns to appropriate data types
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Display updated DataFrame info
print("\nUpdated DataFrame Info:")
print(df.info())

# Handle missing values
df_cleaned = df.dropna()  # Remove rows with any missing values
df_filled = df.fillna(df.mean())  # Fill missing values with column mean

# Calculate summary statistics
summary_stats = df.describe()
print("\nSummary Statistics:")
print(summary_stats)

# Example calculations
print("\nExample calculations:")
for col in df.columns[:5]:  # Calculate mean for the first 5 columns as an example
    print(f"Mean of {col}: {df[col].mean()}")

# Filter data based on conditions (example)
filtered_df = df[df['Y_Kappa'] > 20]
print("\nNumber of rows where Y_Kappa > 20:", len(filtered_df))

# Calculate correlation matrix
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Analyze the 'Observation' column
print("\nUnique values in the Observation column:")
print(observation_column.unique())
print("\nNumber of unique values:", observation_column.nunique())