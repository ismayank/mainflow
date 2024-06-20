import pandas as pd
import matplotlib.pyplot as plt
import csv

# Read the CSV file
df = pd.read_csv('householdtask3.csv')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Bar Chart
bar_data = ['tot_hhs', 'own', 'own_wm', 'income', 'expenditure']
ax1.bar(bar_data, df[bar_data].iloc[0])
ax1.set_title('Household Statistics (2008)')
ax1.set_ylabel('Count/Amount')
ax1.set_xlabel('Categories')
ax1.tick_params(axis='x', rotation=45)

# Add value labels on top of each bar
for i, v in enumerate(df[bar_data].iloc[0]):
    ax1.text(i, v, f'{v:,.0f}', ha='center', va='bottom')

# Line Chart
line_data = ['own_prop', 'own_wm_prop', 'prop_hhs', 'age', 'size']
ax2.plot(line_data, df[line_data].iloc[0], marker='o')
ax2.set_title('Household Proportions and Averages (2008)')
ax2.set_ylabel('Value')
ax2.set_xlabel('Categories')
ax2.tick_params(axis='x', rotation=45)

# Add value labels for each point
for i, v in enumerate(df[line_data].iloc[0]):
    ax2.text(i, v, f'{v:.1f}', ha='center', va='bottom')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Print the first few rows of the DataFrame to verify the data
print(df.head())

# Print column names
print("\nColumn names:")
print(df.columns)