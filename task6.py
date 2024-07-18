import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud

# Load the CSV file
filename = 'disney_plus_titles.csv'
data = pd.read_csv(filename)

# Display the first few rows of the dataset
print(data.head())

# 1. Time Series Analysis
data['date_added'] = pd.to_datetime(data['date_added'])
data.set_index('date_added', inplace=True)

# Resample to monthly frequency
monthly_data = data['show_id'].resample('M').count()

# Plot the time series data
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_data)
plt.title('Monthly Additions of Movies and TV Shows')
plt.xlabel('Date')
plt.ylabel('Number of Additions')
plt.show()

# 2. Text Mining on Descriptions
# Generate word cloud for descriptions
text = ' '.join(data['description'].dropna().tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Descriptions')
plt.show()

# 3. Clustering
# Vectorize the descriptions
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['description'].dropna())

# Perform KMeans clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)
data['cluster'] = kmeans.labels_

# Display the first few rows with cluster labels
print(data[['title', 'cluster']].head())

# Visualize clustering results
plt.figure(figsize=(12, 6))
sns.countplot(x='cluster', data=data)
plt.title('Number of Movies and TV Shows per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()
