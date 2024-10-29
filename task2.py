import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample data: Customer ID, Total Spent, Purchase Frequency
data = {
    'Customer_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Total_Spent': [200, 150, 300, 250, 100, 400, 350, 450, 500, 50],
    'Purchase_Frequency': [5, 3, 6, 4, 2, 8, 7, 9, 10, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Select features for clustering
features = df[['Total_Spent', 'Purchase_Frequency']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plotting the elbow graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Fit K-means model with an optimal number of clusters (e.g., 3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(scaled_features)

# Assign cluster labels to the original DataFrame
df['Cluster'] = kmeans.labels_

# Display the DataFrame with cluster assignments
print(df)

# Optional: Visualize the clusters
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('Customer Clusters')
plt.xlabel('Standardized Total Spent')
plt.ylabel('Standardized Purchase Frequency')
plt.colorbar()
plt.show()
