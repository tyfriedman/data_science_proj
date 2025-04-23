import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv('../cohorts/cohort2.csv')
true_labels = df['dropped_after_test_2']
X = df.drop(columns=['dropped_after_test_2'])

# normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 5))

# Predicted 
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolor='k')
plt.title('KMeans Clustering - Cohort 2')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

# Actual
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='coolwarm', edgecolor='k')
plt.title('Actual Dropout Labels - Cohort 2')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

plt.tight_layout()
plt.savefig(f"{output_dir}/cohort2.png")
plt.close()
