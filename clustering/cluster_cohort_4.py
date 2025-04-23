import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv('../cohorts/cohort4.csv')
true_labels = df['final_dropout']
X = df.drop(columns=['final_dropout'])

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
plt.title('KMeans Clustering - Cohort 4')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

# Actual
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='coolwarm', edgecolor='k')
plt.title('Actual Dropout Labels - Cohort 4')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

plt.tight_layout()
plt.savefig(f"{output_dir}/cohort4.png")
plt.close()
