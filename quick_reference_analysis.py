"""
Quick Reference: Using Preprocessed River Water Quality Data
"""

# ============================================================================
# LOADING PREPROCESSED DATA
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the scaled features (RECOMMENDED for most algorithms)
X_scaled = pd.read_csv('river_water_features_scaled.csv')

# Load unscaled features (for tree-based methods or interpretation)
X_unscaled = pd.read_csv('river_water_features.csv')

# Load full preprocessed dataset (includes all columns)
df_full = pd.read_csv('river_water_preprocessed.csv')

print(f"Scaled features shape: {X_scaled.shape}")
print(f"Features: {list(X_scaled.columns)}")

# ============================================================================
# EXAMPLE 1: K-MEANS CLUSTERING
# ============================================================================

# Find optimal number of clusters using Elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot Elbow curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs k')
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/clustering_optimization.png', dpi=300)
plt.show()

# Apply K-Means with optimal k (example: k=4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_full['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
print("\n=== K-Means Clustering Results ===")
print(f"Optimal k: {optimal_k}")
print(f"Silhouette Score: {silhouette_score(X_scaled, df_full['Cluster_KMeans']):.3f}")

cluster_summary = df_full.groupby('Cluster_KMeans').agg({
    'pH': 'mean',
    'DO': 'mean',
    'Turbidity': 'mean',
    'EC': 'mean',
    'Pollution_Score': 'mean',
    'Sampling_Point': lambda x: x.value_counts().index[0]  # Most common location
}).round(2)
print("\nCluster Profiles:")
print(cluster_summary)

# ============================================================================
# EXAMPLE 2: PCA (DIMENSIONALITY REDUCTION)
# ============================================================================

# Apply PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Variance explained
print("\n=== PCA Results ===")
print("Variance explained by each component:")
for i, var in enumerate(pca.explained_variance_ratio_[:5], 1):
    print(f"PC{i}: {var*100:.2f}%")
print(f"Cumulative variance (first 3 PCs): {pca.explained_variance_ratio_[:3].sum()*100:.2f}%")

# Plot explained variance
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, 11), pca.explained_variance_ratio_[:10])
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('PCA Variance Explained')

plt.subplot(1, 2, 2)
plt.plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_[:10]), 'ro-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance')
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/pca_variance.png', dpi=300)
plt.show()

# Visualize in 2D
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=df_full['Cluster_KMeans'], 
                     cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA: Samples colored by K-Means Cluster')
plt.colorbar(scatter, label='Cluster')

plt.subplot(1, 2, 2)
for point in df_full['Sampling_Point'].unique():
    mask = df_full['Sampling_Point'] == point
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=point, alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA: Samples colored by Sampling Point')
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/pca_2d_visualization.png', dpi=300)
plt.show()

# ============================================================================
# EXAMPLE 3: DBSCAN (DENSITY-BASED CLUSTERING)
# ============================================================================

# Apply DBSCAN
dbscan = DBSCAN(eps=2.5, min_samples=5)
df_full['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)

print("\n=== DBSCAN Results ===")
n_clusters = len(set(df_full['Cluster_DBSCAN'])) - (1 if -1 in df_full['Cluster_DBSCAN'] else 0)
n_noise = list(df_full['Cluster_DBSCAN']).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise} ({n_noise/len(df_full)*100:.1f}%)")

# ============================================================================
# EXAMPLE 4: ANOMALY DETECTION
# ============================================================================

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df_full['Anomaly'] = iso_forest.fit_predict(X_scaled)
df_full['Anomaly_Score'] = iso_forest.score_samples(X_scaled)

print("\n=== Anomaly Detection Results ===")
n_anomalies = (df_full['Anomaly'] == -1).sum()
print(f"Anomalies detected: {n_anomalies} ({n_anomalies/len(df_full)*100:.1f}%)")

# Show top 10 anomalies
anomalies = df_full[df_full['Anomaly'] == -1].sort_values('Anomaly_Score')
print("\nTop 10 Most Anomalous Samples:")
print(anomalies[['Date', 'Sampling_Point', 'pH', 'DO', 'Turbidity', 'Pollution_Score']].head(10))

# ============================================================================
# EXAMPLE 5: HIERARCHICAL CLUSTERING
# ============================================================================

from scipy.cluster.hierarchy import dendrogram, linkage

# Compute linkage
linkage_matrix = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(15, 7))
dendrogram(linkage_matrix, labels=df_full['Sampling_Point'].values)
plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('visualizations/hierarchical_dendrogram.png', dpi=300)
plt.show()

# Apply hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
df_full['Cluster_Hierarchical'] = hierarchical.fit_predict(X_scaled)

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save dataset with all cluster labels
df_full.to_csv('river_water_with_clusters.csv', index=False)
print("\n✓ Results saved to 'river_water_with_clusters.csv'")

# ============================================================================
# FEATURE IMPORTANCE (PCA Loadings)
# ============================================================================

# Get feature contributions to principal components
feature_names = X_scaled.columns
loadings = pd.DataFrame(
    pca.components_[:3].T,
    columns=['PC1', 'PC2', 'PC3'],
    index=feature_names
)

print("\n=== Top Features Contributing to Each PC ===")
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n{pc}:")
    print(loadings[pc].abs().sort_values(ascending=False).head(5))

# Visualize loadings
plt.figure(figsize=(12, 8))
sns.heatmap(loadings.T, cmap='RdBu_r', center=0, annot=True, fmt='.2f')
plt.title('PCA Loadings: Feature Contributions to Principal Components')
plt.tight_layout()
plt.savefig('visualizations/pca_loadings.png', dpi=300)
plt.show()

print("\n" + "="*80)
print("Analysis complete! Check the 'visualizations/' folder for all plots.")
print("="*80)
