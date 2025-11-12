"""
Comprehensive Unsupervised Learning Analysis
River Water Quality Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("UNSUPERVISED LEARNING ANALYSIS - RIVER WATER QUALITY")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")

# Load ML-ready features (using robust scaling for water quality data with outliers)
df_scaled = pd.read_csv('ml_features_robust_scaled.csv')
df_original = pd.read_csv('river_water_features_engineered.csv')

print(f"✓ Scaled features loaded: {df_scaled.shape}")
print(f"✓ Original data loaded: {df_original.shape}")

# Get feature names (remove scaling suffix)
feature_names = [col.replace('_RobustScaled', '') for col in df_scaled.columns]
print(f"✓ Features for analysis: {len(feature_names)}")

# Create output directory
import os
os.makedirs('unsupervised_results', exist_ok=True)
print("✓ Output directory created: unsupervised_results/")

# ============================================================================
# 2. ELBOW METHOD - OPTIMAL K FOR K-MEANS
# ============================================================================
print("\n[2] DETERMINING OPTIMAL NUMBER OF CLUSTERS...")

# Calculate inertia for different k values
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))

# Plot Elbow Curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Elbow plot
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=10)
ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
ax1.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(k_range)

# Silhouette plot
ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=10)
ax2.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax2.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(k_range)
ax2.axhline(y=0.5, color='r', linestyle='--', label='Good threshold (0.5)')
ax2.legend()

plt.tight_layout()
plt.savefig('unsupervised_results/01_optimal_k_selection.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_optimal_k_selection.png")
plt.close()

# Find optimal k (highest silhouette score)
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\n   Optimal K (by Silhouette): {optimal_k}")
print(f"   Silhouette Score: {max(silhouette_scores):.3f}")

# ============================================================================
# 3. K-MEANS CLUSTERING
# ============================================================================
print(f"\n[3] K-MEANS CLUSTERING (k={optimal_k})...")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(df_scaled)

# Add cluster labels to original data
df_original['KMeans_Cluster'] = kmeans_labels

# Calculate clustering metrics
silhouette_kmeans = silhouette_score(df_scaled, kmeans_labels)
davies_bouldin_kmeans = davies_bouldin_score(df_scaled, kmeans_labels)
calinski_kmeans = calinski_harabasz_score(df_scaled, kmeans_labels)

print(f"   Silhouette Score: {silhouette_kmeans:.3f}")
print(f"   Davies-Bouldin Index: {davies_bouldin_kmeans:.3f} (lower is better)")
print(f"   Calinski-Harabasz Index: {calinski_kmeans:.3f} (higher is better)")

# Cluster distribution
print(f"\n   Cluster Distribution:")
cluster_counts = pd.Series(kmeans_labels).value_counts().sort_index()
for cluster, count in cluster_counts.items():
    print(f"   - Cluster {cluster}: {count} samples ({count/len(kmeans_labels)*100:.1f}%)")

# ============================================================================
# 4. HIERARCHICAL CLUSTERING
# ============================================================================
print("\n[4] HIERARCHICAL CLUSTERING...")

# Perform hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(df_scaled)

df_original['Hierarchical_Cluster'] = hierarchical_labels

# Calculate metrics
silhouette_hier = silhouette_score(df_scaled, hierarchical_labels)
print(f"   Silhouette Score: {silhouette_hier:.3f}")

# Create dendrogram
plt.figure(figsize=(16, 8))
linkage_matrix = linkage(df_scaled, method='ward')
dendrogram(linkage_matrix, 
           truncate_mode='lastp',
           p=30,
           leaf_font_size=10,
           show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Sample Index or (Cluster Size)', fontsize=12, fontweight='bold')
plt.ylabel('Distance', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('unsupervised_results/02_hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_hierarchical_dendrogram.png")
plt.close()

# ============================================================================
# 5. DBSCAN CLUSTERING
# ============================================================================
print("\n[5] DBSCAN CLUSTERING (Density-Based)...")

# Try different eps values
eps_values = [0.5, 1.0, 1.5, 2.0]
best_eps = None
best_score = -1

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(df_scaled)
    
    # Skip if only noise or only one cluster
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        score = silhouette_score(df_scaled[labels != -1], labels[labels != -1])
        if score > best_score:
            best_score = score
            best_eps = eps

# Use best eps
dbscan = DBSCAN(eps=best_eps, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_scaled)

df_original['DBSCAN_Cluster'] = dbscan_labels

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"   Optimal eps: {best_eps}")
print(f"   Number of clusters: {n_clusters_dbscan}")
print(f"   Noise points: {n_noise} ({n_noise/len(dbscan_labels)*100:.1f}%)")
if n_clusters_dbscan > 1 and n_noise < len(dbscan_labels):
    valid_labels = dbscan_labels[dbscan_labels != -1]
    if len(set(valid_labels)) > 1:
        silhouette_dbscan = silhouette_score(df_scaled[dbscan_labels != -1], valid_labels)
        print(f"   Silhouette Score: {silhouette_dbscan:.3f}")

# ============================================================================
# 6. PCA - DIMENSIONALITY REDUCTION
# ============================================================================
print("\n[6] PCA - PRINCIPAL COMPONENT ANALYSIS...")

# Perform PCA
pca = PCA()
pca_components = pca.fit_transform(df_scaled)

# Calculate cumulative variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find components for 90% variance
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"   Components for 90% variance: {n_components_90}")
print(f"   Total variance explained: {cumulative_variance[n_components_90-1]:.3f}")

# Plot variance explained
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Scree plot
n_components_to_show = min(20, len(pca.explained_variance_ratio_))
ax1.bar(range(1, n_components_to_show + 1), 
        pca.explained_variance_ratio_[:n_components_to_show], alpha=0.7, color='steelblue')
ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax1.set_ylabel('Variance Explained Ratio', fontsize=12, fontweight='bold')
ax1.set_title(f'PCA Scree Plot (First {n_components_to_show} Components)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Cumulative variance
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
         'o-', linewidth=2, markersize=6, color='darkgreen')
ax2.axhline(y=0.90, color='r', linestyle='--', label='90% variance')
ax2.axvline(x=n_components_90, color='orange', linestyle='--', 
            label=f'{n_components_90} components')
ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Variance Explained', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('unsupervised_results/03_pca_variance_explained.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_pca_variance_explained.png")
plt.close()

# Save first 2 components for visualization
df_original['PCA_1'] = pca_components[:, 0]
df_original['PCA_2'] = pca_components[:, 1]
if pca_components.shape[1] > 2:
    df_original['PCA_3'] = pca_components[:, 2]

# ============================================================================
# 7. ISOLATION FOREST - ANOMALY DETECTION
# ============================================================================
print("\n[7] ISOLATION FOREST - ANOMALY DETECTION...")

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(df_scaled)
anomaly_scores = iso_forest.score_samples(df_scaled)

df_original['Anomaly_Label'] = anomaly_labels  # -1 = anomaly, 1 = normal
df_original['Anomaly_Score'] = anomaly_scores

n_anomalies = (anomaly_labels == -1).sum()
print(f"   Anomalies detected: {n_anomalies} ({n_anomalies/len(anomaly_labels)*100:.1f}%)")

# Identify most anomalous samples
top_anomalies = df_original.nsmallest(10, 'Anomaly_Score')[
    ['Date', 'Sampling_Point', 'WQI_Composite', 'Pollution_Risk', 'Anomaly_Score']
]
print(f"\n   Top 10 Most Anomalous Samples:")
print(top_anomalies.to_string(index=False))

# ============================================================================
# 8. VISUALIZATION - CLUSTER COMPARISON
# ============================================================================
print("\n[8] CREATING VISUALIZATIONS...")

# 8.1 PCA Visualization with K-Means Clusters
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# K-Means clusters in PCA space
scatter1 = axes[0, 0].scatter(df_original['PCA_1'], df_original['PCA_2'],
                              c=df_original['KMeans_Cluster'], cmap='viridis',
                              s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0, 0].set_xlabel('First Principal Component', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Second Principal Component', fontsize=11, fontweight='bold')
axes[0, 0].set_title('K-Means Clustering (PCA Projection)', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')

# Hierarchical clusters in PCA space
scatter2 = axes[0, 1].scatter(df_original['PCA_1'], df_original['PCA_2'],
                              c=df_original['Hierarchical_Cluster'], cmap='plasma',
                              s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0, 1].set_xlabel('First Principal Component', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Second Principal Component', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Hierarchical Clustering (PCA Projection)', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')

# DBSCAN clusters in PCA space
scatter3 = axes[1, 0].scatter(df_original['PCA_1'], df_original['PCA_2'],
                              c=df_original['DBSCAN_Cluster'], cmap='tab10',
                              s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[1, 0].set_xlabel('First Principal Component', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Second Principal Component', fontsize=11, fontweight='bold')
axes[1, 0].set_title('DBSCAN Clustering (PCA Projection)', fontsize=12, fontweight='bold')
plt.colorbar(scatter3, ax=axes[1, 0], label='Cluster (-1=Noise)')

# Anomaly detection in PCA space
scatter4 = axes[1, 1].scatter(df_original['PCA_1'], df_original['PCA_2'],
                              c=df_original['Anomaly_Score'], cmap='RdYlGn',
                              s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[1, 1].set_xlabel('First Principal Component', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Second Principal Component', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Anomaly Detection (Red=Anomalous, Green=Normal)', fontsize=12, fontweight='bold')
plt.colorbar(scatter4, ax=axes[1, 1], label='Anomaly Score')

plt.tight_layout()
plt.savefig('unsupervised_results/04_clustering_comparison_pca.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_clustering_comparison_pca.png")
plt.close()

# 8.2 Cluster Profiling - K-Means
print("\n   Creating cluster profiles...")

# Select key features for profiling
profile_features = ['WQI_Composite', 'Pollution_Risk', 'pH', 'DO', 'Turbidity', 
                    'EC', 'Total_Anomaly_Score', 'DO_Saturation_Percent']

cluster_profiles = df_original.groupby('KMeans_Cluster')[profile_features].mean()

fig, ax = plt.subplots(figsize=(14, 8))
cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
sns.heatmap(cluster_profiles_norm.T, annot=cluster_profiles.T, fmt='.2f',
            cmap='RdYlGn', cbar_kws={'label': 'Normalized Value'},
            linewidths=0.5, ax=ax)
ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('K-Means Cluster Profiles (Cell values = actual, Color = normalized)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('unsupervised_results/05_kmeans_cluster_profiles.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_kmeans_cluster_profiles.png")
plt.close()

# 8.3 Location distribution across clusters
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# K-Means by location
location_cluster = pd.crosstab(df_original['Sampling_Point'], 
                               df_original['KMeans_Cluster'], 
                               normalize='index') * 100
location_cluster.plot(kind='bar', stacked=True, ax=axes[0], 
                      colormap='viridis', edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Sampling Location', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
axes[0].set_title('K-Means Cluster Distribution by Location', fontsize=12, fontweight='bold')
axes[0].legend(title='Cluster', bbox_to_anchor=(1.05, 1))
axes[0].tick_params(axis='x', rotation=45)

# Anomalies by location
anomaly_by_location = df_original.groupby('Sampling_Point')['Anomaly_Label'].apply(
    lambda x: (x == -1).sum() / len(x) * 100
).sort_values(ascending=False)
anomaly_by_location.plot(kind='bar', ax=axes[1], color='coral', 
                         edgecolor='black', linewidth=1)
axes[1].set_xlabel('Sampling Location', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Anomaly Percentage (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Anomaly Detection by Location', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].axhline(y=10, color='red', linestyle='--', label='Expected (10%)')
axes[1].legend()

plt.tight_layout()
plt.savefig('unsupervised_results/06_location_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_location_analysis.png")
plt.close()

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n[9] SAVING RESULTS...")

# Save clustered data
df_original.to_csv('unsupervised_results/clustered_data_complete.csv', 
                   index=False, encoding='utf-8')
print("✓ Saved: clustered_data_complete.csv")

# Save cluster profiles
cluster_profiles.to_csv('unsupervised_results/kmeans_cluster_profiles.csv', 
                        encoding='utf-8')
print("✓ Saved: kmeans_cluster_profiles.csv")

# Save PCA loadings (top features per component)
pca_loadings = pd.DataFrame(
    pca.components_[:5].T,
    columns=[f'PC{i+1}' for i in range(5)],
    index=feature_names
)
top_features_per_pc = pd.DataFrame()
for col in pca_loadings.columns:
    top_features_per_pc[col] = pca_loadings[col].abs().nlargest(10).index.tolist()

top_features_per_pc.to_csv('unsupervised_results/pca_top_features.csv', 
                            encoding='utf-8')
print("✓ Saved: pca_top_features.csv")

# Save clustering metrics
metrics_summary = pd.DataFrame({
    'Algorithm': ['K-Means', 'Hierarchical', 'DBSCAN'],
    'Silhouette_Score': [silhouette_kmeans, silhouette_hier, 
                         silhouette_dbscan if n_clusters_dbscan > 1 else np.nan],
    'N_Clusters': [optimal_k, optimal_k, n_clusters_dbscan],
    'Davies_Bouldin': [davies_bouldin_kmeans, np.nan, np.nan],
    'Calinski_Harabasz': [calinski_kmeans, np.nan, np.nan]
})
metrics_summary.to_csv('unsupervised_results/clustering_metrics_summary.csv', 
                       index=False, encoding='utf-8')
print("✓ Saved: clustering_metrics_summary.csv")

# ============================================================================
# 10. SUMMARY REPORT
# ============================================================================
print("\n[10] GENERATING SUMMARY REPORT...")

report = []
report.append("=" * 80)
report.append("UNSUPERVISED LEARNING ANALYSIS - SUMMARY REPORT")
report.append("=" * 80)
report.append("")
report.append("DATASET INFORMATION")
report.append("-" * 80)
report.append(f"Total Samples: {len(df_original)}")
report.append(f"Features Used: {len(feature_names)}")
report.append(f"Locations: {df_original['Sampling_Point'].nunique()}")
report.append("")

report.append("CLUSTERING RESULTS")
report.append("-" * 80)
report.append(f"\n1. K-MEANS CLUSTERING")
report.append(f"   Optimal K: {optimal_k}")
report.append(f"   Silhouette Score: {silhouette_kmeans:.3f}")
report.append(f"   Davies-Bouldin Index: {davies_bouldin_kmeans:.3f}")
report.append(f"   Calinski-Harabasz Index: {calinski_kmeans:.3f}")
report.append(f"\n   Cluster Distribution:")
for cluster in range(optimal_k):
    count = (kmeans_labels == cluster).sum()
    avg_wqi = df_original[df_original['KMeans_Cluster'] == cluster]['WQI_Composite'].mean()
    report.append(f"   - Cluster {cluster}: {count} samples ({count/len(kmeans_labels)*100:.1f}%) | Avg WQI: {avg_wqi:.2f}")

report.append(f"\n2. HIERARCHICAL CLUSTERING")
report.append(f"   Linkage Method: Ward")
report.append(f"   Silhouette Score: {silhouette_hier:.3f}")

report.append(f"\n3. DBSCAN CLUSTERING")
report.append(f"   Optimal eps: {best_eps}")
report.append(f"   Clusters Found: {n_clusters_dbscan}")
report.append(f"   Noise Points: {n_noise} ({n_noise/len(dbscan_labels)*100:.1f}%)")

report.append("")
report.append("DIMENSIONALITY REDUCTION")
report.append("-" * 80)
report.append(f"PCA Components for 90% Variance: {n_components_90}")
report.append(f"Total Variance Explained: {cumulative_variance[n_components_90-1]:.3f}")
report.append(f"\nTop 3 PC Variance:")
for i in range(min(3, len(pca.explained_variance_ratio_))):
    report.append(f"   PC{i+1}: {pca.explained_variance_ratio_[i]:.3f} ({pca.explained_variance_ratio_[i]*100:.1f}%)")

report.append("")
report.append("ANOMALY DETECTION")
report.append("-" * 80)
report.append(f"Algorithm: Isolation Forest")
report.append(f"Anomalies Detected: {n_anomalies} ({n_anomalies/len(anomaly_labels)*100:.1f}%)")
report.append(f"\nAnomalies by Location:")
for location, pct in anomaly_by_location.items():
    report.append(f"   - {location}: {pct:.1f}%")

report.append("")
report.append("KEY INSIGHTS")
report.append("-" * 80)

# Identify cluster characteristics
for cluster in range(optimal_k):
    cluster_data = df_original[df_original['KMeans_Cluster'] == cluster]
    avg_wqi = cluster_data['WQI_Composite'].mean()
    avg_pollution = cluster_data['Pollution_Risk'].mean()
    dominant_location = cluster_data['Sampling_Point'].mode()[0]
    
    if avg_wqi > 80:
        quality = "Excellent"
    elif avg_wqi > 70:
        quality = "Good"
    elif avg_wqi > 50:
        quality = "Moderate"
    else:
        quality = "Poor"
    
    report.append(f"\nCluster {cluster}: {quality} Water Quality")
    report.append(f"   - Average WQI: {avg_wqi:.2f}")
    report.append(f"   - Average Pollution Risk: {avg_pollution:.2f}")
    report.append(f"   - Dominant Location: {dominant_location}")
    report.append(f"   - Sample Count: {len(cluster_data)}")

report.append("")
report.append("FILES GENERATED")
report.append("-" * 80)
report.append("   1. 01_optimal_k_selection.png - Elbow and Silhouette plots")
report.append("   2. 02_hierarchical_dendrogram.png - Hierarchical clustering tree")
report.append("   3. 03_pca_variance_explained.png - PCA scree and cumulative variance")
report.append("   4. 04_clustering_comparison_pca.png - All algorithms in PCA space")
report.append("   5. 05_kmeans_cluster_profiles.png - Cluster characteristics heatmap")
report.append("   6. 06_location_analysis.png - Location-based clustering insights")
report.append("   7. clustered_data_complete.csv - Full dataset with cluster labels")
report.append("   8. kmeans_cluster_profiles.csv - Average features per cluster")
report.append("   9. pca_top_features.csv - Most important features per PC")
report.append("  10. clustering_metrics_summary.csv - Performance metrics")
report.append("")
report.append("=" * 80)
report.append("ANALYSIS COMPLETE")
report.append("=" * 80)

# Print and save report
report_text = "\n".join(report)
print("\n" + report_text)

with open('unsupervised_results/ANALYSIS_SUMMARY_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("\n✓ Saved: ANALYSIS_SUMMARY_REPORT.txt")

print("\n" + "=" * 80)
print("UNSUPERVISED LEARNING ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nAll results saved in: unsupervised_results/")
