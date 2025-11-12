import pandas as pd

df = pd.read_csv('unsupervised_results/clustered_data_complete.csv')

print("=" * 70)
print("CLUSTERED DATA VERIFICATION")
print("=" * 70)

print(f"\nDataset Shape: {df.shape}")
print(f"  Rows: {df.shape[0]}")
print(f"  Columns: {df.shape[1]}")

print(f"\nClustering Columns Added:")
print(f"  - KMeans_Cluster")
print(f"  - Hierarchical_Cluster")
print(f"  - DBSCAN_Cluster")
print(f"  - Anomaly_Label")
print(f"  - Anomaly_Score")
print(f"  - PCA_1, PCA_2, PCA_3")

print(f"\nK-Means Cluster Distribution:")
for cluster in sorted(df['KMeans_Cluster'].unique()):
    count = (df['KMeans_Cluster'] == cluster).sum()
    avg_wqi = df[df['KMeans_Cluster'] == cluster]['WQI_Composite'].mean()
    print(f"  Cluster {cluster}: {count} samples ({count/len(df)*100:.1f}%) | Avg WQI: {avg_wqi:.2f}")

print(f"\nAnomalies Detected:")
n_anomalies = (df['Anomaly_Label'] == -1).sum()
print(f"  Total: {n_anomalies} ({n_anomalies/len(df)*100:.1f}%)")

print(f"\nTop 5 Anomalies by Location:")
anomalies = df[df['Anomaly_Label'] == -1].groupby('Sampling_Point').size().sort_values(ascending=False)
for loc, count in anomalies.head().items():
    print(f"  {loc}: {count} anomalies")

print("=" * 70)
