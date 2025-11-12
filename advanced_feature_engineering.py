"""
Advanced Feature Engineering for River Water Quality Analysis
Creates domain-specific features for unsupervised learning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADVANCED FEATURE ENGINEERING - RIVER WATER QUALITY")
print("=" * 80)

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("\n[1] LOADING PREPROCESSED DATA...")

try:
    # Load the preprocessed data
    df = pd.read_csv('river_water_preprocessed.csv')
    print("✓ Data loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Existing features: {len(df.columns)}")
except FileNotFoundError:
    print("✗ Error: 'river_water_preprocessed.csv' not found")
    print("   Please run 'river_water_preprocessing.py' first")
    exit()

# Display current columns
print(f"\n   Current columns:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

# ============================================================================
# 2. DOMAIN-SPECIFIC WATER QUALITY FEATURES
# ============================================================================
print("\n[2] CREATING WATER QUALITY FEATURES...")

# 2.1 Water Quality Index (WQI) Components
print("\n   A. Water Quality Index Components:")

# DO Saturation Percentage (based on temperature)
# Theoretical DO saturation decreases with temperature
def calculate_do_saturation(do, temp):
    """Calculate DO as % of theoretical saturation"""
    # Simplified formula: DO_sat ≈ 14.6 - 0.41*temp (for freshwater)
    do_sat = 14.6 - (0.41 * temp)
    return (do / do_sat) * 100 if do_sat > 0 else 0

df['DO_Saturation_Percent'] = df.apply(
    lambda row: calculate_do_saturation(row['DO'], row['Sample_Temp']), axis=1
)
print("   ✓ DO_Saturation_Percent")

# pH Quality Score (7.0 is ideal, deviation is bad)
df['pH_Quality_Score'] = 100 - (np.abs(df['pH'] - 7.0) * 14.3)  # Max deviation ~7
df['pH_Quality_Score'] = df['pH_Quality_Score'].clip(0, 100)
print("   ✓ pH_Quality_Score")

# Turbidity Index (lower is better)
df['Turbidity_Index'] = 100 - (df['Turbidity'] / df['Turbidity'].quantile(0.95) * 100)
df['Turbidity_Index'] = df['Turbidity_Index'].clip(0, 100)
print("   ✓ Turbidity_Index")

# Conductivity Quality (moderate is good, extreme is bad)
ec_median = df['EC'].median()
df['EC_Quality_Score'] = 100 - (np.abs(df['EC'] - ec_median) / ec_median * 100)
df['EC_Quality_Score'] = df['EC_Quality_Score'].clip(0, 100)
print("   ✓ EC_Quality_Score")

# Overall Water Quality Index (composite)
df['WQI_Composite'] = (
    df['DO_Saturation_Percent'] * 0.25 +
    df['pH_Quality_Score'] * 0.25 +
    df['Turbidity_Index'] * 0.25 +
    df['EC_Quality_Score'] * 0.25
)
print("   ✓ WQI_Composite (0-100 scale)")

# 2.2 Pollution Indicators
print("\n   B. Pollution Indicators:")

# Combined pollution risk score
df['Pollution_Risk'] = (
    (df['Turbidity'] > df['Turbidity'].quantile(0.75)).astype(int) * 3 +
    (df['DO'] < df['DO'].quantile(0.25)).astype(int) * 3 +
    (df['pH_Deviation'] > 1.0).astype(int) * 2 +
    (df['Total_Chlorine'] > df['Total_Chlorine'].quantile(0.75)).astype(int) * 2
)
print("   ✓ Pollution_Risk (0-10 scale)")

# Eutrophication indicator (low DO + high nutrients)
df['Eutrophication_Index'] = (
    (10 - df['DO']) * 0.4 +  # Low DO
    (df['Turbidity'] / 100) * 0.3 +  # High turbidity
    (df['Total_Chlorine'] / 10) * 0.3  # Chlorine as nutrient proxy
)
print("   ✓ Eutrophication_Index")

# Heavy pollution flag
df['Heavy_Pollution_Flag'] = (
    (df['Turbidity'] > df['Turbidity'].quantile(0.90)) |
    (df['DO'] < df['DO'].quantile(0.10)) |
    (df['pH_Deviation'] > 1.5)
).astype(int)
print("   ✓ Heavy_Pollution_Flag")

# 2.3 Physical-Chemical Relationships
print("\n   C. Physical-Chemical Relationships:")

# TDS to EC ratio (should be ~0.5-0.7 for clean water)
df['TDS_EC_Ratio'] = df['TDS'] / (df['EC'] + 1)
df['TDS_EC_Anomaly'] = np.abs(df['TDS_EC_Ratio'] - 0.6)
print("   ✓ TDS_EC_Ratio and TDS_EC_Anomaly")

# Solids ratio (TSS to TDS)
df['TSS_TDS_Ratio'] = df['TSS'] / (df['TDS'] + 1)
print("   ✓ TSS_TDS_Ratio")

# Total solids estimation
df['Total_Solids_Est'] = df['TDS'] + df['TSS']
print("   ✓ Total_Solids_Est")

# Hardness per conductivity (mineral content indicator)
df['Hardness_EC_Ratio'] = df['Hardness'] / (df['EC'] + 1)
print("   ✓ Hardness_EC_Ratio")

# Ionic strength proxy (EC * Hardness)
df['Ionic_Strength_Proxy'] = np.log1p(df['EC'] * df['Hardness'])
print("   ✓ Ionic_Strength_Proxy")

# ============================================================================
# 3. TEMPORAL FEATURES
# ============================================================================
print("\n[3] CREATING TEMPORAL FEATURES...")

# 3.1 Cyclical encoding for time features
print("\n   A. Cyclical Time Encoding:")

# Month as sine/cosine (captures seasonal patterns)
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
print("   ✓ Month_Sin, Month_Cos")

# Day of week as sine/cosine
df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
print("   ✓ DayOfWeek_Sin, DayOfWeek_Cos")

# Day of year as sine/cosine
df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['Day'] / 365)
df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['Day'] / 365)
print("   ✓ DayOfYear_Sin, DayOfYear_Cos")

# 3.2 Time-based aggregations (if we have temporal sequence)
print("\n   B. Temporal Statistics:")

# Sort by sampling point and date
df = df.sort_values(['Sampling_Point', 'Date'])

# Calculate rolling statistics by location (7-day window)
temporal_cols = ['pH', 'DO', 'Turbidity', 'EC', 'TDS']

for col in temporal_cols:
    # Rolling mean (trend)
    df[f'{col}_Rolling_Mean_7d'] = df.groupby('Sampling_Point')[col].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Rolling std (volatility)
    df[f'{col}_Rolling_Std_7d'] = df.groupby('Sampling_Point')[col].transform(
        lambda x: x.rolling(window=3, min_periods=1).std()
    )
    
    # Change from previous measurement
    df[f'{col}_Change'] = df.groupby('Sampling_Point')[col].transform(
        lambda x: x.diff()
    )
    
    print(f"   ✓ {col}: Rolling Mean, Std, Change")

# ============================================================================
# 4. LOCATION-BASED FEATURES
# ============================================================================
print("\n[4] CREATING LOCATION-BASED FEATURES...")

# 4.1 Location statistics
print("\n   A. Location Statistics:")

location_stats = df.groupby('Sampling_Point').agg({
    'WQI_Composite': 'mean',
    'Pollution_Risk': 'mean',
    'Turbidity': 'mean',
    'DO': 'mean',
    'pH': 'mean'
}).add_suffix('_Location_Mean')

df = df.merge(location_stats, left_on='Sampling_Point', right_index=True, how='left')
print("   ✓ Location mean statistics merged")

# Deviation from location average
df['WQI_Deviation_From_Location'] = df['WQI_Composite'] - df['WQI_Composite_Location_Mean']
df['Pollution_Deviation_From_Location'] = df['Pollution_Risk'] - df['Pollution_Risk_Location_Mean']
print("   ✓ Deviation from location averages")

# Location rank (based on average water quality)
location_ranks = df.groupby('Sampling_Point')['WQI_Composite'].mean().rank(ascending=False)
df['Location_Quality_Rank'] = df['Sampling_Point'].map(location_ranks)
print("   ✓ Location_Quality_Rank")

# ============================================================================
# 5. ENVIRONMENTAL INTERACTION FEATURES
# ============================================================================
print("\n[5] CREATING ENVIRONMENTAL INTERACTION FEATURES...")

# Temperature impact on dissolved oxygen
df['Temp_DO_Interaction'] = df['Sample_Temp'] * (10 - df['DO'])
print("   ✓ Temp_DO_Interaction")

# Humidity impact on temperature difference
df['Humidity_TempDiff_Interaction'] = df['Ambient_Humidity'] * df['Temp_Difference']
print("   ✓ Humidity_TempDiff_Interaction")

# Season-pollution interaction
season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
df['Season_Numeric'] = df['Season'].map(season_mapping)
df['Season_Pollution_Interaction'] = df['Season_Numeric'] * df['Pollution_Risk']
print("   ✓ Season_Pollution_Interaction")

# Create IsWeekend feature if it doesn't exist
if 'IsWeekend' not in df.columns:
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Weekend vs weekday water quality
df['Weekend_WQI_Interaction'] = df['IsWeekend'] * df['WQI_Composite']
print("   ✓ Weekend_WQI_Interaction")

# ============================================================================
# 6. STATISTICAL FEATURES
# ============================================================================
print("\n[6] CREATING STATISTICAL FEATURES...")

# Z-scores for key parameters (anomaly detection)
key_params = ['pH', 'DO', 'Turbidity', 'EC', 'TDS', 'Total_Chlorine']

for param in key_params:
    df[f'{param}_ZScore'] = stats.zscore(df[param], nan_policy='omit')
    print(f"   ✓ {param}_ZScore")

# Overall anomaly score (sum of absolute z-scores)
z_score_cols = [f'{param}_ZScore' for param in key_params]
df['Total_Anomaly_Score'] = df[z_score_cols].abs().sum(axis=1)
print("   ✓ Total_Anomaly_Score")

# ============================================================================
# 7. RATIO AND LOG FEATURES
# ============================================================================
print("\n[7] CREATING RATIO AND LOG FEATURES...")

# Log transformations for skewed distributions
skewed_features = ['Turbidity', 'EC', 'TDS', 'Hardness', 'Total_Chlorine']

for feature in skewed_features:
    df[f'{feature}_Log'] = np.log1p(df[feature])
    print(f"   ✓ {feature}_Log")

# Square root transformations
df['Turbidity_Sqrt'] = np.sqrt(df['Turbidity'])
df['TSS_Sqrt'] = np.sqrt(df['TSS'])
print("   ✓ Turbidity_Sqrt, TSS_Sqrt")

# ============================================================================
# 8. CLUSTERING-READY FEATURES
# ============================================================================
print("\n[8] CREATING CLUSTERING-READY FEATURES...")

# Select features for unsupervised learning
feature_categories = {
    'water_quality': ['pH', 'DO', 'Turbidity', 'EC', 'TDS', 'TSS', 
                      'Hardness', 'Total_Chlorine'],
    'derived_quality': ['WQI_Composite', 'DO_Saturation_Percent', 'pH_Quality_Score',
                       'Turbidity_Index', 'Pollution_Risk'],
    'physical_chem': ['TDS_EC_Ratio', 'TSS_TDS_Ratio', 'Hardness_EC_Ratio',
                     'Ionic_Strength_Proxy'],
    'temporal': ['Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos'],
    'environmental': ['Ambient_Temp', 'Sample_Temp', 'Temp_Difference',
                     'Ambient_Humidity'],
    'anomaly': ['Total_Anomaly_Score', 'Eutrophication_Index']
}

# Combine all features
all_ml_features = []
for category, features in feature_categories.items():
    available_features = [f for f in features if f in df.columns]
    all_ml_features.extend(available_features)

print(f"\n   Total ML features: {len(all_ml_features)}")
print(f"   Categories: {len(feature_categories)}")

# ============================================================================
# 9. FEATURE SCALING (MULTIPLE METHODS)
# ============================================================================
print("\n[9] APPLYING FEATURE SCALING...")

# Create dataset with only numeric features for scaling
ml_features_df = df[all_ml_features].copy()

# 9.1 Standard Scaler (z-score normalization)
scaler_standard = StandardScaler()
df_standard_scaled = pd.DataFrame(
    scaler_standard.fit_transform(ml_features_df),
    columns=[f'{col}_StandardScaled' for col in all_ml_features],
    index=df.index
)
print("   ✓ StandardScaler applied")

# 9.2 MinMax Scaler (0-1 normalization)
scaler_minmax = MinMaxScaler()
df_minmax_scaled = pd.DataFrame(
    scaler_minmax.fit_transform(ml_features_df),
    columns=[f'{col}_MinMaxScaled' for col in all_ml_features],
    index=df.index
)
print("   ✓ MinMaxScaler applied")

# 9.3 Robust Scaler (median and IQR, robust to outliers)
scaler_robust = RobustScaler()
df_robust_scaled = pd.DataFrame(
    scaler_robust.fit_transform(ml_features_df),
    columns=[f'{col}_RobustScaled' for col in all_ml_features],
    index=df.index
)
print("   ✓ RobustScaler applied")

# ============================================================================
# 10. FEATURE SUMMARY AND STATISTICS
# ============================================================================
print("\n[10] FEATURE SUMMARY")
print("-" * 80)

new_features = [col for col in df.columns if col not in pd.read_csv('river_water_preprocessed.csv').columns]
print(f"\n   New features created: {len(new_features)}")
print(f"   Total features now: {len(df.columns)}")
print(f"\n   Feature breakdown:")
for category, features in feature_categories.items():
    available = [f for f in features if f in df.columns]
    print(f"   - {category}: {len(available)} features")

# Feature quality check
print(f"\n   Feature quality:")
print(f"   - Missing values: {df[all_ml_features].isnull().sum().sum()}")
print(f"   - Infinite values: {np.isinf(df[all_ml_features]).sum().sum()}")

# Replace any inf values with NaN and fill
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
print("   ✓ Cleaned infinite and missing values")

# ============================================================================
# 11. SAVE ENGINEERED FEATURES
# ============================================================================
print("\n[11] SAVING ENGINEERED FEATURES...")

# Save full dataset with all features
df.to_csv('river_water_features_engineered.csv', index=False, encoding='utf-8')
print("✓ Saved: river_water_features_engineered.csv")

# Save ML-ready features only (unscaled)
ml_features_df.to_csv('ml_features_unscaled.csv', index=False, encoding='utf-8')
print("✓ Saved: ml_features_unscaled.csv")

# Save scaled versions
df_standard_scaled.to_csv('ml_features_standard_scaled.csv', index=False, encoding='utf-8')
print("✓ Saved: ml_features_standard_scaled.csv")

df_minmax_scaled.to_csv('ml_features_minmax_scaled.csv', index=False, encoding='utf-8')
print("✓ Saved: ml_features_minmax_scaled.csv")

df_robust_scaled.to_csv('ml_features_robust_scaled.csv', index=False, encoding='utf-8')
print("✓ Saved: ml_features_robust_scaled.csv")

# Save feature metadata
feature_metadata = {
    'total_features': len(df.columns),
    'new_features': len(new_features),
    'ml_features': len(all_ml_features),
    'categories': feature_categories,
    'all_ml_features': all_ml_features,
    'new_feature_list': new_features
}

import json
with open('feature_engineering_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(feature_metadata, f, indent=2, default=str)
print("✓ Saved: feature_engineering_metadata.json")

# Create feature summary report
print("\n   Creating feature summary report...")
feature_summary = pd.DataFrame({
    'Feature': all_ml_features,
    'Mean': ml_features_df.mean(),
    'Std': ml_features_df.std(),
    'Min': ml_features_df.min(),
    'Max': ml_features_df.max(),
    'Skewness': ml_features_df.skew(),
    'Kurtosis': ml_features_df.kurtosis()
})
feature_summary.to_csv('feature_summary_statistics.csv', index=False, encoding='utf-8')
print("✓ Saved: feature_summary_statistics.csv")

# ============================================================================
# 12. FEATURE IMPORTANCE INSIGHTS
# ============================================================================
print("\n[12] FEATURE INSIGHTS")
print("-" * 80)

print("\n   Top 10 Most Variable Features (by CV):")
cv = (ml_features_df.std() / ml_features_df.mean()).abs().sort_values(ascending=False)
for i, (feature, value) in enumerate(cv.head(10).items(), 1):
    print(f"   {i:2d}. {feature}: CV = {value:.2f}")

print("\n   Top 10 Most Skewed Features:")
skewness = ml_features_df.skew().abs().sort_values(ascending=False)
for i, (feature, value) in enumerate(skewness.head(10).items(), 1):
    print(f"   {i:2d}. {feature}: Skew = {value:.2f}")

print("\n   Water Quality Summary:")
print(f"   - Average WQI: {df['WQI_Composite'].mean():.2f}")
print(f"   - Best location: {df.groupby('Sampling_Point')['WQI_Composite'].mean().idxmax()}")
print(f"   - Worst location: {df.groupby('Sampling_Point')['WQI_Composite'].mean().idxmin()}")
print(f"   - High pollution events: {df['Heavy_Pollution_Flag'].sum()}")

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETED!")
print("=" * 80)
print("\nOutput files:")
print("   1. river_water_features_engineered.csv - All features")
print("   2. ml_features_unscaled.csv - ML features (unscaled)")
print("   3. ml_features_standard_scaled.csv - StandardScaler normalized")
print("   4. ml_features_minmax_scaled.csv - MinMax normalized (0-1)")
print("   5. ml_features_robust_scaled.csv - Robust scaler (outlier-resistant)")
print("   6. feature_engineering_metadata.json - Feature catalog")
print("   7. feature_summary_statistics.csv - Descriptive statistics")
print("\nReady for unsupervised learning:")
print("   - K-Means Clustering")
print("   - DBSCAN (density-based)")
print("   - PCA (dimensionality reduction)")
print("   - Isolation Forest (anomaly detection)")
print("   - Hierarchical Clustering")
