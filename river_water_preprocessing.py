"""
River Water Quality - Data Preprocessing and Preparation
for Unsupervised Learning Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

print("=" * 80)
print("RIVER WATER QUALITY - DATA PREPROCESSING & PREPARATION")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")
try:
    df = pd.read_csv(r'c:\Users\Millpark\Downloads\River water parameters.csv')
    print("✓ Data loaded successfully!")
    print(f"   Shape: {df.shape}")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    exit()

# ============================================================================
# 2. INITIAL DATA EXPLORATION
# ============================================================================
print("\n[2] INITIAL DATA EXPLORATION")
print("-" * 80)

print("\n📊 Dataset Info:")
print(f"   Rows: {df.shape[0]}")
print(f"   Columns: {df.shape[1]}")
print(f"\n   Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print("\n📊 First Few Rows:")
print(df.head())

print("\n📊 Data Types:")
print(df.dtypes)

print("\n📊 Basic Statistics:")
print(df.describe())

# ============================================================================
# 3. DATA QUALITY ASSESSMENT
# ============================================================================
print("\n[3] DATA QUALITY ASSESSMENT")
print("-" * 80)

# Check for missing values
print("\n📌 Missing Values:")
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
if len(missing_data) > 0:
    print(missing_data.to_string(index=False))
else:
    print("   ✓ No missing values found!")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\n📌 Duplicate Rows: {duplicates}")

# ============================================================================
# 4. COLUMN RENAMING FOR EASIER HANDLING
# ============================================================================
print("\n[4] CLEANING COLUMN NAMES...")

# Create a mapping for cleaner column names
column_mapping = {
    'Date (DD/MM/YYYY)': 'Date',
    'Time (24 hrs XX:XX)': 'Time',
    'Sampling point': 'Sampling_Point',
    'Ambient temperature (°C)': 'Ambient_Temp',
    'Ambient humidity': 'Ambient_Humidity',
    'Sample temperature (°C)': 'Sample_Temp',
    'pH': 'pH',
    'EC\n(µS/cm)': 'EC',
    'TDS\n(mg/L)': 'TDS',
    'TSS\n(mL sed/L)': 'TSS',
    'DO\n(mg/L)': 'DO',
    'Level (cm)': 'Level',
    'Turbidity (NTU)': 'Turbidity',
    'Hardness\n(mg CaCO3/L)': 'Hardness',
    'Hardness classification': 'Hardness_Class',
    'Total Cl-\n(mg Cl-/L)': 'Total_Chlorine'
}

df = df.rename(columns=column_mapping)
print("✓ Column names cleaned!")
print(f"   New columns: {list(df.columns)}")

# ============================================================================
# 5. DATA TYPE CONVERSION
# ============================================================================
print("\n[5] CONVERTING DATA TYPES...")

# Convert Date and Time
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
print("✓ Date converted to datetime")

# Extract temporal features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] 
                                   else 'Spring' if x in [3, 4, 5]
                                   else 'Summer' if x in [6, 7, 8]
                                   else 'Autumn')
print("✓ Temporal features extracted")

# Convert Time to minutes since midnight
df['Time_Minutes'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour * 60 + \
                     pd.to_datetime(df['Time'], format='%H:%M').dt.minute
print("✓ Time converted to minutes")

# ============================================================================
# 6. HANDLE MISSING VALUES
# ============================================================================
print("\n[6] HANDLING MISSING VALUES...")

# Identify numeric and categorical columns
numeric_cols = ['Ambient_Temp', 'Ambient_Humidity', 'Sample_Temp', 'pH', 
                'EC', 'TDS', 'TSS', 'DO', 'Level', 'Turbidity', 
                'Hardness', 'Total_Chlorine']
categorical_cols = ['Sampling_Point', 'Hardness_Class']

# Check missing values before imputation
print("\n   Missing values before imputation:")
for col in numeric_cols:
    missing = df[col].isnull().sum()
    if missing > 0:
        print(f"   {col}: {missing} ({(missing/len(df)*100):.2f}%)")

# Strategy: Use median for numeric columns (robust to outliers)
# For water quality data, median is often more appropriate than mean
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# For categorical, use mode (most frequent)
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"   ✓ {col} filled with mode: {mode_value}")

print("\n✓ Missing values handled!")
print(f"   Remaining missing values: {df.isnull().sum().sum()}")

# ============================================================================
# 7. OUTLIER DETECTION
# ============================================================================
print("\n[7] OUTLIER DETECTION...")

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

print("\n   Outliers detected (IQR method):")
outlier_summary = []
for col in numeric_cols:
    n_outliers, lower, upper = detect_outliers_iqr(df, col)
    if n_outliers > 0:
        outlier_summary.append({
            'Column': col,
            'N_Outliers': n_outliers,
            'Percentage': f"{(n_outliers/len(df)*100):.2f}%",
            'Range': f"[{lower:.2f}, {upper:.2f}]"
        })
        print(f"   {col}: {n_outliers} outliers ({(n_outliers/len(df)*100):.2f}%)")

# Note: We'll keep outliers as they may represent real pollution events
print("\n   ⚠ Note: Outliers retained - may indicate pollution events")

# ============================================================================
# 8. FEATURE ENGINEERING
# ============================================================================
print("\n[8] FEATURE ENGINEERING...")

# 1. Temperature Difference (Sample vs Ambient)
df['Temp_Difference'] = df['Sample_Temp'] - df['Ambient_Temp']
print("   ✓ Created: Temp_Difference")

# 2. Water Quality Index Components
# Normalize parameters to 0-100 scale based on water quality standards
def calculate_do_index(do_value):
    """Dissolved Oxygen index (higher is better)"""
    if do_value >= 6:
        return 100
    elif do_value >= 4:
        return 80
    elif do_value >= 2:
        return 60
    else:
        return 40

df['DO_Index'] = df['DO'].apply(calculate_do_index)
print("   ✓ Created: DO_Index")

# 3. pH deviation from neutral (7.0)
df['pH_Deviation'] = np.abs(df['pH'] - 7.0)
print("   ✓ Created: pH_Deviation")

# 4. Conductivity to TDS ratio
df['EC_TDS_Ratio'] = df['EC'] / (df['TDS'] + 1)  # +1 to avoid division by zero
print("   ✓ Created: EC_TDS_Ratio")

# 5. Pollution indicators
# High turbidity, low DO, extreme pH indicate pollution
df['Pollution_Score'] = (
    (df['Turbidity'] / df['Turbidity'].max() * 30) +  # Turbidity contribution
    ((1 - df['DO'] / 10) * 40) +  # DO contribution (inverted)
    (df['pH_Deviation'] * 30)  # pH deviation contribution
)
print("   ✓ Created: Pollution_Score")

# 6. Encode categorical variables
le_sampling = LabelEncoder()
df['Sampling_Point_Encoded'] = le_sampling.fit_transform(df['Sampling_Point'])
print(f"   ✓ Encoded Sampling_Point: {dict(enumerate(le_sampling.classes_))}")

le_hardness = LabelEncoder()
df['Hardness_Class_Encoded'] = le_hardness.fit_transform(df['Hardness_Class'])
print(f"   ✓ Encoded Hardness_Class: {dict(enumerate(le_hardness.classes_))}")

le_season = LabelEncoder()
df['Season_Encoded'] = le_season.fit_transform(df['Season'])
print(f"   ✓ Encoded Season: {dict(enumerate(le_season.classes_))}")

# ============================================================================
# 9. SAMPLING POINT ANALYSIS
# ============================================================================
print("\n[9] SAMPLING POINT ANALYSIS...")

sampling_stats = df.groupby('Sampling_Point').agg({
    'pH': ['mean', 'std'],
    'DO': ['mean', 'std'],
    'Turbidity': ['mean', 'std'],
    'EC': ['mean', 'std'],
    'Pollution_Score': 'mean'
}).round(2)

print("\n   Water Quality by Sampling Point:")
print(sampling_stats)

# ============================================================================
# 10. CREATE FEATURE MATRIX FOR UNSUPERVISED LEARNING
# ============================================================================
print("\n[10] CREATING FEATURE MATRIX FOR UNSUPERVISED LEARNING...")

# Select features for clustering/dimensionality reduction
feature_columns = [
    # Water quality parameters
    'pH', 'EC', 'TDS', 'TSS', 'DO', 'Turbidity', 'Hardness', 'Total_Chlorine',
    # Environmental factors
    'Ambient_Temp', 'Ambient_Humidity', 'Sample_Temp',
    # Engineered features
    'Temp_Difference', 'DO_Index', 'pH_Deviation', 'EC_TDS_Ratio', 'Pollution_Score',
    # Spatial
    'Sampling_Point_Encoded',
    # Temporal
    'Month', 'Season_Encoded', 'Time_Minutes'
]

# Create feature matrix
X = df[feature_columns].copy()
print(f"✓ Feature matrix created: {X.shape}")
print(f"   Features: {len(feature_columns)}")
print(f"   Samples: {len(X)}")

# ============================================================================
# 11. FEATURE SCALING
# ============================================================================
print("\n[11] FEATURE SCALING (STANDARDIZATION)...")

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)

print("✓ Features scaled using StandardScaler")
print(f"   Mean: {X_scaled.mean():.2e}")
print(f"   Std: {X_scaled.std():.2e}")

# ============================================================================
# 12. SAVE PREPROCESSED DATA
# ============================================================================
print("\n[12] SAVING PREPROCESSED DATA...")

# Save full preprocessed dataframe
df.to_csv(r'c:\Users\Millpark\Downloads\Unsupervised Learning Project11\river_water_preprocessed.csv', 
          index=False)
print("✓ Saved: river_water_preprocessed.csv")

# Save feature matrix (unscaled)
X.to_csv(r'c:\Users\Millpark\Downloads\Unsupervised Learning Project11\river_water_features.csv', 
         index=False)
print("✓ Saved: river_water_features.csv")

# Save scaled feature matrix
X_scaled_df.to_csv(r'c:\Users\Millpark\Downloads\Unsupervised Learning Project11\river_water_features_scaled.csv', 
                   index=False)
print("✓ Saved: river_water_features_scaled.csv")

# Save preprocessing metadata
metadata = {
    'sampling_points': le_sampling.classes_.tolist(),
    'hardness_classes': le_hardness.classes_.tolist(),
    'seasons': le_season.classes_.tolist(),
    'feature_columns': feature_columns,
    'original_shape': df.shape,
    'feature_matrix_shape': X.shape
}

import json
with open(r'c:\Users\Millpark\Downloads\Unsupervised Learning Project11\preprocessing_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Saved: preprocessing_metadata.json")

# ============================================================================
# 13. SUMMARY STATISTICS
# ============================================================================
print("\n[13] PREPROCESSING SUMMARY")
print("=" * 80)

print(f"""
✓ PREPROCESSING COMPLETED SUCCESSFULLY!

📊 DATASET SUMMARY:
   • Original rows: {df.shape[0]}
   • Original columns: {len(column_mapping)}
   • Final columns: {df.shape[1]}
   • Features for ML: {len(feature_columns)}

📍 SAMPLING POINTS:
   • Total unique points: {df['Sampling_Point'].nunique()}
   • Points: {', '.join(df['Sampling_Point'].unique())}

📅 TEMPORAL COVERAGE:
   • Date range: {df['Date'].min().strftime('%d/%m/%Y')} to {df['Date'].max().strftime('%d/%m/%Y')}
   • Months covered: {df['Month'].nunique()}
   • Seasons: {', '.join(df['Season'].unique())}

🎯 DATA QUALITY:
   • Missing values handled: ✓
   • Outliers detected: {sum([x['N_Outliers'] for x in outlier_summary])} (retained)
   • Features scaled: ✓

💾 OUTPUT FILES:
   1. river_water_preprocessed.csv - Full preprocessed dataset
   2. river_water_features.csv - Unscaled feature matrix
   3. river_water_features_scaled.csv - Scaled feature matrix (ready for ML)
   4. preprocessing_metadata.json - Preprocessing information

🚀 READY FOR UNSUPERVISED LEARNING:
   • Clustering (K-Means, DBSCAN, Hierarchical)
   • Dimensionality Reduction (PCA, t-SNE, UMAP)
   • Anomaly Detection (Isolation Forest, Local Outlier Factor)
""")

# ============================================================================
# 14. VISUALIZATION OF KEY RELATIONSHIPS
# ============================================================================
print("\n[14] GENERATING VISUALIZATIONS...")

# Create visualization directory
import os
viz_dir = r'c:\Users\Millpark\Downloads\Unsupervised Learning Project11\visualizations'
os.makedirs(viz_dir, exist_ok=True)

# 1. Correlation heatmap
plt.figure(figsize=(14, 12))
correlation_features = ['pH', 'EC', 'TDS', 'TSS', 'DO', 'Turbidity', 'Hardness', 
                        'Total_Chlorine', 'Ambient_Temp', 'Sample_Temp']
corr_matrix = df[correlation_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Water Quality Parameters', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{viz_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_matrix.png")
plt.close()

# 2. Distribution of key parameters by sampling point
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
params = ['pH', 'DO', 'Turbidity', 'EC', 'Total_Chlorine', 'Pollution_Score']
for idx, param in enumerate(params):
    ax = axes[idx // 3, idx % 3]
    df.boxplot(column=param, by='Sampling_Point', ax=ax)
    ax.set_title(f'{param} by Sampling Point')
    ax.set_xlabel('Sampling Point')
    ax.set_ylabel(param)
plt.suptitle('Water Quality Parameters Distribution by Sampling Point', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{viz_dir}/parameters_by_location.png', dpi=300, bbox_inches='tight')
print("✓ Saved: parameters_by_location.png")
plt.close()

# 3. Temporal trends
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
temporal_params = ['pH', 'DO', 'Turbidity', 'Pollution_Score']
for idx, param in enumerate(temporal_params):
    ax = axes[idx // 2, idx % 2]
    for point in df['Sampling_Point'].unique():
        point_data = df[df['Sampling_Point'] == point]
        ax.plot(point_data['Date'], point_data[param], label=point, alpha=0.7, marker='o', markersize=2)
    ax.set_title(f'{param} Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel(param)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle('Temporal Trends in Water Quality Parameters', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{viz_dir}/temporal_trends.png', dpi=300, bbox_inches='tight')
print("✓ Saved: temporal_trends.png")
plt.close()

print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE! You can now proceed with unsupervised learning.")
print("=" * 80)
