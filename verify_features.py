import pandas as pd
import json

# Load and display feature engineering results
df = pd.read_csv('river_water_features_engineered.csv')
print("=" * 80)
print("FEATURE ENGINEERING VERIFICATION")
print("=" * 80)

print(f"\nFull Dataset Shape: {df.shape}")
print(f"  Rows: {df.shape[0]}")
print(f"  Total Columns: {df.shape[1]}")

print(f"\nFirst 10 columns:")
for i, col in enumerate(df.columns[:10], 1):
    print(f"  {i:2d}. {col}")

print(f"\nLast 10 columns:")
for i, col in enumerate(df.columns[-10:], 1):
    print(f"  {i:2d}. {col}")

# ML Features
ml = pd.read_csv('ml_features_unscaled.csv')
print(f"\n\nML Features (Unscaled):")
print(f"  Shape: {ml.shape}")
print(f"  Features: {ml.shape[1]}")

# Load metadata
with open('feature_engineering_metadata.json', 'r') as f:
    meta = json.load(f)

print(f"\n\nFeature Categories:")
for cat, feats in meta['categories'].items():
    print(f"  - {cat}: {len(feats)} features")
    
print(f"\n\nML Features by Category:")
for cat, feats in meta['categories'].items():
    print(f"\n  {cat.upper()}:")
    for feat in feats:
        print(f"    - {feat}")

# Statistics
stats = pd.read_csv('feature_summary_statistics.csv')
print(f"\n\nFeature Statistics Summary:")
print(f"  Total features analyzed: {len(stats)}")
print(f"\n  Top 5 features by mean:")
top_mean = stats.nlargest(5, 'Mean')[['Feature', 'Mean']]
for idx, row in top_mean.iterrows():
    print(f"    {row['Feature']}: {row['Mean']:.2f}")
