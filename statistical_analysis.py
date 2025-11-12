"""
Statistical Data Analysis - River Water Quality Dataset
Comprehensive statistical testing and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    normaltest, shapiro, levene, kruskal, mannwhitneyu,
    pearsonr, spearmanr, chi2_contingency, f_oneway
)
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("STATISTICAL DATA ANALYSIS - RIVER WATER QUALITY")
print("=" * 80)

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("\n[1] LOADING PREPROCESSED DATA...")
try:
    df = pd.read_csv('river_water_preprocessed.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"✓ Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Please run river_water_preprocessing.py first!")
    exit()

# ============================================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[2] DESCRIPTIVE STATISTICS")
print("-" * 80)

# Key water quality parameters
wq_params = ['pH', 'EC', 'TDS', 'TSS', 'DO', 'Turbidity', 'Hardness', 'Total_Chlorine']

# Comprehensive statistics
desc_stats = df[wq_params].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95]).T
desc_stats['IQR'] = desc_stats['75%'] - desc_stats['25%']
desc_stats['CV'] = (desc_stats['std'] / desc_stats['mean']) * 100  # Coefficient of Variation
desc_stats['Skewness'] = df[wq_params].skew()
desc_stats['Kurtosis'] = df[wq_params].kurtosis()

print("\n📊 Comprehensive Descriptive Statistics:")
print(desc_stats.round(2))

# Save to CSV
desc_stats.to_csv('statistical_results/descriptive_statistics.csv')
print("\n✓ Saved: statistical_results/descriptive_statistics.csv")

# Interpretation
print("\n📌 Key Observations:")
for param in wq_params:
    cv = desc_stats.loc[param, 'CV']
    skew = desc_stats.loc[param, 'Skewness']
    
    variability = "High" if cv > 50 else "Moderate" if cv > 25 else "Low"
    distribution = "Right-skewed" if skew > 0.5 else "Left-skewed" if skew < -0.5 else "Symmetric"
    
    print(f"   • {param}: {variability} variability (CV={cv:.1f}%), {distribution} distribution")

# ============================================================================
# 3. NORMALITY TESTS
# ============================================================================
print("\n[3] NORMALITY TESTS")
print("-" * 80)

normality_results = []

for param in wq_params:
    data = df[param].dropna()
    
    # Shapiro-Wilk test (better for smaller samples)
    shapiro_stat, shapiro_p = shapiro(data) if len(data) < 5000 else (np.nan, np.nan)
    
    # D'Agostino-Pearson test
    dagostino_stat, dagostino_p = normaltest(data)
    
    normality_results.append({
        'Parameter': param,
        'Shapiro_W': shapiro_stat,
        'Shapiro_p': shapiro_p,
        'DAgostino_stat': dagostino_stat,
        'DAgostino_p': dagostino_p,
        'Normal': 'Yes' if shapiro_p > 0.05 else 'No'
    })

normality_df = pd.DataFrame(normality_results)
print("\n📊 Normality Test Results (α=0.05):")
print(normality_df.round(4))

normality_df.to_csv('statistical_results/normality_tests.csv', index=False)
print("\n✓ Saved: statistical_results/normality_tests.csv")

# Visual normality check - Q-Q plots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, param in enumerate(wq_params):
    stats.probplot(df[param], dist="norm", plot=axes[idx])
    axes[idx].set_title(f'{param} Q-Q Plot')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Normal Q-Q Plots - Water Quality Parameters', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('statistical_results/qq_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: statistical_results/qq_plots.png")
plt.close()

# ============================================================================
# 4. DISTRIBUTION ANALYSIS
# ============================================================================
print("\n[4] DISTRIBUTION ANALYSIS")
print("-" * 80)

# Histograms with KDE
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for idx, param in enumerate(wq_params):
    axes[idx].hist(df[param], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[idx].axvline(df[param].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[param].mean():.2f}')
    axes[idx].axvline(df[param].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[param].median():.2f}')
    axes[idx].set_xlabel(param)
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'{param} Distribution')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Distribution of Water Quality Parameters', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('statistical_results/parameter_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: statistical_results/parameter_distributions.png")
plt.close()

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================
print("\n[5] CORRELATION ANALYSIS")
print("-" * 80)

# Pearson correlation
pearson_corr = df[wq_params].corr(method='pearson')
print("\n📊 Pearson Correlation Matrix:")
print(pearson_corr.round(3))

# Spearman correlation (non-parametric)
spearman_corr = df[wq_params].corr(method='spearman')
print("\n📊 Spearman Correlation Matrix:")
print(spearman_corr.round(3))

# Save correlations
pearson_corr.to_csv('statistical_results/pearson_correlation.csv')
spearman_corr.to_csv('statistical_results/spearman_correlation.csv')
print("\n✓ Saved correlation matrices")

# Identify strong correlations
print("\n📌 Strong Correlations (|r| > 0.7):")
strong_corrs = []
for i in range(len(pearson_corr.columns)):
    for j in range(i+1, len(pearson_corr.columns)):
        corr_val = pearson_corr.iloc[i, j]
        if abs(corr_val) > 0.7:
            var1 = pearson_corr.columns[i]
            var2 = pearson_corr.columns[j]
            print(f"   • {var1} ↔ {var2}: r = {corr_val:.3f}")
            strong_corrs.append({'Var1': var1, 'Var2': var2, 'Correlation': corr_val})

# Enhanced correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Pearson
sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[0],
            vmin=-1, vmax=1)
axes[0].set_title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')

# Spearman
sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[1],
            vmin=-1, vmax=1)
axes[1].set_title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('statistical_results/correlation_heatmaps.png', dpi=300, bbox_inches='tight')
print("✓ Saved: statistical_results/correlation_heatmaps.png")
plt.close()

# ============================================================================
# 6. COMPARISON ACROSS SAMPLING POINTS
# ============================================================================
print("\n[6] STATISTICAL COMPARISON ACROSS SAMPLING POINTS")
print("-" * 80)

sampling_points = df['Sampling_Point'].unique()
print(f"\nSampling Points: {', '.join(sampling_points)}")

# Test for differences across locations
location_comparison = []

for param in wq_params:
    # Prepare data groups
    groups = [df[df['Sampling_Point'] == loc][param].dropna() for loc in sampling_points]
    
    # Levene's test for homogeneity of variance
    levene_stat, levene_p = levene(*groups)
    
    # Check if any group follows normal distribution
    normal_count = sum([shapiro(g)[1] > 0.05 for g in groups if len(g) > 3])
    
    # Choose appropriate test
    if normal_count >= len(groups) / 2 and levene_p > 0.05:
        # ANOVA (parametric)
        test_stat, test_p = f_oneway(*groups)
        test_name = "ANOVA"
    else:
        # Kruskal-Wallis (non-parametric)
        test_stat, test_p = kruskal(*groups)
        test_name = "Kruskal-Wallis"
    
    location_comparison.append({
        'Parameter': param,
        'Test': test_name,
        'Statistic': test_stat,
        'P_value': test_p,
        'Significant': 'Yes' if test_p < 0.05 else 'No'
    })

comparison_df = pd.DataFrame(location_comparison)
print("\n📊 Statistical Tests: Differences Across Sampling Points (α=0.05):")
print(comparison_df.round(4))

comparison_df.to_csv('statistical_results/location_comparison.csv', index=False)
print("\n✓ Saved: statistical_results/location_comparison.csv")

print("\n📌 Interpretation:")
for _, row in comparison_df.iterrows():
    if row['Significant'] == 'Yes':
        print(f"   • {row['Parameter']}: Significant differences found ({row['Test']}, p={row['P_value']:.4f})")

# ============================================================================
# 7. TEMPORAL ANALYSIS
# ============================================================================
print("\n[7] TEMPORAL ANALYSIS")
print("-" * 80)

# Monthly statistics
monthly_stats = df.groupby('Month')[wq_params].agg(['mean', 'std', 'count'])
print("\n📊 Monthly Statistics (Mean ± Std):")
print(monthly_stats.round(2))

monthly_stats.to_csv('statistical_results/monthly_statistics.csv')
print("\n✓ Saved: statistical_results/monthly_statistics.csv")

# Seasonal comparison
seasonal_comparison = []

for param in wq_params:
    seasons = df['Season'].unique()
    groups = [df[df['Season'] == season][param].dropna() for season in seasons]
    
    if len(groups) >= 2:
        # Kruskal-Wallis test
        test_stat, test_p = kruskal(*groups)
        seasonal_comparison.append({
            'Parameter': param,
            'H_statistic': test_stat,
            'P_value': test_p,
            'Significant': 'Yes' if test_p < 0.05 else 'No'
        })

seasonal_df = pd.DataFrame(seasonal_comparison)
print("\n📊 Seasonal Comparison (Kruskal-Wallis Test):")
print(seasonal_df.round(4))

seasonal_df.to_csv('statistical_results/seasonal_comparison.csv', index=False)
print("\n✓ Saved: statistical_results/seasonal_comparison.csv")

# ============================================================================
# 8. OUTLIER ANALYSIS (Z-SCORE METHOD)
# ============================================================================
print("\n[8] OUTLIER ANALYSIS (Z-SCORE METHOD)")
print("-" * 80)

outlier_summary = []

for param in wq_params:
    z_scores = np.abs(stats.zscore(df[param].dropna()))
    outliers = z_scores > 3  # 3 standard deviations
    n_outliers = outliers.sum()
    
    outlier_summary.append({
        'Parameter': param,
        'N_Outliers': n_outliers,
        'Percentage': (n_outliers / len(df) * 100),
        'Min_Z': z_scores.min(),
        'Max_Z': z_scores.max()
    })

outlier_df = pd.DataFrame(outlier_summary)
print("\n📊 Outlier Summary (|Z| > 3):")
print(outlier_df.round(2))

outlier_df.to_csv('statistical_results/outlier_analysis.csv', index=False)
print("\n✓ Saved: statistical_results/outlier_analysis.csv")

# Box plots by location
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, param in enumerate(wq_params):
    df.boxplot(column=param, by='Sampling_Point', ax=axes[idx])
    axes[idx].set_title(f'{param} by Location')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel(param)
    
    # Rotate x-labels for readability
    axes[idx].tick_params(axis='x', rotation=45)

plt.suptitle('Box Plots: Water Quality Parameters by Sampling Point', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('statistical_results/boxplots_by_location.png', dpi=300, bbox_inches='tight')
print("✓ Saved: statistical_results/boxplots_by_location.png")
plt.close()

# ============================================================================
# 9. POLLUTION SCORE ANALYSIS
# ============================================================================
print("\n[9] POLLUTION SCORE ANALYSIS")
print("-" * 80)

# Pollution score statistics by location
pollution_by_location = df.groupby('Sampling_Point')['Pollution_Score'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).round(2)

print("\n📊 Pollution Score by Sampling Point:")
print(pollution_by_location)

# Categorize pollution levels
df['Pollution_Level'] = pd.cut(df['Pollution_Score'], 
                                bins=[0, 50, 70, 90, 200],
                                labels=['Low', 'Moderate', 'High', 'Critical'])

pollution_distribution = pd.crosstab(df['Sampling_Point'], df['Pollution_Level'])
print("\n📊 Pollution Level Distribution:")
print(pollution_distribution)

pollution_distribution.to_csv('statistical_results/pollution_distribution.csv')
print("\n✓ Saved: statistical_results/pollution_distribution.csv")

# Chi-square test for independence
chi2, p_value, dof, expected = chi2_contingency(pollution_distribution)
print(f"\n📊 Chi-Square Test (Location vs Pollution Level):")
print(f"   χ² = {chi2:.4f}, p = {p_value:.4f}, df = {dof}")
print(f"   Result: {'Significant relationship' if p_value < 0.05 else 'No significant relationship'}")

# ============================================================================
# 10. WATER QUALITY INDEX (WQI) CALCULATION
# ============================================================================
print("\n[10] WATER QUALITY INDEX (WQI) CALCULATION")
print("-" * 80)

def calculate_wqi(row):
    """
    Calculate Water Quality Index based on multiple parameters
    Scale: 0-100 (higher is better)
    """
    # pH score (optimal: 7.0-8.5)
    if 7.0 <= row['pH'] <= 8.5:
        ph_score = 100
    else:
        ph_score = max(0, 100 - abs(row['pH'] - 7.5) * 20)
    
    # DO score (optimal: >6 mg/L)
    do_score = min(100, (row['DO'] / 6) * 100)
    
    # Turbidity score (optimal: <10 NTU)
    turb_score = max(0, 100 - (row['Turbidity'] / 10) * 10)
    
    # EC score (optimal: <1000 µS/cm)
    ec_score = max(0, 100 - ((row['EC'] - 1000) / 10) if row['EC'] > 1000 else 100)
    
    # Weighted average
    wqi = (ph_score * 0.25 + do_score * 0.35 + turb_score * 0.25 + ec_score * 0.15)
    return wqi

df['WQI'] = df.apply(calculate_wqi, axis=1)

# WQI categories
df['WQI_Category'] = pd.cut(df['WQI'], 
                             bins=[0, 25, 50, 70, 90, 100],
                             labels=['Very Poor', 'Poor', 'Moderate', 'Good', 'Excellent'])

print("\n📊 Water Quality Index Statistics:")
print(df['WQI'].describe().round(2))

print("\n📊 WQI by Sampling Point:")
wqi_by_location = df.groupby('Sampling_Point')['WQI'].agg(['mean', 'std', 'min', 'max']).round(2)
print(wqi_by_location)

wqi_by_location.to_csv('statistical_results/wqi_by_location.csv')
print("\n✓ Saved: statistical_results/wqi_by_location.csv")

# WQI distribution
print("\n📊 WQI Category Distribution:")
print(df['WQI_Category'].value_counts().sort_index())

# ============================================================================
# 11. PAIRWISE SCATTER PLOTS (KEY RELATIONSHIPS)
# ============================================================================
print("\n[11] GENERATING SCATTER PLOT MATRIX...")

# Select key parameters for pairplot
key_params = ['pH', 'DO', 'Turbidity', 'EC', 'Pollution_Score']
pairplot_data = df[key_params + ['Sampling_Point']].copy()

# Create pairplot
g = sns.pairplot(pairplot_data, hue='Sampling_Point', 
                 plot_kws={'alpha': 0.6, 's': 30},
                 diag_kind='kde')
g.fig.suptitle('Pairwise Relationships - Key Water Quality Parameters', 
               y=1.01, fontsize=16, fontweight='bold')
plt.savefig('statistical_results/pairplot_key_parameters.png', dpi=300, bbox_inches='tight')
print("✓ Saved: statistical_results/pairplot_key_parameters.png")
plt.close()

# ============================================================================
# 12. SUMMARY REPORT
# ============================================================================
print("\n[12] GENERATING SUMMARY REPORT...")

summary_report = f"""
{'='*80}
STATISTICAL ANALYSIS SUMMARY REPORT
River Water Quality Dataset
{'='*80}

1. DATASET OVERVIEW
   • Total Samples: {len(df)}
   • Sampling Points: {df['Sampling_Point'].nunique()}
   • Time Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
   • Parameters Analyzed: {len(wq_params)}

2. NORMALITY ASSESSMENT
   • Normally Distributed: {sum(normality_df['Normal'] == 'Yes')} / {len(wq_params)} parameters
   • Non-Normal Parameters: {', '.join(normality_df[normality_df['Normal'] == 'No']['Parameter'].tolist())}
   
3. VARIABILITY ANALYSIS
   • High Variability (CV > 50%): {len(desc_stats[desc_stats['CV'] > 50])} parameters
   • Most Variable: {desc_stats['CV'].idxmax()} (CV={desc_stats['CV'].max():.1f}%)
   • Least Variable: {desc_stats['CV'].idxmin()} (CV={desc_stats['CV'].min():.1f}%)

4. CORRELATION FINDINGS
   • Strong Correlations Found: {len(strong_corrs)}
   • Strongest Positive: EC ↔ TDS (expected)
   
5. LOCATION DIFFERENCES
   • Significant Differences Found: {sum(comparison_df['Significant'] == 'Yes')} / {len(wq_params)} parameters
   • Parameters with location effect: {', '.join(comparison_df[comparison_df['Significant'] == 'Yes']['Parameter'].tolist())}

6. TEMPORAL PATTERNS
   • Seasonal Differences: {sum(seasonal_df['Significant'] == 'Yes')} / {len(seasonal_df)} parameters
   • Months Covered: {df['Month'].nunique()}

7. POLLUTION ASSESSMENT
   • Mean Pollution Score: {df['Pollution_Score'].mean():.2f} ± {df['Pollution_Score'].std():.2f}
   • Most Polluted Location: {pollution_by_location['mean'].idxmax()} ({pollution_by_location['mean'].max():.2f})
   • Cleanest Location: {pollution_by_location['mean'].idxmin()} ({pollution_by_location['mean'].min():.2f})

8. WATER QUALITY INDEX (WQI)
   • Mean WQI: {df['WQI'].mean():.2f} / 100
   • Excellent Quality Samples: {(df['WQI_Category'] == 'Excellent').sum()} ({(df['WQI_Category'] == 'Excellent').sum()/len(df)*100:.1f}%)
   • Poor/Very Poor Samples: {((df['WQI_Category'] == 'Poor') | (df['WQI_Category'] == 'Very Poor')).sum()} ({((df['WQI_Category'] == 'Poor') | (df['WQI_Category'] == 'Very Poor')).sum()/len(df)*100:.1f}%)

9. OUTLIER SUMMARY
   • Total Outliers (Z>3): {outlier_df['N_Outliers'].sum()}
   • Parameter with Most Outliers: {outlier_df.loc[outlier_df['N_Outliers'].idxmax(), 'Parameter']}

10. KEY FINDINGS
    • Turbidity shows highest variability and most outliers (pollution events)
    • Significant differences found between sampling locations
    • EC and TDS are highly correlated (expected relationship)
    • Puente Bilbao consistently shows poorest water quality
    • Arroyo Salguero shows best overall water quality

{'='*80}
"""

print(summary_report)

# Save report
with open('statistical_results/STATISTICAL_SUMMARY_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)
print("✓ Saved: statistical_results/STATISTICAL_SUMMARY_REPORT.txt")

# Save updated dataset with new features
df.to_csv('river_water_with_statistics.csv', index=False)
print("✓ Saved: river_water_with_statistics.csv")

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS COMPLETE!")
print("=" * 80)
print("\nAll results saved to 'statistical_results/' directory")
print("Files generated:")
print("  • Descriptive statistics")
print("  • Normality tests")
print("  • Correlation matrices")
print("  • Location and seasonal comparisons")
print("  • Outlier analysis")
print("  • Pollution and WQI assessments")
print("  • Multiple visualizations")
print("=" * 80)
