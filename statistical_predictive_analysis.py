"""
Statistical and Predictive Analysis
River Water Quality Dataset
Comprehensive statistical testing, correlation analysis, and predictive modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson, kstest
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import ttest_ind, mannwhitneyu, kruskal, f_oneway
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("STATISTICAL AND PREDICTIVE ANALYSIS")
print("River Water Quality Dataset")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n[1] LOADING DATA...")

# Load datasets
df_original = pd.read_csv('river_water_dates_cleaned.csv')
df_preprocessed = pd.read_csv('river_water_preprocessed.csv')
df_engineered = pd.read_csv('river_water_features_engineered.csv')

print(f"✓ Original data: {df_original.shape}")
print(f"✓ Preprocessed data: {df_preprocessed.shape}")
print(f"✓ Engineered features: {df_engineered.shape}")

# ============================================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("[2] DESCRIPTIVE STATISTICS")
print("=" * 80)

# Select numeric columns for analysis
numeric_cols = df_preprocessed.select_dtypes(include=[np.number]).columns.tolist()

# Calculate comprehensive statistics
desc_stats = pd.DataFrame({
    'Mean': df_preprocessed[numeric_cols].mean(),
    'Median': df_preprocessed[numeric_cols].median(),
    'Std': df_preprocessed[numeric_cols].std(),
    'Min': df_preprocessed[numeric_cols].min(),
    'Max': df_preprocessed[numeric_cols].max(),
    'Q1': df_preprocessed[numeric_cols].quantile(0.25),
    'Q3': df_preprocessed[numeric_cols].quantile(0.75),
    'IQR': df_preprocessed[numeric_cols].quantile(0.75) - df_preprocessed[numeric_cols].quantile(0.25),
    'Skewness': df_preprocessed[numeric_cols].skew(),
    'Kurtosis': df_preprocessed[numeric_cols].kurtosis(),
    'CV': (df_preprocessed[numeric_cols].std() / df_preprocessed[numeric_cols].mean()) * 100
})

print("\nDescriptive Statistics Summary:")
print(desc_stats.round(3))

# Save descriptive statistics
desc_stats.to_csv('statistical_analysis/descriptive_statistics.csv')
print("\n✓ Saved: descriptive_statistics.csv")

# ============================================================================
# 3. NORMALITY TESTS
# ============================================================================

print("\n" + "=" * 80)
print("[3] NORMALITY TESTS")
print("=" * 80)

normality_results = []

for col in numeric_cols:
    data = df_preprocessed[col].dropna()
    
    if len(data) > 3:  # Need at least 3 samples
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = shapiro(data)
        
        # Anderson-Darling test
        anderson_result = anderson(data)
        
        # D'Agostino-Pearson test
        if len(data) >= 8:
            dagostino_stat, dagostino_p = normaltest(data)
        else:
            dagostino_stat, dagostino_p = np.nan, np.nan
        
        normality_results.append({
            'Variable': col,
            'Shapiro_Stat': shapiro_stat,
            'Shapiro_p': shapiro_p,
            'Shapiro_Normal': 'Yes' if shapiro_p > 0.05 else 'No',
            'Anderson_Stat': anderson_result.statistic,
            'DAgostino_Stat': dagostino_stat,
            'DAgostino_p': dagostino_p,
            'DAgostino_Normal': 'Yes' if dagostino_p > 0.05 else 'No' if not np.isnan(dagostino_p) else 'N/A'
        })

normality_df = pd.DataFrame(normality_results)
print("\nNormality Test Results:")
print(normality_df.to_string(index=False))

# Save normality results
normality_df.to_csv('statistical_analysis/normality_tests.csv', index=False)
print("\n✓ Saved: normality_tests.csv")

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("[4] CORRELATION ANALYSIS")
print("=" * 80)

# Pearson correlation
pearson_corr = df_preprocessed[numeric_cols].corr(method='pearson')

# Spearman correlation
spearman_corr = df_preprocessed[numeric_cols].corr(method='spearman')

# Find strong correlations (|r| > 0.7)
print("\nStrong Correlations (|r| > 0.7):")
strong_corr = []
for i in range(len(pearson_corr.columns)):
    for j in range(i+1, len(pearson_corr.columns)):
        if abs(pearson_corr.iloc[i, j]) > 0.7:
            strong_corr.append({
                'Variable_1': pearson_corr.columns[i],
                'Variable_2': pearson_corr.columns[j],
                'Pearson_r': pearson_corr.iloc[i, j],
                'Spearman_rho': spearman_corr.iloc[i, j]
            })

if strong_corr:
    strong_corr_df = pd.DataFrame(strong_corr)
    print(strong_corr_df.to_string(index=False))
else:
    print("No strong correlations found.")

# Create correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Pearson correlation heatmap
sns.heatmap(pearson_corr, annot=False, cmap='coolwarm', center=0, 
            vmin=-1, vmax=1, ax=axes[0], cbar_kws={'label': 'Correlation'})
axes[0].set_title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')

# Spearman correlation heatmap
sns.heatmap(spearman_corr, annot=False, cmap='coolwarm', center=0,
            vmin=-1, vmax=1, ax=axes[1], cbar_kws={'label': 'Correlation'})
axes[1].set_title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('statistical_analysis/correlation_heatmaps.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: correlation_heatmaps.png")
plt.close()

# Save correlation matrices
pearson_corr.to_csv('statistical_analysis/pearson_correlation.csv')
spearman_corr.to_csv('statistical_analysis/spearman_correlation.csv')

# ============================================================================
# 5. STATISTICAL TESTS - GROUP COMPARISONS
# ============================================================================

print("\n" + "=" * 80)
print("[5] GROUP COMPARISON TESTS")
print("=" * 80)

# Test if there are significant differences between sampling points
if 'Sampling_Point' in df_preprocessed.columns:
    print("\nTesting differences between Sampling Points:")
    
    group_comparison_results = []
    
    for col in numeric_cols[:10]:  # Test first 10 numeric variables
        groups = [df_preprocessed[df_preprocessed['Sampling_Point'] == point][col].dropna() 
                  for point in df_preprocessed['Sampling_Point'].unique()]
        
        # Filter out empty groups
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            # Kruskal-Wallis test (non-parametric)
            h_stat, kruskal_p = kruskal(*groups)
            
            # ANOVA (parametric)
            f_stat, anova_p = f_oneway(*groups)
            
            group_comparison_results.append({
                'Variable': col,
                'Kruskal_H': h_stat,
                'Kruskal_p': kruskal_p,
                'Significant_KW': 'Yes' if kruskal_p < 0.05 else 'No',
                'ANOVA_F': f_stat,
                'ANOVA_p': anova_p,
                'Significant_ANOVA': 'Yes' if anova_p < 0.05 else 'No'
            })
    
    if group_comparison_results:
        group_comp_df = pd.DataFrame(group_comparison_results)
        print("\nGroup Comparison Results:")
        print(group_comp_df.to_string(index=False))
        
        group_comp_df.to_csv('statistical_analysis/group_comparison_tests.csv', index=False)
        print("\n✓ Saved: group_comparison_tests.csv")

# ============================================================================
# 6. PREDICTIVE MODELING - REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("[6] PREDICTIVE MODELING - REGRESSION")
print("=" * 80)
print("\nTarget Variable: DO (Dissolved Oxygen)")

# Prepare data for modeling
# Select features (exclude target and non-numeric columns)
feature_cols = [col for col in numeric_cols if col not in ['DO', 'Year', 'Month', 'Day']]

# Remove rows with missing DO values
modeling_data = df_preprocessed[feature_cols + ['DO']].dropna()

X = modeling_data[feature_cols]
y = modeling_data['DO']

print(f"\nDataset size: {len(X)} samples, {len(feature_cols)} features")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define regression models
regression_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'Support Vector Regression': SVR(kernel='rbf', C=1.0)
}

regression_results = []

print("\nTraining Regression Models...")
for name, model in regression_models.items():
    # Train model
    if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net', 'Support Vector Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                     scoring='neg_mean_squared_error')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                     scoring='neg_mean_squared_error')
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    regression_results.append({
        'Model': name,
        'R2_Score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'CV_RMSE': cv_rmse
    })
    
    print(f"✓ {name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")

# Create results dataframe
regression_df = pd.DataFrame(regression_results).sort_values('R2_Score', ascending=False)
print("\n" + "=" * 60)
print("REGRESSION MODEL COMPARISON")
print("=" * 60)
print(regression_df.to_string(index=False))

regression_df.to_csv('statistical_analysis/regression_results.csv', index=False)
print("\n✓ Saved: regression_results.csv")

# Visualize model comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# R² Score comparison
axes[0, 0].barh(regression_df['Model'], regression_df['R2_Score'], color='steelblue')
axes[0, 0].set_xlabel('R² Score')
axes[0, 0].set_title('Model Performance - R² Score', fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)

# RMSE comparison
axes[0, 1].barh(regression_df['Model'], regression_df['RMSE'], color='coral')
axes[0, 1].set_xlabel('RMSE')
axes[0, 1].set_title('Model Performance - RMSE (Lower is Better)', fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# MAE comparison
axes[1, 0].barh(regression_df['Model'], regression_df['MAE'], color='lightgreen')
axes[1, 0].set_xlabel('MAE')
axes[1, 0].set_title('Model Performance - MAE (Lower is Better)', fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# Cross-validation RMSE
axes[1, 1].barh(regression_df['Model'], regression_df['CV_RMSE'], color='mediumpurple')
axes[1, 1].set_xlabel('Cross-Validation RMSE')
axes[1, 1].set_title('Model Performance - CV RMSE (Lower is Better)', fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('statistical_analysis/regression_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: regression_model_comparison.png")
plt.close()

# ============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("[7] FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Train Random Forest for feature importance
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Save feature importance
feature_importance.to_csv('statistical_analysis/feature_importance.csv', index=False)
print("\n✓ Saved: feature_importance.csv")

# Visualize top 20 features
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['Importance'], color='teal')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance Score')
plt.title('Top 20 Most Important Features for DO Prediction', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('statistical_analysis/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
plt.close()

# ============================================================================
# 8. PREDICTIVE MODELING - CLASSIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("[8] PREDICTIVE MODELING - CLASSIFICATION")
print("=" * 80)
print("\nTarget: Water Quality Category (based on DO levels)")

# Create water quality categories based on DO levels
def categorize_water_quality(do_value):
    if do_value < 2.0:
        return 'Severely Hypoxic'
    elif do_value < 4.0:
        return 'Hypoxic'
    elif do_value < 6.0:
        return 'Low'
    else:
        return 'Adequate'

modeling_data['WQ_Category'] = modeling_data['DO'].apply(categorize_water_quality)

# Prepare classification data
X_class = modeling_data[feature_cols]
y_class = modeling_data['WQ_Category']

# Encode labels
le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)

print(f"\nClass Distribution:")
print(pd.Series(y_class).value_counts())

# Split data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class_encoded, test_size=0.2, random_state=42, stratify=y_class_encoded
)

# Scale features
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

# Define classification models
classification_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
}

classification_results = []

print("\nTraining Classification Models...")
for name, model in classification_models.items():
    # Train model
    if name == 'Logistic Regression':
        model.fit(X_train_c_scaled, y_train_c)
        y_pred_c = model.predict(X_test_c_scaled)
        cv_scores = cross_val_score(model, X_train_c_scaled, y_train_c, cv=5)
    else:
        model.fit(X_train_c, y_train_c)
        y_pred_c = model.predict(X_test_c)
        cv_scores = cross_val_score(model, X_train_c, y_train_c, cv=5)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_c, y_pred_c)
    cv_accuracy = cv_scores.mean()
    
    classification_results.append({
        'Model': name,
        'Accuracy': accuracy,
        'CV_Accuracy': cv_accuracy
    })
    
    print(f"✓ {name}: Accuracy = {accuracy:.4f}, CV Accuracy = {cv_accuracy:.4f}")

# Create results dataframe
classification_df = pd.DataFrame(classification_results).sort_values('Accuracy', ascending=False)
print("\n" + "=" * 60)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 60)
print(classification_df.to_string(index=False))

classification_df.to_csv('statistical_analysis/classification_results.csv', index=False)
print("\n✓ Saved: classification_results.csv")

# ============================================================================
# 9. STATISTICAL SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("[9] GENERATING SUMMARY REPORT")
print("=" * 80)

summary_report = f"""
{'=' * 80}
STATISTICAL AND PREDICTIVE ANALYSIS - SUMMARY REPORT
{'=' * 80}

Dataset: River Water Quality Parameters
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Samples: {len(df_preprocessed)}
Total Features: {len(numeric_cols)}

{'=' * 80}
1. DESCRIPTIVE STATISTICS
{'=' * 80}

Key Water Quality Parameters:
- DO (Dissolved Oxygen): {df_preprocessed['DO'].mean():.2f} ± {df_preprocessed['DO'].std():.2f} mg/L
- pH: {df_preprocessed['pH'].mean():.2f} ± {df_preprocessed['pH'].std():.2f}
- Turbidity: {df_preprocessed['Turbidity'].mean():.2f} ± {df_preprocessed['Turbidity'].std():.2f} NTU
- Temperature: {df_preprocessed['Sample_Temp'].mean():.2f} ± {df_preprocessed['Sample_Temp'].std():.2f} °C

Variability Analysis (Coefficient of Variation):
{desc_stats['CV'].sort_values(ascending=False).head(5).to_string()}

{'=' * 80}
2. NORMALITY ASSESSMENT
{'=' * 80}

Variables following normal distribution (Shapiro-Wilk test, p > 0.05):
{len(normality_df[normality_df['Shapiro_Normal'] == 'Yes'])} out of {len(normality_df)} variables

Variables requiring non-parametric methods:
{len(normality_df[normality_df['Shapiro_Normal'] == 'No'])} variables

{'=' * 80}
3. CORRELATION ANALYSIS
{'=' * 80}

Strong correlations detected: {len(strong_corr) if strong_corr else 0}

Top correlations saved in: correlation_heatmaps.png

{'=' * 80}
4. PREDICTIVE MODELING RESULTS
{'=' * 80}

REGRESSION (Predicting DO levels):
Best Model: {regression_df.iloc[0]['Model']}
  - R² Score: {regression_df.iloc[0]['R2_Score']:.4f}
  - RMSE: {regression_df.iloc[0]['RMSE']:.4f}
  - MAE: {regression_df.iloc[0]['MAE']:.4f}

CLASSIFICATION (Water Quality Categories):
Best Model: {classification_df.iloc[0]['Model']}
  - Accuracy: {classification_df.iloc[0]['Accuracy']:.4f}
  - CV Accuracy: {classification_df.iloc[0]['CV_Accuracy']:.4f}

{'=' * 80}
5. FEATURE IMPORTANCE
{'=' * 80}

Top 5 Most Important Features for DO Prediction:
{feature_importance.head(5).to_string(index=False)}

{'=' * 80}
FILES GENERATED
{'=' * 80}

Statistical Analysis:
✓ descriptive_statistics.csv
✓ normality_tests.csv
✓ pearson_correlation.csv
✓ spearman_correlation.csv
✓ correlation_heatmaps.png
✓ group_comparison_tests.csv

Predictive Modeling:
✓ regression_results.csv
✓ regression_model_comparison.png
✓ classification_results.csv
✓ feature_importance.csv
✓ feature_importance.png

{'=' * 80}
RECOMMENDATIONS
{'=' * 80}

1. Model Selection:
   - For DO prediction: Use {regression_df.iloc[0]['Model']} (R² = {regression_df.iloc[0]['R2_Score']:.4f})
   - For quality classification: Use {classification_df.iloc[0]['Model']} (Acc = {classification_df.iloc[0]['Accuracy']:.4f})

2. Key Predictors:
   - Focus on top 5 features: {', '.join(feature_importance.head(5)['Feature'].tolist())}

3. Statistical Considerations:
   - {len(normality_df[normality_df['Shapiro_Normal'] == 'No'])} variables are non-normal
   - Use non-parametric methods for these variables

4. Future Work:
   - Collect more samples to improve model accuracy
   - Consider temporal patterns (seasonal effects)
   - Explore interaction effects between parameters

{'=' * 80}
END OF REPORT
{'=' * 80}
"""

print(summary_report)

# Save summary report
with open('statistical_analysis/statistical_summary_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print("\n✓ Saved: statistical_summary_report.txt")

print("\n" + "=" * 80)
print("STATISTICAL AND PREDICTIVE ANALYSIS COMPLETED!")
print("=" * 80)
print("\nAll results saved to: statistical_analysis/")
