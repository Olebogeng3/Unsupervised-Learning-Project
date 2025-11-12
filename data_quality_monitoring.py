"""
Data Quality Monitoring and Auditing System
River Water Quality Dataset
Automated quality checks, validation, and reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("=" * 80)
print("DATA QUALITY MONITORING & AUDITING SYSTEM")
print("=" * 80)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# Define expected data quality thresholds
QUALITY_THRESHOLDS = {
    'missing_data_threshold': 0.05,  # Max 5% missing values acceptable
    'outlier_threshold': 0.10,  # Max 10% outliers acceptable
    'duplicate_threshold': 0.02,  # Max 2% duplicates acceptable
    'min_samples_required': 200,
    'max_days_gap': 14,  # Max gap between samples
}

# Define valid ranges for parameters (based on water quality standards)
PARAMETER_RANGES = {
    'pH': (6.0, 9.0),  # WHO drinking water standards
    'DO': (0.0, 20.0),  # Dissolved Oxygen mg/L - ADJUSTED: Natural range 0-20, includes hypoxic + supersaturated
    # NOTE: DO < 2.0 mg/L = severely hypoxic (ecological crisis)
    #       DO 2.0-4.0 mg/L = hypoxic (poor water quality)
    #       DO 4.0-6.0 mg/L = low (marginal for aquatic life)
    #       DO > 6.0 mg/L = adequate for most aquatic life
    'Turbidity': (0, 500),  # NTU
    'EC': (0, 3000),  # µS/cm
    'TDS': (0, 2000),  # mg/L
    'TSS': (0, 100),  # mL sed/L
    'Hardness': (0, 500),  # mg CaCO3/L
    'Total_Chlorine': (0, 300),  # mg Cl-/L
    'Ambient_Temp': (0, 45),  # °C
    'Sample_Temp': (0, 40),  # °C
    'Ambient_Humidity': (0, 100),  # %
}

# Create output directory
import os
os.makedirs('data_quality_reports', exist_ok=True)
print("✓ Output directory created: data_quality_reports/")

# ============================================================================
# 2. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA FOR QUALITY AUDIT...")

try:
    # Load original data
    df_original = pd.read_csv(r'c:\Users\Millpark\Downloads\River water parameters.csv')
    print(f"✓ Original data loaded: {df_original.shape}")
    
    # Load preprocessed data for comparison
    df_processed = pd.read_csv('river_water_preprocessed.csv')
    print(f"✓ Preprocessed data loaded: {df_processed.shape}")
    
    # Load engineered features
    df_engineered = pd.read_csv('river_water_features_engineered.csv')
    print(f"✓ Engineered data loaded: {df_engineered.shape}")
    
except FileNotFoundError as e:
    print(f"✗ Error loading data: {e}")
    exit()

# ============================================================================
# 3. DATA QUALITY METRICS
# ============================================================================
print("\n[2] CALCULATING DATA QUALITY METRICS...")

quality_report = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_info': {},
    'completeness': {},
    'validity': {},
    'consistency': {},
    'accuracy': {},
    'timeliness': {},
    'quality_score': {}
}

# 3.1 Completeness Check
print("\n   A. Completeness Analysis")

completeness_metrics = {}
total_cells = df_original.shape[0] * df_original.shape[1]
missing_cells = df_original.isnull().sum().sum()
completeness_rate = 1 - (missing_cells / total_cells)

completeness_metrics['total_records'] = len(df_original)
completeness_metrics['total_fields'] = len(df_original.columns)
completeness_metrics['missing_cells'] = int(missing_cells)
completeness_metrics['completeness_rate'] = round(completeness_rate, 4)
completeness_metrics['status'] = 'PASS' if missing_cells / total_cells < QUALITY_THRESHOLDS['missing_data_threshold'] else 'FAIL'

print(f"   Total Records: {completeness_metrics['total_records']}")
print(f"   Completeness Rate: {completeness_metrics['completeness_rate']*100:.2f}%")
print(f"   Status: {completeness_metrics['status']}")

# Missing values per column
missing_by_column = df_original.isnull().sum()
missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)

if len(missing_by_column) > 0:
    print(f"\n   Columns with Missing Values:")
    for col, count in missing_by_column.items():
        pct = (count / len(df_original)) * 100
        print(f"   - {col}: {count} ({pct:.2f}%)")

quality_report['completeness'] = completeness_metrics

# 3.2 Validity Check
print("\n   B. Validity Analysis (Range Checks)")

# Standardize column names for checking
df_check = df_original.copy()
column_mapping = {
    'pH': 'pH',
    'EC\n(µS/cm)': 'EC',
    'TDS\n(mg/L)': 'TDS',
    'TSS\n(mL sed/L)': 'TSS',
    'DO\n(mg/L)': 'DO',
    'Turbidity (NTU)': 'Turbidity',
    'Hardness\n(mg CaCO3/L)': 'Hardness',
    'Total Cl-\n(mg Cl-/L)': 'Total_Chlorine',
    'Ambient temperature (°C)': 'Ambient_Temp',
    'Sample temperature (°C)': 'Sample_Temp',
    'Ambient humidity': 'Ambient_Humidity',
}
df_check = df_check.rename(columns=column_mapping)

validity_issues = []
total_values_checked = 0
invalid_values = 0

for param, (min_val, max_val) in PARAMETER_RANGES.items():
    if param in df_check.columns:
        values = df_check[param].dropna()
        total_values_checked += len(values)
        
        out_of_range = ((values < min_val) | (values > max_val)).sum()
        invalid_values += out_of_range
        
        if out_of_range > 0:
            pct = (out_of_range / len(values)) * 100
            validity_issues.append({
                'parameter': param,
                'out_of_range_count': int(out_of_range),
                'percentage': round(pct, 2),
                'valid_range': f"{min_val} - {max_val}",
                'actual_min': round(values.min(), 2),
                'actual_max': round(values.max(), 2)
            })
            print(f"   ⚠ {param}: {out_of_range} values outside range [{min_val}, {max_val}] ({pct:.2f}%)")

validity_rate = 1 - (invalid_values / total_values_checked) if total_values_checked > 0 else 1
validity_metrics = {
    'total_values_checked': total_values_checked,
    'invalid_values': int(invalid_values),
    'validity_rate': round(validity_rate, 4),
    'issues': validity_issues,
    'status': 'PASS' if validity_rate > 0.90 else 'FAIL'
}

print(f"\n   Validity Rate: {validity_rate*100:.2f}%")
print(f"   Status: {validity_metrics['status']}")

quality_report['validity'] = validity_metrics

# 3.3 Consistency Check
print("\n   C. Consistency Analysis")

consistency_issues = []

# Check 1: Date consistency
if 'Date (DD/MM/YYYY)' in df_original.columns:
    dates = pd.to_datetime(df_original['Date (DD/MM/YYYY)'], format='%d/%m/%Y', errors='coerce')
    invalid_dates = dates.isnull().sum()
    if invalid_dates > 0:
        consistency_issues.append(f"Invalid date formats: {invalid_dates}")

# Check 2: Temperature consistency (Sample should be close to Ambient)
if 'Sample_Temp' in df_check.columns and 'Ambient_Temp' in df_check.columns:
    temp_diff = (df_check['Sample_Temp'] - df_check['Ambient_Temp']).abs()
    extreme_diff = (temp_diff > 15).sum()
    if extreme_diff > 0:
        consistency_issues.append(f"Sample-Ambient temp difference >15°C: {extreme_diff} records")
        print(f"   ⚠ Large temperature differences: {extreme_diff} records")

# Check 3: EC-TDS relationship (TDS should be ~0.5-0.7 of EC)
if 'EC' in df_check.columns and 'TDS' in df_check.columns:
    ratio = df_check['TDS'] / (df_check['EC'] + 1)
    abnormal_ratio = ((ratio < 0.3) | (ratio > 0.9)).sum()
    if abnormal_ratio > 0:
        consistency_issues.append(f"Abnormal EC-TDS ratio: {abnormal_ratio} records")
        print(f"   ⚠ Abnormal EC-TDS ratios: {abnormal_ratio} records")

# Check 4: Duplicate records
duplicates = df_original.duplicated().sum()
if duplicates > 0:
    consistency_issues.append(f"Duplicate records: {duplicates}")
    print(f"   ⚠ Duplicate records: {duplicates}")

consistency_rate = 1 - (len(consistency_issues) / 10)  # Assume 10 possible consistency checks
consistency_metrics = {
    'issues_found': len(consistency_issues),
    'issues': consistency_issues,
    'duplicate_records': int(duplicates),
    'consistency_rate': round(max(0, consistency_rate), 4),
    'status': 'PASS' if len(consistency_issues) < 3 else 'WARNING'
}

print(f"\n   Consistency Issues: {len(consistency_issues)}")
print(f"   Status: {consistency_metrics['status']}")

quality_report['consistency'] = consistency_metrics

# 3.4 Accuracy Check (Statistical)
print("\n   D. Accuracy Analysis (Statistical Validation)")

accuracy_metrics = {}

# Check for statistical outliers using IQR method
outlier_count = 0
total_numeric_values = 0

for param in PARAMETER_RANGES.keys():
    if param in df_check.columns:
        values = df_check[param].dropna()
        total_numeric_values += len(values)
        
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((values < lower_bound) | (values > upper_bound)).sum()
        outlier_count += outliers

outlier_rate = outlier_count / total_numeric_values if total_numeric_values > 0 else 0
accuracy_metrics['total_values'] = total_numeric_values
accuracy_metrics['outliers_detected'] = int(outlier_count)
accuracy_metrics['outlier_rate'] = round(outlier_rate, 4)
accuracy_metrics['status'] = 'PASS' if outlier_rate < QUALITY_THRESHOLDS['outlier_threshold'] else 'WARNING'

print(f"   Statistical Outliers: {outlier_count} ({outlier_rate*100:.2f}%)")
print(f"   Status: {accuracy_metrics['status']}")

quality_report['accuracy'] = accuracy_metrics

# 3.5 Timeliness Check
print("\n   E. Timeliness Analysis")

if 'Date (DD/MM/YYYY)' in df_original.columns:
    dates = pd.to_datetime(df_original['Date (DD/MM/YYYY)'], format='%d/%m/%Y', errors='coerce')
    dates = dates.dropna().sort_values()
    
    if len(dates) > 0:
        date_gaps = dates.diff().dt.days.dropna()
        max_gap = date_gaps.max()
        avg_gap = date_gaps.mean()
        
        large_gaps = (date_gaps > QUALITY_THRESHOLDS['max_days_gap']).sum()
        
        timeliness_metrics = {
            'start_date': dates.min().strftime('%Y-%m-%d'),
            'end_date': dates.max().strftime('%Y-%m-%d'),
            'total_days': (dates.max() - dates.min()).days,
            'unique_dates': len(dates.unique()),
            'max_gap_days': int(max_gap),
            'avg_gap_days': round(avg_gap, 2),
            'large_gaps_count': int(large_gaps),
            'status': 'PASS' if large_gaps < 5 else 'WARNING'
        }
        
        print(f"   Date Range: {timeliness_metrics['start_date']} to {timeliness_metrics['end_date']}")
        print(f"   Max Gap: {max_gap} days")
        print(f"   Large Gaps (>{QUALITY_THRESHOLDS['max_days_gap']} days): {large_gaps}")
        print(f"   Status: {timeliness_metrics['status']}")
        
        quality_report['timeliness'] = timeliness_metrics

# ============================================================================
# 4. CALCULATE OVERALL QUALITY SCORE
# ============================================================================
print("\n[3] CALCULATING OVERALL QUALITY SCORE...")

scores = {
    'Completeness': completeness_metrics['completeness_rate'] * 100,
    'Validity': validity_metrics['validity_rate'] * 100,
    'Consistency': consistency_metrics['consistency_rate'] * 100,
    'Accuracy': (1 - accuracy_metrics['outlier_rate']) * 100,
}

overall_score = np.mean(list(scores.values()))

quality_report['quality_score'] = {
    'completeness_score': round(scores['Completeness'], 2),
    'validity_score': round(scores['Validity'], 2),
    'consistency_score': round(scores['Consistency'], 2),
    'accuracy_score': round(scores['Accuracy'], 2),
    'overall_score': round(overall_score, 2),
    'grade': 'A' if overall_score >= 90 else 'B' if overall_score >= 80 else 'C' if overall_score >= 70 else 'D'
}

print(f"\n   Overall Data Quality Score: {overall_score:.2f}/100")
print(f"   Grade: {quality_report['quality_score']['grade']}")

# ============================================================================
# 5. VISUALIZE QUALITY METRICS
# ============================================================================
print("\n[4] CREATING QUALITY VISUALIZATIONS...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 5.1 Quality Score Dashboard
ax1 = fig.add_subplot(gs[0, :])
categories = list(scores.keys())
values = list(scores.values())
colors = ['#2ecc71' if v >= 90 else '#f39c12' if v >= 70 else '#e74c3c' for v in values]

bars = ax1.barh(categories, values, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlim(0, 100)
ax1.set_xlabel('Quality Score (%)', fontsize=12, fontweight='bold')
ax1.set_title(f'Data Quality Dashboard - Overall Score: {overall_score:.1f}/100 (Grade {quality_report["quality_score"]["grade"]})', 
              fontsize=14, fontweight='bold')
ax1.axvline(x=90, color='green', linestyle='--', alpha=0.5, label='Excellent (90%)')
ax1.axvline(x=70, color='orange', linestyle='--', alpha=0.5, label='Acceptable (70%)')

for i, (bar, val) in enumerate(zip(bars, values)):
    ax1.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')

ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)

# 5.2 Completeness by Column
ax2 = fig.add_subplot(gs[1, 0])
missing_pct = (df_original.isnull().sum() / len(df_original) * 100).sort_values(ascending=False)[:10]
if len(missing_pct) > 0:
    missing_pct.plot(kind='barh', ax=ax2, color='coral', edgecolor='black')
    ax2.set_xlabel('Missing %', fontsize=10, fontweight='bold')
    ax2.set_title('Top 10 Columns with Missing Data', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'No Missing Data!', ha='center', va='center', 
             fontsize=14, fontweight='bold', color='green')
    ax2.set_title('Completeness Check', fontsize=11, fontweight='bold')

# 5.3 Outlier Detection
ax3 = fig.add_subplot(gs[1, 1])
outlier_data = []
for param in ['pH', 'DO', 'Turbidity', 'EC', 'TDS']:
    if param in df_check.columns:
        values = df_check[param].dropna()
        Q1, Q3 = values.quantile(0.25), values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((values < Q1 - 3*IQR) | (values > Q3 + 3*IQR)).sum()
        outlier_data.append({'Parameter': param, 'Outliers': outliers})

if outlier_data:
    outlier_df = pd.DataFrame(outlier_data)
    outlier_df.plot(x='Parameter', y='Outliers', kind='bar', ax=ax3, 
                    color='salmon', edgecolor='black', legend=False)
    ax3.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax3.set_title('Statistical Outliers by Parameter', fontsize=11, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)

# 5.4 Temporal Coverage
ax4 = fig.add_subplot(gs[1, 2])
if 'Date (DD/MM/YYYY)' in df_original.columns:
    dates = pd.to_datetime(df_original['Date (DD/MM/YYYY)'], format='%d/%m/%Y', errors='coerce')
    dates = dates.dropna()
    date_counts = dates.value_counts().sort_index()
    
    ax4.plot(date_counts.index, date_counts.values, marker='o', linewidth=2, 
             markersize=6, color='steelblue')
    ax4.set_xlabel('Date', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Samples', fontsize=10, fontweight='bold')
    ax4.set_title('Sampling Frequency Over Time', fontsize=11, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(alpha=0.3)

# 5.5 Validity Issues
ax5 = fig.add_subplot(gs[2, 0])
if validity_issues:
    issue_df = pd.DataFrame(validity_issues)
    issue_df.plot(x='parameter', y='percentage', kind='bar', ax=ax5,
                  color='orange', edgecolor='black', legend=False)
    ax5.set_ylabel('Out-of-Range %', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Parameter', fontsize=10, fontweight='bold')
    ax5.set_title('Validity Issues (Out-of-Range Values)', fontsize=11, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(axis='y', alpha=0.3)
else:
    ax5.text(0.5, 0.5, 'All Values Within\nValid Ranges!', ha='center', va='center',
             fontsize=14, fontweight='bold', color='green')
    ax5.set_title('Validity Check', fontsize=11, fontweight='bold')

# 5.6 Consistency Matrix
ax6 = fig.add_subplot(gs[2, 1])
consistency_data = pd.DataFrame({
    'Check': ['Dates', 'Temperature', 'EC-TDS', 'Duplicates'],
    'Status': [1 if 'Invalid date' not in str(consistency_issues) else 0,
               1 if 'temp difference' not in str(consistency_issues) else 0,
               1 if 'EC-TDS' not in str(consistency_issues) else 0,
               1 if duplicates == 0 else 0]
})

colors_matrix = ['green' if s == 1 else 'red' for s in consistency_data['Status']]
consistency_data.plot(x='Check', y='Status', kind='bar', ax=ax6, 
                      color=colors_matrix, edgecolor='black', legend=False)
ax6.set_ylim(0, 1.2)
ax6.set_ylabel('Pass (1) / Fail (0)', fontsize=10, fontweight='bold')
ax6.set_title('Consistency Checks', fontsize=11, fontweight='bold')
ax6.tick_params(axis='x', rotation=45)
ax6.grid(axis='y', alpha=0.3)

# 5.7 Quality Trend (if historical data available)
ax7 = fig.add_subplot(gs[2, 2])
quality_trend = pd.DataFrame({
    'Dimension': ['Completeness', 'Validity', 'Consistency', 'Accuracy'],
    'Current': [scores['Completeness'], scores['Validity'], 
                scores['Consistency'], scores['Accuracy']],
    'Target': [95, 95, 95, 90]
})

x = np.arange(len(quality_trend))
width = 0.35

bars1 = ax7.bar(x - width/2, quality_trend['Current'], width, label='Current',
                color='steelblue', edgecolor='black')
bars2 = ax7.bar(x + width/2, quality_trend['Target'], width, label='Target',
                color='lightgreen', edgecolor='black')

ax7.set_ylabel('Score (%)', fontsize=10, fontweight='bold')
ax7.set_title('Current vs Target Quality', fontsize=11, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(quality_trend['Dimension'], rotation=45, ha='right')
ax7.legend()
ax7.grid(axis='y', alpha=0.3)
ax7.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('data_quality_reports/quality_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: quality_dashboard.png")
plt.close()

# ============================================================================
# 6. GENERATE DETAILED AUDIT REPORT
# ============================================================================
print("\n[5] GENERATING AUDIT REPORTS...")

# Save JSON report
with open('data_quality_reports/quality_audit_report.json', 'w', encoding='utf-8') as f:
    json.dump(quality_report, f, indent=2, default=str)
print("✓ Saved: quality_audit_report.json")

# Generate text report
report_lines = []
report_lines.append("=" * 80)
report_lines.append("DATA QUALITY AUDIT REPORT")
report_lines.append("=" * 80)
report_lines.append(f"\nGenerated: {quality_report['timestamp']}")
report_lines.append(f"Dataset: River Water Quality Parameters")
report_lines.append(f"Records: {len(df_original)}")
report_lines.append(f"Columns: {len(df_original.columns)}")

report_lines.append("\n" + "=" * 80)
report_lines.append("OVERALL QUALITY SCORE")
report_lines.append("=" * 80)
report_lines.append(f"\nOverall Score: {overall_score:.2f}/100")
report_lines.append(f"Grade: {quality_report['quality_score']['grade']}")
report_lines.append("\nDimension Scores:")
for dim, score in scores.items():
    status = "✓ PASS" if score >= 90 else "⚠ WARNING" if score >= 70 else "✗ FAIL"
    report_lines.append(f"  {dim:15s}: {score:6.2f}% {status}")

report_lines.append("\n" + "=" * 80)
report_lines.append("DETAILED FINDINGS")
report_lines.append("=" * 80)

report_lines.append("\n1. COMPLETENESS")
report_lines.append(f"   Completeness Rate: {completeness_metrics['completeness_rate']*100:.2f}%")
report_lines.append(f"   Missing Cells: {completeness_metrics['missing_cells']}")
report_lines.append(f"   Status: {completeness_metrics['status']}")

if len(missing_by_column) > 0:
    report_lines.append("\n   Columns with Missing Data:")
    for col, count in missing_by_column.items():
        pct = (count / len(df_original)) * 100
        report_lines.append(f"   - {col}: {count} ({pct:.2f}%)")

report_lines.append("\n2. VALIDITY")
report_lines.append(f"   Validity Rate: {validity_metrics['validity_rate']*100:.2f}%")
report_lines.append(f"   Invalid Values: {validity_metrics['invalid_values']}")
report_lines.append(f"   Status: {validity_metrics['status']}")

if validity_issues:
    report_lines.append("\n   Out-of-Range Values:")
    for issue in validity_issues:
        report_lines.append(f"   - {issue['parameter']}: {issue['out_of_range_count']} values " +
                          f"({issue['percentage']:.2f}%) outside {issue['valid_range']}")

report_lines.append("\n3. CONSISTENCY")
report_lines.append(f"   Issues Found: {consistency_metrics['issues_found']}")
report_lines.append(f"   Status: {consistency_metrics['status']}")

if consistency_issues:
    report_lines.append("\n   Consistency Issues:")
    for issue in consistency_issues:
        report_lines.append(f"   - {issue}")

report_lines.append("\n4. ACCURACY")
report_lines.append(f"   Statistical Outliers: {accuracy_metrics['outliers_detected']} ({accuracy_metrics['outlier_rate']*100:.2f}%)")
report_lines.append(f"   Status: {accuracy_metrics['status']}")

if 'timeliness' in quality_report:
    report_lines.append("\n5. TIMELINESS")
    report_lines.append(f"   Date Range: {quality_report['timeliness']['start_date']} to {quality_report['timeliness']['end_date']}")
    report_lines.append(f"   Total Days: {quality_report['timeliness']['total_days']}")
    report_lines.append(f"   Unique Dates: {quality_report['timeliness']['unique_dates']}")
    report_lines.append(f"   Max Gap: {quality_report['timeliness']['max_gap_days']} days")
    report_lines.append(f"   Large Gaps: {quality_report['timeliness']['large_gaps_count']}")
    report_lines.append(f"   Status: {quality_report['timeliness']['status']}")

report_lines.append("\n" + "=" * 80)
report_lines.append("RECOMMENDATIONS")
report_lines.append("=" * 80)

recommendations = []

if completeness_metrics['completeness_rate'] < 0.95:
    recommendations.append("- Investigate and reduce missing data (target: >95% completeness)")

if validity_metrics['validity_rate'] < 0.95:
    recommendations.append("- Review data collection procedures to reduce out-of-range values")

if len(consistency_issues) > 2:
    recommendations.append("- Implement automated consistency checks during data entry")

if accuracy_metrics['outlier_rate'] > 0.05:
    recommendations.append("- Review outlier detection procedures and validate extreme values")

if 'timeliness' in quality_report and quality_report['timeliness']['large_gaps_count'] > 3:
    recommendations.append("- Establish more consistent sampling schedule to reduce data gaps")

if not recommendations:
    recommendations.append("✓ Data quality meets all standards - continue current practices")

for rec in recommendations:
    report_lines.append(rec)

report_lines.append("\n" + "=" * 80)
report_lines.append("END OF REPORT")
report_lines.append("=" * 80)

# Save text report
report_text = "\n".join(report_lines)
with open('data_quality_reports/quality_audit_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)
print("✓ Saved: quality_audit_report.txt")

# Print report to console
print("\n" + report_text)

# ============================================================================
# 7. CREATE MONITORING SUMMARY
# ============================================================================

summary_df = pd.DataFrame({
    'Metric': ['Overall Score', 'Completeness', 'Validity', 'Consistency', 'Accuracy'],
    'Score': [overall_score, scores['Completeness'], scores['Validity'], 
              scores['Consistency'], scores['Accuracy']],
    'Status': [quality_report['quality_score']['grade'],
               completeness_metrics['status'],
               validity_metrics['status'],
               consistency_metrics['status'],
               accuracy_metrics['status']]
})

summary_df.to_csv('data_quality_reports/quality_summary.csv', index=False, encoding='utf-8')
print("✓ Saved: quality_summary.csv")

print("\n" + "=" * 80)
print("DATA QUALITY AUDIT COMPLETED!")
print("=" * 80)
print("\nGenerated Files:")
print("  1. quality_dashboard.png - Visual quality metrics")
print("  2. quality_audit_report.json - Machine-readable audit")
print("  3. quality_audit_report.txt - Human-readable report")
print("  4. quality_summary.csv - Summary metrics")
print(f"\nOverall Data Quality: {overall_score:.2f}/100 (Grade {quality_report['quality_score']['grade']})")
