"""
Enhanced Date Cleaning Script for River Water Quality Dataset
Performs comprehensive date validation, cleaning, and feature extraction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ENHANCED DATE CLEANING FOR RIVER WATER QUALITY DATA")
print("=" * 80)

# ============================================================================
# 1. LOAD RAW DATA
# ============================================================================
print("\n[1] LOADING DATA...")
try:
    # Try to load from parent directory
    df = pd.read_csv(r'c:\Users\Millpark\Downloads\River water parameters.csv')
    print("✓ Data loaded successfully from parent directory!")
except FileNotFoundError:
    # Fallback to current directory
    try:
        df = pd.read_csv('River water parameters.csv')
        print("✓ Data loaded successfully from current directory!")
    except FileNotFoundError:
        print("✗ Error: Could not find 'River water parameters.csv'")
        print("   Please ensure the file exists in the correct location.")
        exit()

print(f"   Shape: {df.shape}")
print(f"\n   Columns: {list(df.columns)}")

# ============================================================================
# 2. IDENTIFY DATE COLUMN
# ============================================================================
print("\n[2] IDENTIFYING DATE COLUMN...")

# Find the date column (could be 'Date (DD/MM/YYYY)' or already renamed)
date_col = None
for col in df.columns:
    if 'date' in col.lower():
        date_col = col
        break

if date_col is None:
    print("✗ Error: No date column found!")
    exit()

print(f"✓ Date column identified: '{date_col}'")
print(f"\n   Sample values:")
print(df[date_col].head(10))
print(f"\n   Data type: {df[date_col].dtype}")
print(f"   Total records: {len(df)}")
print(f"   Missing values: {df[date_col].isnull().sum()}")

# ============================================================================
# 3. DATE CLEANING AND VALIDATION
# ============================================================================
print("\n[3] CLEANING AND VALIDATING DATES...")

# Store original data for comparison
df['Date_Original'] = df[date_col].copy()

# Initialize tracking variables
issues_found = {
    'null_dates': 0,
    'invalid_format': 0,
    'future_dates': 0,
    'unrealistic_dates': 0,
    'duplicate_dates': 0
}

# 3.1 Check for null/missing dates
null_dates = df[date_col].isnull().sum()
issues_found['null_dates'] = null_dates
if null_dates > 0:
    print(f"\n⚠ Found {null_dates} missing dates")
    print(f"   Action: Will mark for review")

# 3.2 Convert to datetime with error handling
print("\n   Converting to datetime format...")
df['Date_Cleaned'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')

# Check for conversion errors
conversion_errors = df['Date_Cleaned'].isnull().sum() - null_dates
issues_found['invalid_format'] = conversion_errors
if conversion_errors > 0:
    print(f"⚠ Found {conversion_errors} dates with invalid format")
    invalid_samples = df[df['Date_Cleaned'].isnull() & df[date_col].notna()][date_col].head(5)
    print(f"   Examples: {list(invalid_samples)}")
    print(f"   Action: These will be marked for manual review")

# 3.3 Check for future dates
today = pd.Timestamp.now()
future_dates = (df['Date_Cleaned'] > today).sum()
issues_found['future_dates'] = future_dates
if future_dates > 0:
    print(f"\n⚠ Found {future_dates} dates in the future")
    print(f"   Latest date in dataset: {df['Date_Cleaned'].max()}")
    print(f"   Today's date: {today.date()}")
    print(f"   Action: These may need correction")

# 3.4 Check for unrealistic dates (e.g., before 1900 or more than 1 year in future)
min_reasonable_date = pd.Timestamp('1900-01-01')
max_reasonable_date = today + pd.Timedelta(days=365)
unrealistic = ((df['Date_Cleaned'] < min_reasonable_date) | 
               (df['Date_Cleaned'] > max_reasonable_date)).sum()
issues_found['unrealistic_dates'] = unrealistic
if unrealistic > 0:
    print(f"\n⚠ Found {unrealistic} dates outside reasonable range (1900 - next year)")
    print(f"   Date range in dataset: {df['Date_Cleaned'].min()} to {df['Date_Cleaned'].max()}")

# 3.5 Check for duplicate dates at same location
print("\n   Checking for duplicate date-location combinations...")
# First, identify the sampling point column
sampling_col = None
for col in df.columns:
    if 'sampling' in col.lower() or 'point' in col.lower() or 'location' in col.lower():
        sampling_col = col
        break

if sampling_col:
    duplicates = df.groupby([sampling_col, 'Date_Cleaned']).size()
    duplicate_entries = duplicates[duplicates > 1]
    issues_found['duplicate_dates'] = len(duplicate_entries)
    if len(duplicate_entries) > 0:
        print(f"⚠ Found {len(duplicate_entries)} date-location combinations with multiple readings")
        print(f"   This may be intentional (multiple samples per day)")
        print(f"   Examples:")
        print(duplicate_entries.head())

# ============================================================================
# 4. SUMMARY OF DATE CLEANING
# ============================================================================
print("\n[4] DATE CLEANING SUMMARY")
print("-" * 80)

total_issues = sum(issues_found.values())
print(f"\n   Total records: {len(df)}")
print(f"   Valid dates: {df['Date_Cleaned'].notna().sum()}")
print(f"   Issues found: {total_issues}")
print(f"\n   Breakdown:")
for issue, count in issues_found.items():
    if count > 0:
        print(f"   - {issue.replace('_', ' ').title()}: {count}")

if total_issues == 0:
    print("\n✓ All dates are clean and valid!")
else:
    print(f"\n⚠ Data quality: {((len(df) - total_issues) / len(df) * 100):.2f}% clean")

# ============================================================================
# 5. EXTRACT COMPREHENSIVE TEMPORAL FEATURES
# ============================================================================
print("\n[5] EXTRACTING TEMPORAL FEATURES...")

# Basic temporal features
df['Year'] = df['Date_Cleaned'].dt.year
df['Month'] = df['Date_Cleaned'].dt.month
df['Month_Name'] = df['Date_Cleaned'].dt.month_name()
df['Day'] = df['Date_Cleaned'].dt.day
df['DayOfWeek'] = df['Date_Cleaned'].dt.dayofweek
df['DayOfWeek_Name'] = df['Date_Cleaned'].dt.day_name()
df['Week'] = df['Date_Cleaned'].dt.isocalendar().week
df['Quarter'] = df['Date_Cleaned'].dt.quarter
df['DayOfYear'] = df['Date_Cleaned'].dt.dayofyear

# Season classification (Southern Hemisphere - South Africa)
def get_season_southern_hemisphere(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:  # 9, 10, 11
        return 'Spring'

df['Season'] = df['Month'].apply(get_season_southern_hemisphere)

# Weekend indicator
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Days since first measurement
if df['Date_Cleaned'].notna().sum() > 0:
    first_date = df['Date_Cleaned'].min()
    df['Days_Since_Start'] = (df['Date_Cleaned'] - first_date).dt.days
    df['Weeks_Since_Start'] = df['Days_Since_Start'] // 7

print("✓ Temporal features extracted:")
temporal_features = ['Year', 'Month', 'Month_Name', 'Day', 'DayOfWeek', 
                     'DayOfWeek_Name', 'Week', 'Quarter', 'DayOfYear', 
                     'Season', 'IsWeekend', 'Days_Since_Start', 'Weeks_Since_Start']
for feature in temporal_features:
    print(f"   - {feature}")

# ============================================================================
# 6. DATE RANGE ANALYSIS
# ============================================================================
print("\n[6] DATE RANGE ANALYSIS")
print("-" * 80)

if df['Date_Cleaned'].notna().sum() > 0:
    min_date = df['Date_Cleaned'].min()
    max_date = df['Date_Cleaned'].max()
    date_range = (max_date - min_date).days
    
    print(f"\n   Study Period:")
    print(f"   - Start Date: {min_date.strftime('%d %B %Y')}")
    print(f"   - End Date: {max_date.strftime('%d %B %Y')}")
    print(f"   - Duration: {date_range} days ({date_range/365.25:.2f} years)")
    
    print(f"\n   Sampling Frequency:")
    unique_dates = df['Date_Cleaned'].nunique()
    print(f"   - Unique sampling dates: {unique_dates}")
    print(f"   - Total measurements: {len(df)}")
    print(f"   - Average measurements per date: {len(df)/unique_dates:.1f}")
    
    # Monthly distribution
    print(f"\n   Monthly Distribution:")
    monthly_counts = df.groupby('Month_Name')['Date_Cleaned'].count()
    # Sort by month number
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_counts = monthly_counts.reindex([m for m in month_order if m in monthly_counts.index])
    for month, count in monthly_counts.items():
        print(f"   - {month}: {count} measurements")
    
    # Seasonal distribution
    print(f"\n   Seasonal Distribution:")
    seasonal_counts = df.groupby('Season')['Date_Cleaned'].count()
    for season, count in seasonal_counts.items():
        print(f"   - {season}: {count} measurements")

# ============================================================================
# 7. IDENTIFY DATA GAPS
# ============================================================================
print("\n[7] IDENTIFYING DATA GAPS")
print("-" * 80)

if df['Date_Cleaned'].notna().sum() > 1:
    # Sort by date
    df_sorted = df.sort_values('Date_Cleaned')
    
    # Calculate gaps between consecutive measurements
    df_sorted['Date_Diff'] = df_sorted['Date_Cleaned'].diff()
    
    # Find significant gaps (more than 7 days)
    significant_gaps = df_sorted[df_sorted['Date_Diff'] > pd.Timedelta(days=7)]
    
    if len(significant_gaps) > 0:
        print(f"\n   Found {len(significant_gaps)} gaps of more than 7 days:")
        for idx, row in significant_gaps.head(10).iterrows():
            gap_days = row['Date_Diff'].days
            print(f"   - {gap_days} days gap ending on {row['Date_Cleaned'].strftime('%d %B %Y')}")
    else:
        print("\n   ✓ No significant gaps found (all measurements within 7 days)")

# ============================================================================
# 8. SAVE CLEANED DATA
# ============================================================================
print("\n[8] SAVING CLEANED DATA...")

# Create output DataFrame with cleaned dates
output_cols = ['Date_Original', 'Date_Cleaned'] + temporal_features
available_cols = [col for col in output_cols if col in df.columns]

# Save date cleaning report
date_report = df[available_cols].copy()
date_report.to_csv('date_cleaning_report.csv', index=False, encoding='utf-8')
print("✓ Saved: date_cleaning_report.csv")

# Save full dataset with cleaned dates
df.to_csv('river_water_dates_cleaned.csv', index=False, encoding='utf-8')
print("✓ Saved: river_water_dates_cleaned.csv")

# ============================================================================
# 9. RECOMMENDATIONS
# ============================================================================
print("\n[9] RECOMMENDATIONS")
print("-" * 80)

recommendations = []

if issues_found['null_dates'] > 0:
    recommendations.append(f"- Review {issues_found['null_dates']} records with missing dates")

if issues_found['invalid_format'] > 0:
    recommendations.append(f"- Manually verify {issues_found['invalid_format']} dates with invalid format")

if issues_found['future_dates'] > 0:
    recommendations.append(f"- Check {issues_found['future_dates']} future dates for data entry errors")

if unique_dates < 50:
    recommendations.append("- Consider increasing sampling frequency for better temporal resolution")

if len(significant_gaps) > 5:
    recommendations.append("- Review data collection schedule to minimize sampling gaps")

if len(recommendations) > 0:
    print("\n   Actions needed:")
    for rec in recommendations:
        print(f"   {rec}")
else:
    print("\n   ✓ No immediate actions required - dates are clean!")

print("\n" + "=" * 80)
print("DATE CLEANING COMPLETED!")
print("=" * 80)
print("\nOutput files:")
print("   1. date_cleaning_report.csv - Temporal features for each record")
print("   2. river_water_dates_cleaned.csv - Full dataset with cleaned dates")
print("\nNext steps:")
print("   - Review the date_cleaning_report.csv for any anomalies")
print("   - Use temporal features for trend analysis and seasonal patterns")
print("   - Integrate cleaned dates into your main analysis pipeline")
