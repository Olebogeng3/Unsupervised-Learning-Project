"""
Power BI Dashboard Data Preparation
River Water Quality Dataset
Prepares optimized datasets and creates dashboard specification for Power BI
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("POWER BI DASHBOARD DATA PREPARATION")
print("River Water Quality Analysis")
print("=" * 80)

# ============================================================================
# 1. LOAD AND PREPARE DATA FOR POWER BI
# ============================================================================

print("\n[1] LOADING DATA...")

# Load all datasets
df_preprocessed = pd.read_csv('river_water_preprocessed.csv')
df_engineered = pd.read_csv('river_water_features_engineered.csv')
df_dates = pd.read_csv('river_water_dates_cleaned.csv')

print(f"✓ Preprocessed data: {df_preprocessed.shape}")
print(f"✓ Engineered features: {df_engineered.shape}")
print(f"✓ Date-cleaned data: {df_dates.shape}")

# ============================================================================
# 2. CREATE MAIN DASHBOARD DATASET
# ============================================================================

print("\n[2] CREATING MAIN DASHBOARD DATASET...")

# Combine key data for Power BI
dashboard_data = df_dates.copy()

# Add water quality categorization
def categorize_do(do_value):
    if pd.isna(do_value):
        return 'Unknown'
    elif do_value < 2.0:
        return 'Severely Hypoxic'
    elif do_value < 4.0:
        return 'Hypoxic'
    elif do_value < 6.0:
        return 'Low'
    else:
        return 'Adequate'

def categorize_overall_quality(do_value):
    if pd.isna(do_value):
        return 'Unknown'
    elif do_value < 2.0:
        return 'Critical'
    elif do_value < 4.0:
        return 'Poor'
    elif do_value < 6.0:
        return 'Fair'
    else:
        return 'Good'

dashboard_data['DO_Category'] = dashboard_data['DO\n(mg/L)'].apply(categorize_do)
dashboard_data['Overall_Quality'] = dashboard_data['DO\n(mg/L)'].apply(categorize_overall_quality)

# Add pollution severity
def pollution_severity(turbidity, tss):
    if pd.isna(turbidity) or pd.isna(tss):
        return 'Unknown'
    score = (turbidity / 100) + (tss / 50)
    if score < 1:
        return 'Low'
    elif score < 3:
        return 'Moderate'
    elif score < 6:
        return 'High'
    else:
        return 'Very High'

dashboard_data['Pollution_Severity'] = dashboard_data.apply(
    lambda row: pollution_severity(row['Turbidity (NTU)'], row['TSS\n(mL sed/L)']), axis=1
)

# Rename columns for Power BI (remove special characters)
dashboard_data = dashboard_data.rename(columns={
    'Date (DD/MM/YYYY)': 'Date',
    'Time (24 hrs XX:XX)': 'Time',
    'Sampling point': 'Sampling_Point',
    'Ambient temperature (°C)': 'Ambient_Temp_C',
    'Ambient humidity': 'Ambient_Humidity_Pct',
    'Sample temperature (°C)': 'Sample_Temp_C',
    'EC\n(µS/cm)': 'EC_uS_cm',
    'TDS\n(mg/L)': 'TDS_mg_L',
    'TSS\n(mL sed/L)': 'TSS_mL_L',
    'DO\n(mg/L)': 'DO_mg_L',
    'Level (cm)': 'Level_cm',
    'Turbidity (NTU)': 'Turbidity_NTU',
    'Hardness\n(mg CaCO3/L)': 'Hardness_mg_L',
    'Hardness classification': 'Hardness_Class',
    'Total Cl-\n(mg Cl-/L)': 'Total_Chlorine_mg_L'
})

# Convert date to proper datetime format for Power BI
dashboard_data['Date'] = pd.to_datetime(dashboard_data['Date'], format='%d/%m/%Y')

# Add additional time dimensions
dashboard_data['Year'] = dashboard_data['Date'].dt.year
dashboard_data['Month'] = dashboard_data['Date'].dt.month
dashboard_data['Month_Name'] = dashboard_data['Date'].dt.strftime('%B')
dashboard_data['Week'] = dashboard_data['Date'].dt.isocalendar().week
dashboard_data['Day_of_Week'] = dashboard_data['Date'].dt.day_name()
dashboard_data['Quarter'] = dashboard_data['Date'].dt.quarter

# Add season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

dashboard_data['Season'] = dashboard_data['Month'].apply(get_season)

# Add WHO compliance flags
dashboard_data['DO_WHO_Compliant'] = dashboard_data['DO_mg_L'] >= 6.0
dashboard_data['pH_WHO_Compliant'] = (dashboard_data['pH'] >= 6.5) & (dashboard_data['pH'] <= 8.5)
dashboard_data['Turbidity_WHO_Compliant'] = dashboard_data['Turbidity_NTU'] <= 5.0

# Overall compliance
dashboard_data['WHO_Compliant'] = (
    dashboard_data['DO_WHO_Compliant'] & 
    dashboard_data['pH_WHO_Compliant'] & 
    dashboard_data['Turbidity_WHO_Compliant']
)

# Save main dashboard dataset
dashboard_data.to_csv('powerbi_data/main_dashboard_data.csv', index=False)
print(f"✓ Created main dashboard dataset: {dashboard_data.shape}")
print("✓ Saved: powerbi_data/main_dashboard_data.csv")

# ============================================================================
# 3. CREATE AGGREGATED SUMMARY TABLES
# ============================================================================

print("\n[3] CREATING AGGREGATED SUMMARY TABLES...")

# Summary by Sampling Point
location_summary = dashboard_data.groupby('Sampling_Point').agg({
    'DO_mg_L': ['mean', 'min', 'max', 'std'],
    'pH': ['mean', 'min', 'max'],
    'EC_uS_cm': ['mean', 'min', 'max'],
    'Turbidity_NTU': ['mean', 'min', 'max'],
    'TSS_mL_L': ['mean', 'min', 'max'],
    'WHO_Compliant': ['sum', 'count']
}).round(2)

location_summary.columns = ['_'.join(col).strip() for col in location_summary.columns.values]
location_summary['Compliance_Rate_Pct'] = (
    location_summary['WHO_Compliant_sum'] / location_summary['WHO_Compliant_count'] * 100
).round(1)
location_summary = location_summary.reset_index()

location_summary.to_csv('powerbi_data/location_summary.csv', index=False)
print("✓ Saved: powerbi_data/location_summary.csv")

# Summary by Month
monthly_summary = dashboard_data.groupby(['Year', 'Month', 'Month_Name']).agg({
    'DO_mg_L': ['mean', 'min', 'max'],
    'Sample_Temp_C': ['mean'],
    'Turbidity_NTU': ['mean'],
    'WHO_Compliant': ['sum', 'count']
}).round(2)

monthly_summary.columns = ['_'.join(col).strip() for col in monthly_summary.columns.values]
monthly_summary['Compliance_Rate_Pct'] = (
    monthly_summary['WHO_Compliant_sum'] / monthly_summary['WHO_Compliant_count'] * 100
).round(1)
monthly_summary = monthly_summary.reset_index()

monthly_summary.to_csv('powerbi_data/monthly_summary.csv', index=False)
print("✓ Saved: powerbi_data/monthly_summary.csv")

# Summary by Water Quality Category
quality_summary = dashboard_data.groupby('DO_Category').agg({
    'DO_mg_L': ['count', 'mean', 'min', 'max'],
    'Sample_Temp_C': ['mean'],
    'pH': ['mean']
}).round(2)

quality_summary.columns = ['_'.join(col).strip() for col in quality_summary.columns.values]
quality_summary['Percentage'] = (
    quality_summary['DO_mg_L_count'] / quality_summary['DO_mg_L_count'].sum() * 100
).round(1)
quality_summary = quality_summary.reset_index()

quality_summary.to_csv('powerbi_data/quality_category_summary.csv', index=False)
print("✓ Saved: powerbi_data/quality_category_summary.csv")

# ============================================================================
# 4. CREATE KPI DATASET
# ============================================================================

print("\n[4] CREATING KPI DATASET...")

kpis = {
    'Total_Samples': len(dashboard_data),
    'Sampling_Locations': dashboard_data['Sampling_Point'].nunique(),
    'Sampling_Period_Days': (dashboard_data['Date'].max() - dashboard_data['Date'].min()).days,
    'Average_DO_mg_L': dashboard_data['DO_mg_L'].mean(),
    'Min_DO_mg_L': dashboard_data['DO_mg_L'].min(),
    'Max_DO_mg_L': dashboard_data['DO_mg_L'].max(),
    'Average_pH': dashboard_data['pH'].mean(),
    'Average_Turbidity_NTU': dashboard_data['Turbidity_NTU'].mean(),
    'Severely_Hypoxic_Count': (dashboard_data['DO_Category'] == 'Severely Hypoxic').sum(),
    'Severely_Hypoxic_Pct': ((dashboard_data['DO_Category'] == 'Severely Hypoxic').sum() / len(dashboard_data) * 100),
    'WHO_Compliant_Count': dashboard_data['WHO_Compliant'].sum(),
    'WHO_Compliance_Rate_Pct': (dashboard_data['WHO_Compliant'].sum() / len(dashboard_data) * 100),
    'Critical_Quality_Count': (dashboard_data['Overall_Quality'] == 'Critical').sum(),
    'Critical_Quality_Pct': ((dashboard_data['Overall_Quality'] == 'Critical').sum() / len(dashboard_data) * 100),
    'Average_Sample_Temp_C': dashboard_data['Sample_Temp_C'].mean(),
    'Average_EC_uS_cm': dashboard_data['EC_uS_cm'].mean(),
    'High_Pollution_Count': (dashboard_data['Pollution_Severity'].isin(['High', 'Very High'])).sum(),
    'High_Pollution_Pct': ((dashboard_data['Pollution_Severity'].isin(['High', 'Very High'])).sum() / len(dashboard_data) * 100)
}

kpi_df = pd.DataFrame([kpis])
kpi_df = kpi_df.round(2)

kpi_df.to_csv('powerbi_data/kpis.csv', index=False)
print("✓ Saved: powerbi_data/kpis.csv")

# ============================================================================
# 5. CREATE TIME SERIES DATASET
# ============================================================================

print("\n[5] CREATING TIME SERIES DATASET...")

# Daily aggregates
time_series = dashboard_data.groupby('Date').agg({
    'DO_mg_L': 'mean',
    'pH': 'mean',
    'Sample_Temp_C': 'mean',
    'Turbidity_NTU': 'mean',
    'EC_uS_cm': 'mean',
    'TSS_mL_L': 'mean',
    'WHO_Compliant': 'mean',
    'Sampling_Point': 'count'
}).round(2)

time_series = time_series.rename(columns={'Sampling_Point': 'Sample_Count'})
time_series = time_series.reset_index()

# Add moving averages
time_series['DO_7day_MA'] = time_series['DO_mg_L'].rolling(window=7, min_periods=1).mean().round(2)
time_series['Temp_7day_MA'] = time_series['Sample_Temp_C'].rolling(window=7, min_periods=1).mean().round(2)

time_series.to_csv('powerbi_data/time_series_data.csv', index=False)
print("✓ Saved: powerbi_data/time_series_data.csv")

# ============================================================================
# 6. CREATE CORRELATION MATRIX FOR HEATMAP
# ============================================================================

print("\n[6] CREATING CORRELATION MATRIX...")

corr_cols = ['DO_mg_L', 'pH', 'Sample_Temp_C', 'EC_uS_cm', 'TDS_mg_L', 
             'TSS_mL_L', 'Turbidity_NTU', 'Hardness_mg_L', 'Total_Chlorine_mg_L']

corr_matrix = dashboard_data[corr_cols].corr().round(3)

# Convert to long format for Power BI
corr_long = corr_matrix.reset_index().melt(id_vars='index', var_name='Parameter_2', value_name='Correlation')
corr_long = corr_long.rename(columns={'index': 'Parameter_1'})

corr_long.to_csv('powerbi_data/correlation_matrix.csv', index=False)
print("✓ Saved: powerbi_data/correlation_matrix.csv")

# ============================================================================
# 7. CREATE POWER BI DASHBOARD SPECIFICATION
# ============================================================================

print("\n[7] CREATING POWER BI DASHBOARD SPECIFICATION...")

dashboard_spec = {
    "dashboard_name": "River Water Quality Monitoring Dashboard",
    "version": "1.0",
    "created_date": datetime.now().strftime("%Y-%m-%d"),
    "data_sources": {
        "main_data": "powerbi_data/main_dashboard_data.csv",
        "location_summary": "powerbi_data/location_summary.csv",
        "monthly_summary": "powerbi_data/monthly_summary.csv",
        "quality_summary": "powerbi_data/quality_category_summary.csv",
        "kpis": "powerbi_data/kpis.csv",
        "time_series": "powerbi_data/time_series_data.csv",
        "correlation": "powerbi_data/correlation_matrix.csv"
    },
    "pages": [
        {
            "page_name": "Executive Summary",
            "description": "High-level KPIs and overview",
            "visuals": [
                {
                    "type": "Card",
                    "title": "Total Samples",
                    "measure": "Total_Samples",
                    "source": "kpis"
                },
                {
                    "type": "Card",
                    "title": "WHO Compliance Rate",
                    "measure": "WHO_Compliance_Rate_Pct",
                    "source": "kpis",
                    "format": "Percentage"
                },
                {
                    "type": "Card",
                    "title": "Average DO",
                    "measure": "Average_DO_mg_L",
                    "source": "kpis",
                    "format": "Decimal (2)",
                    "suffix": " mg/L"
                },
                {
                    "type": "Card",
                    "title": "Critical Quality %",
                    "measure": "Critical_Quality_Pct",
                    "source": "kpis",
                    "format": "Percentage",
                    "alert": "Red if > 50%"
                },
                {
                    "type": "Donut Chart",
                    "title": "Water Quality Distribution",
                    "legend": "DO_Category",
                    "values": "DO_mg_L_count",
                    "source": "quality_summary",
                    "colors": {
                        "Adequate": "#00B050",
                        "Low": "#FFC000",
                        "Hypoxic": "#FF6600",
                        "Severely Hypoxic": "#C00000"
                    }
                },
                {
                    "type": "Column Chart",
                    "title": "Samples by Location",
                    "x_axis": "Sampling_Point",
                    "y_axis": "Count",
                    "source": "main_data"
                },
                {
                    "type": "Line Chart",
                    "title": "DO Trend Over Time",
                    "x_axis": "Date",
                    "y_axis": ["DO_mg_L", "DO_7day_MA"],
                    "source": "time_series",
                    "reference_line": {
                        "value": 6.0,
                        "label": "WHO Minimum",
                        "color": "Red"
                    }
                }
            ]
        },
        {
            "page_name": "Water Quality Analysis",
            "description": "Detailed water quality parameters",
            "visuals": [
                {
                    "type": "Gauge",
                    "title": "Average DO Level",
                    "value": "Average_DO_mg_L",
                    "source": "kpis",
                    "min": 0,
                    "max": 10,
                    "target": 6.0,
                    "ranges": [
                        {"min": 0, "max": 2, "color": "Red"},
                        {"min": 2, "max": 4, "color": "Orange"},
                        {"min": 4, "max": 6, "color": "Yellow"},
                        {"min": 6, "max": 10, "color": "Green"}
                    ]
                },
                {
                    "type": "Clustered Bar Chart",
                    "title": "DO by Sampling Location",
                    "y_axis": "Sampling_Point",
                    "x_axis": ["DO_mg_L_mean", "DO_mg_L_min", "DO_mg_L_max"],
                    "source": "location_summary"
                },
                {
                    "type": "Scatter Chart",
                    "title": "Temperature vs DO Relationship",
                    "x_axis": "Sample_Temp_C",
                    "y_axis": "DO_mg_L",
                    "legend": "DO_Category",
                    "source": "main_data"
                },
                {
                    "type": "Matrix Heatmap",
                    "title": "Parameter Correlation Matrix",
                    "rows": "Parameter_1",
                    "columns": "Parameter_2",
                    "values": "Correlation",
                    "source": "correlation",
                    "color_scale": "Red-Yellow-Green",
                    "format": "Decimal (2)"
                },
                {
                    "type": "Table",
                    "title": "Location Summary Statistics",
                    "columns": [
                        "Sampling_Point",
                        "DO_mg_L_mean",
                        "pH_mean",
                        "Turbidity_NTU_mean",
                        "Compliance_Rate_Pct"
                    ],
                    "source": "location_summary"
                }
            ]
        },
        {
            "page_name": "Temporal Trends",
            "description": "Time-based analysis and seasonality",
            "visuals": [
                {
                    "type": "Line Chart",
                    "title": "DO Trend with Moving Average",
                    "x_axis": "Date",
                    "y_axis": ["DO_mg_L", "DO_7day_MA"],
                    "source": "time_series"
                },
                {
                    "type": "Area Chart",
                    "title": "Temperature Trend",
                    "x_axis": "Date",
                    "y_axis": ["Sample_Temp_C", "Temp_7day_MA"],
                    "source": "time_series"
                },
                {
                    "type": "Column Chart",
                    "title": "Monthly Average DO",
                    "x_axis": "Month_Name",
                    "y_axis": "DO_mg_L_mean",
                    "source": "monthly_summary",
                    "sort": "Month"
                },
                {
                    "type": "Clustered Column Chart",
                    "title": "Compliance Rate by Month",
                    "x_axis": "Month_Name",
                    "y_axis": "Compliance_Rate_Pct",
                    "source": "monthly_summary"
                },
                {
                    "type": "Ribbon Chart",
                    "title": "Water Quality Category Over Time",
                    "x_axis": "Date",
                    "legend": "DO_Category",
                    "y_axis": "Count",
                    "source": "main_data"
                }
            ]
        },
        {
            "page_name": "Pollution Analysis",
            "description": "Pollution indicators and severity",
            "visuals": [
                {
                    "type": "Stacked Bar Chart",
                    "title": "Pollution Severity by Location",
                    "y_axis": "Sampling_Point",
                    "legend": "Pollution_Severity",
                    "x_axis": "Count",
                    "source": "main_data"
                },
                {
                    "type": "Scatter Chart",
                    "title": "Turbidity vs TSS",
                    "x_axis": "Turbidity_NTU",
                    "y_axis": "TSS_mL_L",
                    "legend": "Sampling_Point",
                    "source": "main_data"
                },
                {
                    "type": "Line and Column Chart",
                    "title": "EC and TDS Over Time",
                    "x_axis": "Date",
                    "column_y": "EC_uS_cm",
                    "line_y": "TDS_mg_L",
                    "source": "time_series"
                },
                {
                    "type": "Funnel Chart",
                    "title": "Water Quality Funnel",
                    "values": [
                        {"category": "Total Samples", "value": "Total_Samples"},
                        {"category": "Non-Critical", "value": "WHO_Compliant_Count"},
                        {"category": "Low DO", "value": "calculated"},
                        {"category": "Adequate", "value": "calculated"}
                    ],
                    "source": "kpis"
                }
            ]
        },
        {
            "page_name": "Compliance Dashboard",
            "description": "WHO standards compliance tracking",
            "visuals": [
                {
                    "type": "KPI",
                    "title": "WHO Compliance Rate",
                    "value": "WHO_Compliance_Rate_Pct",
                    "goal": 95,
                    "trend_axis": "Date",
                    "source": "kpis"
                },
                {
                    "type": "Decomposition Tree",
                    "title": "Non-Compliance Analysis",
                    "analyze": "WHO_Compliant",
                    "explain_by": ["Sampling_Point", "Month_Name", "DO_Category"],
                    "source": "main_data"
                },
                {
                    "type": "Stacked Column Chart",
                    "title": "Compliance by Location",
                    "x_axis": "Sampling_Point",
                    "legend": "WHO_Compliant",
                    "y_axis": "Count",
                    "source": "main_data"
                },
                {
                    "type": "Table",
                    "title": "Detailed Compliance Breakdown",
                    "columns": [
                        "Sampling_Point",
                        "DO_WHO_Compliant",
                        "pH_WHO_Compliant",
                        "Turbidity_WHO_Compliant",
                        "WHO_Compliant"
                    ],
                    "source": "main_data",
                    "conditional_formatting": {
                        "WHO_Compliant": {
                            "True": "Green",
                            "False": "Red"
                        }
                    }
                }
            ]
        },
        {
            "page_name": "Drill-Down Details",
            "description": "Detailed data exploration",
            "visuals": [
                {
                    "type": "Table",
                    "title": "All Sample Data",
                    "columns": "All",
                    "source": "main_data",
                    "features": ["Sort", "Filter", "Search", "Export"]
                },
                {
                    "type": "Slicer",
                    "title": "Date Range",
                    "field": "Date",
                    "type": "Between"
                },
                {
                    "type": "Slicer",
                    "title": "Sampling Location",
                    "field": "Sampling_Point",
                    "type": "Dropdown"
                },
                {
                    "type": "Slicer",
                    "title": "Water Quality Category",
                    "field": "DO_Category",
                    "type": "List"
                }
            ]
        }
    ],
    "color_theme": {
        "primary": "#0078D4",
        "secondary": "#004578",
        "accent": "#FFC000",
        "success": "#00B050",
        "warning": "#FF6600",
        "danger": "#C00000",
        "background": "#FFFFFF",
        "text": "#000000"
    },
    "measures_to_create": [
        {
            "name": "Total Samples",
            "dax": "COUNTROWS('main_data')"
        },
        {
            "name": "Average DO",
            "dax": "AVERAGE('main_data'[DO_mg_L])"
        },
        {
            "name": "Compliance Rate",
            "dax": "DIVIDE(COUNTROWS(FILTER('main_data', 'main_data'[WHO_Compliant] = TRUE)), COUNTROWS('main_data'), 0) * 100"
        },
        {
            "name": "Critical Samples",
            "dax": "COUNTROWS(FILTER('main_data', 'main_data'[Overall_Quality] = \"Critical\"))"
        },
        {
            "name": "WHO Target",
            "dax": "6.0"
        }
    ],
    "interactions": {
        "cross_filtering": "Enabled",
        "drill_through": "Enabled on all pages",
        "bookmarks": ["Critical Samples", "Compliant Samples", "By Location"],
        "tooltips": "Custom tooltips enabled for all visuals"
    }
}

# Save specification as JSON
with open('powerbi_data/dashboard_specification.json', 'w', encoding='utf-8') as f:
    json.dump(dashboard_spec, f, indent=2, ensure_ascii=False)

print("✓ Saved: powerbi_data/dashboard_specification.json")

# ============================================================================
# 8. CREATE README FOR POWER BI
# ============================================================================

print("\n[8] CREATING POWER BI README...")

readme_content = f"""# Power BI Dashboard - River Water Quality Monitoring

## Overview
This Power BI dashboard provides comprehensive visualization and analysis of river water quality data collected from {dashboard_data['Sampling_Point'].nunique()} sampling locations between {dashboard_data['Date'].min().strftime('%B %d, %Y')} and {dashboard_data['Date'].max().strftime('%B %d, %Y')}.

## Data Files

### Main Datasets
1. **main_dashboard_data.csv** - Primary dataset with all measurements
   - {len(dashboard_data)} samples
   - {len(dashboard_data.columns)} columns
   - Includes temporal, spatial, and quality categorizations

2. **location_summary.csv** - Aggregated statistics by sampling location
   - Summary statistics (mean, min, max, std)
   - Compliance rates
   - {len(location_summary)} locations

3. **monthly_summary.csv** - Temporal aggregations
   - Monthly trends
   - Seasonal patterns
   - {len(monthly_summary)} monthly records

4. **quality_category_summary.csv** - Quality distribution
   - DO-based categorization
   - Percentage breakdowns
   - 4 quality categories

5. **kpis.csv** - Key Performance Indicators
   - Overall metrics
   - Compliance rates
   - Critical thresholds

6. **time_series_data.csv** - Time-based analysis
   - Daily aggregates
   - Moving averages
   - Trend data

7. **correlation_matrix.csv** - Parameter relationships
   - Correlation coefficients
   - Long format for heatmap visualization

## Dashboard Pages

### 1. Executive Summary
High-level overview with:
- Total samples and locations
- WHO compliance rate
- Average DO levels
- Quality distribution donut chart
- Trend analysis

### 2. Water Quality Analysis
Detailed parameter analysis:
- DO gauge charts
- Location comparisons
- Temperature-DO relationship
- Correlation matrix heatmap
- Summary statistics table

### 3. Temporal Trends
Time-based insights:
- DO trends with moving averages
- Temperature trends
- Monthly patterns
- Seasonal variations
- Quality category ribbons

### 4. Pollution Analysis
Pollution indicators:
- Severity by location
- Turbidity vs TSS scatter
- EC and TDS trends
- Quality funnel

### 5. Compliance Dashboard
WHO standards tracking:
- Compliance KPIs
- Decomposition tree
- Location-based compliance
- Detailed breakdown table

### 6. Drill-Down Details
Interactive exploration:
- Full data table
- Slicers for filtering
- Export capabilities

## Key Metrics

### Water Quality Status
- **Total Samples:** {kpis['Total_Samples']}
- **Average DO:** {kpis['Average_DO_mg_L']:.2f} mg/L
- **Severely Hypoxic:** {kpis['Severely_Hypoxic_Pct']:.1f}%
- **WHO Compliance:** {kpis['WHO_Compliance_Rate_Pct']:.1f}%

### Critical Findings
- {kpis['Critical_Quality_Pct']:.1f}% of samples in critical quality range
- {kpis['High_Pollution_Pct']:.1f}% samples show high pollution levels
- DO range: {kpis['Min_DO_mg_L']:.2f} - {kpis['Max_DO_mg_L']:.2f} mg/L

## Color Coding

### Water Quality Categories
- 🟢 **Adequate** (DO > 6.0 mg/L) - Green (#00B050)
- 🟡 **Low** (DO 4.0-6.0 mg/L) - Yellow (#FFC000)
- 🟠 **Hypoxic** (DO 2.0-4.0 mg/L) - Orange (#FF6600)
- 🔴 **Severely Hypoxic** (DO < 2.0 mg/L) - Red (#C00000)

### Overall Quality
- 🟢 **Good** - Adequate DO levels
- 🟡 **Fair** - Low DO levels
- 🟠 **Poor** - Hypoxic conditions
- 🔴 **Critical** - Severely hypoxic conditions

## Setup Instructions

### 1. Import Data
1. Open Power BI Desktop
2. Get Data → Text/CSV
3. Import all 7 CSV files from the `powerbi_data` folder
4. Ensure data types are correct (Date, Decimal, Boolean)

### 2. Create Relationships
- No relationships needed (single fact table architecture)
- All aggregations pre-calculated

### 3. Create Measures
Copy DAX formulas from `dashboard_specification.json`:
- Total Samples
- Average DO
- Compliance Rate
- Critical Samples
- WHO Target (constant)

### 4. Build Visuals
Follow the specification in `dashboard_specification.json` for each page

### 5. Apply Theme
Use the color theme specified:
- Primary: #0078D4
- Danger: #C00000
- Success: #00B050
- Warning: #FF6600

### 6. Configure Interactions
- Enable cross-filtering between visuals
- Set up drill-through pages
- Create bookmarks for key views

## Usage Tips

### Filtering
- Use slicers to filter by date range, location, or quality category
- Click on any visual element to cross-filter other visuals
- Right-click on data points for drill-through options

### Exporting
- Export visuals as images
- Export data tables to Excel
- Share dashboard via Power BI Service

### Refreshing Data
- Update CSV files with new data
- Refresh dataset in Power BI
- All visuals will update automatically

## Technical Specifications

### Data Refresh
- **Frequency:** Manual (update CSV files)
- **Last Refresh:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Data Period:** {dashboard_data['Date'].min().strftime('%Y-%m-%d')} to {dashboard_data['Date'].max().strftime('%Y-%m-%d')}

### Performance
- Dataset size: ~{len(dashboard_data)} rows
- Expected load time: < 5 seconds
- Recommended: Power BI Desktop version 2.0+

## Contact & Support

**Project Repository:** https://github.com/Olebogeng3/Unsupervised-Learning-Project  
**Data Source:** River Water Quality Monitoring Program  
**Created:** {datetime.now().strftime('%B %d, %Y')}

## Version History

- **v1.0** ({datetime.now().strftime('%Y-%m-%d')}) - Initial dashboard creation
  - 6 dashboard pages
  - 7 data tables
  - 18 KPI metrics
  - Full interactivity
"""

with open('powerbi_data/README_POWER_BI.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("✓ Saved: powerbi_data/README_POWER_BI.md")

# ============================================================================
# 9. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("POWER BI DASHBOARD DATA PREPARATION COMPLETED!")
print("=" * 80)

print(f"\n📊 Dashboard Summary:")
print(f"  ✓ Main dataset: {len(dashboard_data)} samples")
print(f"  ✓ Locations: {dashboard_data['Sampling_Point'].nunique()}")
print(f"  ✓ Date range: {dashboard_data['Date'].min().strftime('%Y-%m-%d')} to {dashboard_data['Date'].max().strftime('%Y-%m-%d')}")
print(f"  ✓ Parameters: {len(corr_cols)}")

print(f"\n📁 Files Created (8 files):")
print("  ✓ main_dashboard_data.csv")
print("  ✓ location_summary.csv")
print("  ✓ monthly_summary.csv")
print("  ✓ quality_category_summary.csv")
print("  ✓ kpis.csv")
print("  ✓ time_series_data.csv")
print("  ✓ correlation_matrix.csv")
print("  ✓ dashboard_specification.json")

print(f"\n📋 Dashboard Specification:")
print(f"  ✓ Pages: 6")
print(f"  ✓ Visuals: ~30")
print(f"  ✓ KPIs: 18")
print(f"  ✓ DAX Measures: 5")

print(f"\n🎨 Color Theme: Defined")
print(f"🔄 Interactions: Cross-filtering enabled")
print(f"📖 Documentation: README_POWER_BI.md created")

print("\n" + "=" * 80)
print("Next Steps:")
print("1. Open Power BI Desktop")
print("2. Import CSV files from powerbi_data/")
print("3. Follow dashboard_specification.json")
print("4. Apply color theme and interactions")
print("5. Publish to Power BI Service")
print("=" * 80)
