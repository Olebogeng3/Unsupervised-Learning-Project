# Power BI Dashboard - River Water Quality Monitoring

## Overview
This Power BI dashboard provides comprehensive visualization and analysis of river water quality data collected from 5 sampling locations between May 09, 2023 and November 28, 2023.

## Data Files

### Main Datasets
1. **main_dashboard_data.csv** - Primary dataset with all measurements
   - 219 samples
   - 39 columns
   - Includes temporal, spatial, and quality categorizations

2. **location_summary.csv** - Aggregated statistics by sampling location
   - Summary statistics (mean, min, max, std)
   - Compliance rates
   - 5 locations

3. **monthly_summary.csv** - Temporal aggregations
   - Monthly trends
   - Seasonal patterns
   - 7 monthly records

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
- **Total Samples:** 219
- **Average DO:** 2.62 mg/L
- **Severely Hypoxic:** 55.7%
- **WHO Compliance:** 0.5%

### Critical Findings
- 55.7% of samples in critical quality range
- 20.5% samples show high pollution levels
- DO range: 0.00 - 9.12 mg/L

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
- **Last Refresh:** 2025-11-12 20:23:36
- **Data Period:** 2023-05-09 to 2023-11-28

### Performance
- Dataset size: ~219 rows
- Expected load time: < 5 seconds
- Recommended: Power BI Desktop version 2.0+

## Contact & Support

**Project Repository:** https://github.com/Olebogeng3/Unsupervised-Learning-Project  
**Data Source:** River Water Quality Monitoring Program  
**Created:** November 12, 2025

## Version History

- **v1.0** (2025-11-12) - Initial dashboard creation
  - 6 dashboard pages
  - 7 data tables
  - 18 KPI metrics
  - Full interactivity
