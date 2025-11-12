# River Water Quality - Data Preprocessing Report

## Executive Summary

Successfully preprocessed river water quality dataset for unsupervised learning analysis. The dataset contains **219 water quality measurements** from **5 sampling points** collected over **7 months** (May - November 2023).

---

## 1. Problem Identification

### **Business Problem**
- Monitor and analyze river water quality across multiple sampling locations
- Identify pollution patterns and water quality clusters
- Detect anomalous water quality events requiring investigation
- Optimize environmental monitoring resources

### **Technical Challenges**
- **Missing Data**: 17.81% missing in Level measurements, smaller gaps in other parameters
- **Outliers**: 11.42% outliers in Turbidity (may indicate pollution events)
- **Multi-parameter Complexity**: 15+ water quality parameters to analyze simultaneously
- **Temporal & Spatial Variation**: Data varies across time and location

---

## 2. Data Preprocessing Steps

### **2.1 Data Loading & Exploration**
- **Dataset**: 219 rows × 16 columns
- **Sampling Points**: 5 locations (Puente Bilbao, Arroyo Las Torres, Puente Irigoyen, Puente Falbo, Arroyo Salguero)
- **Time Period**: May 9, 2023 - November 28, 2023 (7 months)
- **Parameters Measured**: pH, EC, TDS, TSS, DO, Turbidity, Hardness, Chlorine, Temperature, Humidity, Water Level

### **2.2 Column Standardization**
Renamed columns for easier handling:
- `Date (DD/MM/YYYY)` → `Date`
- `EC (µS/cm)` → `EC`
- `TDS (mg/L)` → `TDS`
- `Total Cl- (mg Cl-/L)` → `Total_Chlorine`
- etc.

### **2.3 Missing Value Handling**

| Column | Missing Count | Percentage | Strategy |
|--------|--------------|------------|----------|
| Level | 39 | 17.81% | Median imputation |
| TSS | 6 | 2.74% | Median imputation |
| Total_Chlorine | 6 | 2.74% | Median imputation |
| Hardness | 2 | 0.91% | Median imputation |
| Hardness_Class | 2 | 0.91% | Mode imputation |
| Turbidity | 1 | 0.46% | Median imputation |

**Rationale**: Median imputation chosen for numeric data as it's robust to outliers (important for water quality data where outliers may represent real pollution events).

### **2.4 Outlier Detection (IQR Method)**

| Parameter | Outlier Count | Percentage | Action |
|-----------|--------------|------------|--------|
| Turbidity | 25 | 11.42% | Retained |
| pH | 11 | 5.02% | Retained |
| Ambient_Temp | 10 | 4.57% | Retained |
| TSS | 10 | 4.57% | Retained |
| EC | 2 | 0.91% | Retained |
| TDS | 2 | 0.91% | Retained |
| Level | 2 | 0.91% | Retained |
| DO | 1 | 0.46% | Retained |

**Decision**: All outliers retained as they likely represent genuine pollution events or extreme weather conditions that are valuable for analysis.

### **2.5 Feature Engineering**

Created **10 new features** to enhance analysis:

#### **Temporal Features**
1. `Year` - Extracted from date
2. `Month` - Extracted from date (5-11)
3. `Day` - Day of month
4. `DayOfWeek` - 0=Monday, 6=Sunday
5. `Season` - Spring, Summer, Autumn
6. `Time_Minutes` - Minutes since midnight
7. `Season_Encoded` - Numeric encoding (0-2)

#### **Environmental Features**
8. `Temp_Difference` = Sample_Temp - Ambient_Temp (indicates water heating/cooling)

#### **Water Quality Indices**
9. `DO_Index` - Dissolved Oxygen quality index (0-100 scale)
   - ≥6 mg/L: 100 (Excellent)
   - ≥4 mg/L: 80 (Good)
   - ≥2 mg/L: 60 (Fair)
   - <2 mg/L: 40 (Poor)

10. `pH_Deviation` = |pH - 7.0| (deviation from neutral)

11. `EC_TDS_Ratio` = EC / TDS (electrical conductivity to dissolved solids ratio)

12. `Pollution_Score` - Composite pollution indicator combining:
    - Turbidity (30% weight)
    - DO inverted (40% weight - low DO indicates pollution)
    - pH deviation (30% weight)

#### **Categorical Encodings**
13. `Sampling_Point_Encoded` (0-4 for 5 locations)
14. `Hardness_Class_Encoded` (0=BLANDA, 1=SEMIDURA)

### **2.6 Feature Matrix Creation**

**Selected 20 features for unsupervised learning**:

**Water Quality Parameters (8)**:
- pH, EC, TDS, TSS, DO, Turbidity, Hardness, Total_Chlorine

**Environmental Factors (3)**:
- Ambient_Temp, Ambient_Humidity, Sample_Temp

**Engineered Features (6)**:
- Temp_Difference, DO_Index, pH_Deviation, EC_TDS_Ratio, Pollution_Score, Sampling_Point_Encoded

**Temporal Features (3)**:
- Month, Season_Encoded, Time_Minutes

### **2.7 Feature Scaling**

Applied **StandardScaler** (z-score normalization):
- Mean = 0
- Standard Deviation = 1
- Ensures all features contribute equally to distance-based algorithms (K-Means, DBSCAN)

---

## 3. Key Findings from Exploratory Analysis

### **3.1 Sampling Point Characteristics**

| Location | Avg pH | Avg DO | Avg Turbidity | Avg Pollution Score | Water Quality |
|----------|--------|--------|---------------|---------------------|---------------|
| Arroyo Salguero | 7.98 | 3.39 | 13.68 | 56.22 | **Best** |
| Arroyo Las Torres | 8.22 | 3.99 | 133.24 | 64.66 | Moderate |
| Puente Falbo | 7.96 | 1.51 | 117.41 | 66.41 | Poor |
| Puente Irigoyen | 8.06 | 1.85 | 80.88 | 66.92 | Poor |
| Puente Bilbao | 7.93 | 2.41 | 366.61 | 69.30 | **Worst** |

**Key Insights**:
- **Puente Bilbao** shows highest pollution (high turbidity, low DO)
- **Arroyo Salguero** has best water quality (low turbidity, higher DO)
- All locations show slightly alkaline pH (7.93-8.22)

### **3.2 Temporal Patterns**
- **Coverage**: 7 months (May - November 2023)
- **Seasons**: Spring (May), Summer (June-August), Autumn (September-November)
- **Measurements**: Regular biweekly/weekly sampling

### **3.3 Correlation Insights**
Strong correlations expected between:
- EC ↔ TDS (electrical conductivity and dissolved solids)
- Sample_Temp ↔ Ambient_Temp (environmental conditions)
- Turbidity ↔ Pollution_Score (by design)

---

## 4. Output Files Generated

### **Primary Datasets**
1. **river_water_preprocessed.csv** (219 rows × 30 columns)
   - Full dataset with all original + engineered features
   - Includes date/time, categorical variables, all encodings

2. **river_water_features.csv** (219 rows × 20 columns)
   - **Unscaled** feature matrix
   - Ready for tree-based models or visualization
   - Original scale preserved

3. **river_water_features_scaled.csv** (219 rows × 20 columns)
   - **Standardized** feature matrix (mean=0, std=1)
   - **USE THIS for**: K-Means, DBSCAN, Hierarchical Clustering, PCA
   - Distance-based algorithms require scaling

### **Metadata**
4. **preprocessing_metadata.json**
   - Sampling point encodings
   - Hardness class mappings
   - Season encodings
   - Feature column names
   - Dataset dimensions

### **Visualizations**
5. **correlation_matrix.png** - Heatmap showing parameter correlations
6. **parameters_by_location.png** - Boxplots of key parameters by sampling point
7. **temporal_trends.png** - Time series of pH, DO, Turbidity, Pollution Score

---

## 5. Ready for Unsupervised Learning

### **Recommended Analyses**

#### **5.1 Clustering Analysis**
**Objective**: Group similar water quality profiles

**Methods to Try**:
- **K-Means**: Find k distinct water quality clusters
  - Recommended: k=3-5 clusters
  - Use: Elbow method, Silhouette score for optimal k
  
- **DBSCAN**: Density-based clustering for outlier detection
  - Useful for identifying pollution hotspots
  - Can find irregular-shaped clusters
  
- **Hierarchical Clustering**: Understand relationships between sampling points
  - Use dendrogram for visualization
  - Agglomerative approach recommended

**Expected Outcomes**:
- Cluster 1: Clean water profiles (low turbidity, high DO)
- Cluster 2: Moderate pollution (medium metrics)
- Cluster 3: High pollution events (high turbidity, low DO)
- Possibly location-based clusters or temporal patterns

#### **5.2 Dimensionality Reduction**
**Objective**: Visualize 20-dimensional data in 2D/3D

**Methods to Try**:
- **PCA**: Principal Component Analysis
  - Identify which parameters explain most variance
  - Typically 2-3 components capture 70-80% variance
  
- **t-SNE**: Better for visualization, preserves local structure
  - Great for identifying distinct clusters visually
  - Perplexity: 30-50 recommended
  
- **UMAP**: Fast, preserves both local and global structure
  - Alternative to t-SNE with better scalability

#### **5.3 Anomaly Detection**
**Objective**: Identify unusual water quality events

**Methods to Try**:
- **Isolation Forest**: Effective for multivariate outlier detection
  - Contamination parameter: 0.05-0.15 (5-15% outliers)
  
- **Local Outlier Factor (LOF)**: Density-based anomaly detection
  - Good for identifying localized pollution events
  
- **One-Class SVM**: Robust outlier detection

**Expected Anomalies**:
- Extreme turbidity events (>1000 NTU)
- Very low DO readings (<0.5 mg/L)
- pH extremes (<7 or >8.5)

---

## 6. Data Quality Assessment

### **Strengths** ✓
- Comprehensive 15-parameter monitoring
- Multiple sampling locations for spatial analysis
- Regular temporal sampling (7 months)
- All critical missing values handled
- Feature engineering enhances analytical power

### **Limitations** ⚠
- Moderate sample size (n=219)
- 17.81% missing in Level measurements
- Only 7 months of data (seasonal coverage incomplete)
- Duplicate measurements at same time/location (quality control?)

### **Recommendations**
1. Collect additional data to increase sample size
2. Investigate cause of Level measurement gaps
3. Extend monitoring to cover full annual cycle
4. Consider additional parameters (biological oxygen demand, nutrients)

---

## 7. Next Steps

1. **Exploratory Clustering**
   - Start with K-Means (k=3,4,5)
   - Evaluate Silhouette scores
   - Profile each cluster

2. **PCA Analysis**
   - Determine variance explained
   - Identify key contributing parameters
   - Visualize in 2D/3D

3. **Anomaly Detection**
   - Apply Isolation Forest
   - Flag suspicious measurements for investigation
   - Cross-reference with rainfall/industrial events

4. **Reporting**
   - Create water quality dashboard
   - Identify high-risk locations
   - Recommend targeted monitoring

---

## 8. Technical Details

### **Environment**
- Python 3.13.2
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

### **Preprocessing Script**
- File: `river_water_preprocessing.py`
- Runtime: ~10 seconds
- Fully reproducible

### **Data Integrity**
- No duplicates found
- All missing values imputed
- Outliers retained (domain knowledge)
- Scaling applied appropriately

---

## Contact & Support

For questions about the preprocessing:
1. Review `river_water_preprocessing.py` for implementation details
2. Check `preprocessing_metadata.json` for encodings
3. Examine visualizations in `visualizations/` folder

**Ready to proceed with unsupervised learning analysis!** 🚀
