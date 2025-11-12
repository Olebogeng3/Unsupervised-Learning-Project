# River Water Quality Analysis - Unsupervised Learning Project

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌊 Project Overview

Comprehensive unsupervised learning analysis of river water quality data from 5 sampling locations across a 203-day monitoring period (May-November 2023). This project applies advanced machine learning techniques to identify water quality patterns, detect pollution events, and classify sampling locations based on environmental parameters.

### Key Achievements
- ✅ **219 samples analyzed** across 5 locations
- ✅ **93 engineered features** from 16 original parameters
- ✅ **5 unsupervised algorithms** applied successfully
- ✅ **3 distinct water quality clusters** identified
- ✅ **22 pollution anomalies** detected
- ✅ **38 visualizations** generated at 300 DPI

---

## 📊 Table of Contents
* [1. Dataset](#dataset)
* [2. Analysis Pipeline](#analysis-pipeline)
* [3. Key Findings](#key-findings)
* [4. Project Structure](#project-structure)
* [5. Installation](#installation)
* [6. Usage](#usage)
* [7. Results](#results)
* [8. Documentation](#documentation)
* [9. Technologies](#technologies)

---

## 📂 Dataset <a id="dataset"></a>

### Source
**River Water Parameters Dataset**
- 219 water samples
- 5 sampling locations
- 16 water quality parameters
- Time period: May 9 - November 28, 2023

### Sampling Locations
1. **Arroyo Las Torres** (Best quality - 2.3% anomalies)
2. **Puente Irigoyen** (Excellent - 0% anomalies)
3. **Arroyo Salguero** (Good - 11.9% anomalies)
4. **Puente Falbo** (Moderate - 9.1% anomalies)
5. **Puente Bilbao** (Critical - 26.7% anomalies) 🔴

### Parameters Measured
- **Physical**: pH, Turbidity, Temperature, Level
- **Chemical**: EC (Electrical Conductivity), TDS, TSS, DO (Dissolved Oxygen)
- **Mineral**: Hardness, Total Chlorine
- **Environmental**: Ambient Temperature, Humidity
- **Temporal**: Date, Time

---

## 🔬 Analysis Pipeline <a id="analysis-pipeline"></a>

### Phase 1: Data Preprocessing
**Script**: `river_water_preprocessing.py`
- Data cleaning and validation
- Missing value imputation (median/mode)
- Outlier detection (IQR method)
- Feature scaling (Standard, MinMax, Robust)

**Output**: `river_water_preprocessed.csv` (219 × 30)

### Phase 2: Date Cleaning
**Script**: `date_cleaning_enhanced.py`
- Date validation and format conversion
- Temporal feature extraction
- Gap analysis and frequency assessment
- Cyclical encoding (sin/cos)

**Output**: `date_cleaning_report.csv`

### Phase 3: Feature Engineering
**Script**: `advanced_feature_engineering.py`
- 63 new features created
- 6 feature categories:
  * Water Quality Indices (WQI, DO saturation, pH quality)
  * Physical-Chemical Relationships
  * Temporal Features (cyclical encoding)
  * Environmental Interactions
  * Anomaly Detection Features
  * Statistical Transformations

**Output**: 
- `river_water_features_engineered.csv` (219 × 93)
- `ml_features_unscaled.csv` (27 ML features)
- 3 scaled versions (Standard, MinMax, Robust)

### Phase 4: Statistical Analysis
**Script**: `statistical_analysis.py`
- Descriptive statistics with CV, skewness, kurtosis
- Normality testing (Shapiro-Wilk)
- Correlation analysis (Pearson/Spearman)
- Hypothesis testing (Kruskal-Wallis)
- Water Quality Index calculation

**Output**: 10 CSV files + 5 visualizations in `statistical_results/`

### Phase 5: Visualization & Reporting
**Script**: `visualization_reporting.py`
- Executive dashboard (6 panels)
- Parameter comparison (violin plots)
- Temporal analysis (time-series)
- Location comparison (radar charts)
- Pollution events tracking
- Interactive HTML report

**Output**: 6 PNG files + HTML report in `visualizations_reports/`

### Phase 6: Unsupervised Learning
**Script**: `unsupervised_learning_analysis.py`

#### Algorithms Applied:

**1. K-Means Clustering**
- Optimal K: 3 clusters
- Silhouette Score: 0.659 (Good)
- Davies-Bouldin: 0.609
- Calinski-Harabasz: 109.43

**2. Hierarchical Clustering**
- Ward linkage method
- Silhouette Score: 0.646
- Validates K-Means structure

**3. DBSCAN**
- Density-based clustering
- Optimal eps: 1.5
- 2 dense clusters + noise detection

**4. PCA (Dimensionality Reduction)**
- 7 components → 90% variance
- PC1 (41.9%): Mineralization
- PC2 (21.1%): Pollution events
- PC3 (8.4%): Temporal patterns

**5. Isolation Forest**
- Anomaly detection
- 22 anomalies identified (10%)
- Concentrated at Puente Bilbao

**Output**: 6 visualizations + 4 CSV files in `unsupervised_results/`

---

## 🎯 Key Findings <a id="key-findings"></a>

### Water Quality Classification

| Cluster | Samples | % | Avg WQI | Interpretation |
|---------|---------|---|---------|----------------|
| **Cluster 0** | 206 | 94.1% | 74.81 | 🟢 Good Quality |
| **Cluster 1** | 12 | 5.5% | 47.53 | 🔴 Poor Quality |
| **Cluster 2** | 1 | 0.5% | 68.34 | 🟡 Outlier |

### Critical Insights

🔴 **URGENT: Puente Bilbao requires immediate intervention**
- 26.7% anomaly rate (2.7× expected)
- Dominates poor quality cluster
- Consistent pollution pattern

📊 **Variance Breakdown**
- 41.9% from mineralization (EC, TDS, Hardness)
- 21.1% from pollution events (Turbidity, Risk)
- 8.4% from temporal patterns (seasonal)

⚠️ **Temporal Risk Pattern**
- Peak anomaly period: August-November
- Warmer months correlate with pollution

🏆 **Location Rankings**
1. Arroyo Las Torres (WQI: ~80)
2. Puente Irigoyen (WQI: ~75)
3. Arroyo Salguero (WQI: ~72)
4. Puente Falbo (WQI: ~65)
5. Puente Bilbao (WQI: ~48) ⚠️

---

## 📁 Project Structure <a id="project-structure"></a>

```
unsupervised-learning-project/
│
├── 📊 Data Processing Scripts
│   ├── river_water_preprocessing.py          # Data cleaning pipeline
│   ├── date_cleaning_enhanced.py             # Date validation
│   ├── advanced_feature_engineering.py       # Feature creation
│   └── statistical_analysis.py               # Statistical tests
│
├── 🤖 Machine Learning Scripts
│   ├── unsupervised_learning_analysis.py     # 5 algorithms
│   ├── visualization_reporting.py            # Executive dashboards
│   ├── verify_features.py                    # Feature verification
│   └── verify_clustering.py                  # Cluster validation
│
├── 📈 Results & Outputs
│   ├── statistical_results/                  # Statistical analysis outputs
│   │   ├── *.png (5 visualizations)
│   │   ├── *.csv (10 data files)
│   │   └── STATISTICAL_SUMMARY_REPORT.txt
│   │
│   ├── visualizations_reports/               # Executive visualizations
│   │   ├── executive_dashboard.png
│   │   ├── parameter_comparison_detailed.png
│   │   ├── temporal_analysis_dashboard.png
│   │   ├── location_comparison_report.png
│   │   ├── pollution_events_analysis.png
│   │   ├── statistical_summary_visualization.png
│   │   └── water_quality_report.html
│   │
│   └── unsupervised_results/                 # Clustering outputs
│       ├── 01_optimal_k_selection.png
│       ├── 02_hierarchical_dendrogram.png
│       ├── 03_pca_variance_explained.png
│       ├── 04_clustering_comparison_pca.png
│       ├── 05_kmeans_cluster_profiles.png
│       ├── 06_location_analysis.png
│       ├── clustered_data_complete.csv (219 × 101)
│       ├── kmeans_cluster_profiles.csv
│       ├── pca_top_features.csv
│       ├── clustering_metrics_summary.csv
│       └── ANALYSIS_SUMMARY_REPORT.txt
│
├── 📚 Documentation
│   ├── README.md                             # This file
│   ├── PREPROCESSING_REPORT.md               # Data cleaning guide
│   ├── FEATURE_ENGINEERING_REPORT.md         # Feature documentation
│   ├── STATISTICAL_ANALYSIS_DETAILED_REPORT.md
│   ├── VISUALIZATION_REPORTING_GUIDE.md
│   └── UNSUPERVISED_LEARNING_GUIDE.md        # ML analysis guide
│
├── 🔧 Configuration
│   ├── requirements.txt                      # Python dependencies
│   ├── .gitignore                            # Git exclusions
│   ├── feature_engineering_metadata.json
│   └── preprocessing_metadata.json
│
└── 📊 Processed Data (not in git)
    ├── river_water_preprocessed.csv
    ├── river_water_features_engineered.csv
    ├── ml_features_*.csv (4 versions)
    └── date_cleaning_report.csv
```

---

## 🚀 Installation <a id="installation"></a>

### Prerequisites
- Python 3.13.2 or higher
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Olebogeng3/Unsupervised-Learning-Project.git
cd Unsupervised-Learning-Project
```

2. **Create virtual environment**
```bash
python -m venv .venv
```

3. **Activate virtual environment**
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
- pandas 2.2.2
- numpy 1.26.4
- matplotlib 3.8.4
- seaborn 0.13.2
- scikit-learn 1.5.0
- scipy 1.13.0

---

## 💻 Usage <a id="usage"></a>

### Complete Analysis Pipeline

Run scripts in order:

```bash
# 1. Data Preprocessing
python river_water_preprocessing.py

# 2. Date Cleaning (optional)
python date_cleaning_enhanced.py

# 3. Feature Engineering
python advanced_feature_engineering.py

# 4. Statistical Analysis
python statistical_analysis.py

# 5. Visualization & Reporting
python visualization_reporting.py

# 6. Unsupervised Learning
python unsupervised_learning_analysis.py
```

### Quick Verification

```bash
# Verify features
python verify_features.py

# Verify clustering
python verify_clustering.py

# Summary
python feature_summary.py
```

### View Results

1. **Interactive Report**: Open `visualizations_reports/water_quality_report.html` in browser
2. **Visualizations**: Check `unsupervised_results/` and `visualizations_reports/`
3. **Data**: Review CSV files in each results directory

---

## 📊 Results <a id="results"></a>

### Clustering Performance

| Metric | K-Means | Hierarchical | DBSCAN |
|--------|---------|--------------|---------|
| **Silhouette Score** | 0.659 ✅ | 0.646 | 0.641 |
| **Davies-Bouldin** | 0.609 ✅ | N/A | N/A |
| **Calinski-Harabasz** | 109.43 ✅ | N/A | N/A |
| **Clusters** | 3 | 3 | 2 |

### Dimensionality Reduction

- **PCA Components**: 7 (for 90% variance)
- **Total Variance**: 91.9%
- **Top Component**: PC1 - 41.9% (Mineralization)

### Anomaly Detection

- **Total Anomalies**: 22 (10.0%)
- **Most Affected**: Puente Bilbao (12 anomalies)
- **Peak Period**: August-November 2023

---

## 📖 Documentation <a id="documentation"></a>

### Comprehensive Guides

1. **PREPROCESSING_REPORT.md**
   - Data cleaning methodology
   - Missing value strategies
   - Outlier handling

2. **FEATURE_ENGINEERING_REPORT.md**
   - 93 features explained
   - 27 ML features detailed
   - Scaling methods comparison

3. **STATISTICAL_ANALYSIS_DETAILED_REPORT.md**
   - Hypothesis testing results
   - Correlation analysis
   - WQI calculation methodology

4. **VISUALIZATION_REPORTING_GUIDE.md**
   - All visualizations explained
   - Stakeholder communication guide
   - Design choices rationale

5. **UNSUPERVISED_LEARNING_GUIDE.md** (12 sections)
   - Algorithm explanations
   - Cluster interpretations
   - Performance metrics
   - Recommendations

### Quick Reference

- `ANALYSIS_SUMMARY_REPORT.txt` - Console-friendly summary
- `STATISTICAL_SUMMARY_REPORT.txt` - Statistical test results
- `feature_engineering_metadata.json` - Feature catalog

---

## 🛠️ Technologies <a id="technologies"></a>

### Core Libraries
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: ML algorithms (K-Means, PCA, Isolation Forest)
- **SciPy**: Statistical testing, hierarchical clustering
- **Matplotlib & Seaborn**: Visualization

### Algorithms
- K-Means Clustering
- Hierarchical Clustering (Ward linkage)
- DBSCAN (Density-based)
- PCA (Principal Component Analysis)
- Isolation Forest (Anomaly detection)

### Scaling Methods
- StandardScaler (Z-score)
- MinMaxScaler (0-1 range)
- RobustScaler (median-IQR, recommended)

---

## 👥 Author

**Olebogeng Mohuba**
- GitHub: [@Olebogeng3](https://github.com/Olebogeng3)
- Repository: [Unsupervised-Learning-Project](https://github.com/Olebogeng3/Unsupervised-Learning-Project)

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Dataset: River water quality monitoring data (May-November 2023)
- Analysis Period: November 2025
- Total Lines of Code: 6,598+
- Visualizations Created: 38
- Documentation Pages: 5 comprehensive reports

---

## 📧 Contact

For questions, suggestions, or collaboration:
- Open an issue on GitHub
- Repository: https://github.com/Olebogeng3/Unsupervised-Learning-Project

---

**Status**: ✅ Analysis Complete | 🚀 Ready for Production | 📊 Fully Documented

**Last Updated**: November 12, 2025
