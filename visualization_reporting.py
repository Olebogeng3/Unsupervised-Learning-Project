"""
Data Visualization and Reporting - River Water Quality
Comprehensive visualizations for stakeholder communication
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set enhanced visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

print("=" * 80)
print("DATA VISUALIZATION & REPORTING - RIVER WATER QUALITY")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")
try:
    df = pd.read_csv('river_water_with_statistics.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"✓ Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
except FileNotFoundError:
    print("✗ Error: Please run statistical_analysis.py first!")
    exit()

# Create output directory
import os
viz_dir = 'visualizations_reports'
os.makedirs(viz_dir, exist_ok=True)

# Define color schemes
location_colors = {
    'Puente Bilbao': '#e74c3c',      # Red (worst quality)
    'Arroyo_Las Torres': '#f39c12',   # Orange
    'Puente Irigoyen': '#f1c40f',     # Yellow
    'Puente Falbo': '#95a5a6',        # Gray
    'Arroyo Salguero': '#27ae60'      # Green (best quality)
}

# ============================================================================
# 2. EXECUTIVE DASHBOARD
# ============================================================================
print("\n[2] CREATING EXECUTIVE DASHBOARD...")

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('River Water Quality Monitoring - Executive Dashboard', 
             fontsize=20, fontweight='bold', y=0.98)

# Subplot 1: Water Quality Index by Location (Bar Chart)
ax1 = fig.add_subplot(gs[0, 0:2])
wqi_by_loc = df.groupby('Sampling_Point')['WQI'].mean().sort_values(ascending=False)
bars = ax1.barh(wqi_by_loc.index, wqi_by_loc.values, 
                color=[location_colors[loc] for loc in wqi_by_loc.index])
ax1.set_xlabel('Water Quality Index (0-100)')
ax1.set_title('Average Water Quality Index by Location', fontweight='bold')
ax1.axvline(x=70, color='green', linestyle='--', alpha=0.5, label='Good Quality (70)')
ax1.axvline(x=50, color='orange', linestyle='--', alpha=0.5, label='Moderate Quality (50)')
ax1.legend()
for i, (loc, val) in enumerate(wqi_by_loc.items()):
    ax1.text(val + 1, i, f'{val:.1f}', va='center', fontweight='bold')

# Subplot 2: Pollution Score by Location (Bar Chart)
ax2 = fig.add_subplot(gs[0, 2:4])
pollution_by_loc = df.groupby('Sampling_Point')['Pollution_Score'].mean().sort_values()
bars = ax2.barh(pollution_by_loc.index, pollution_by_loc.values,
                color=[location_colors[loc] for loc in pollution_by_loc.index])
ax2.set_xlabel('Pollution Score (Higher = More Polluted)')
ax2.set_title('Average Pollution Score by Location', fontweight='bold')
for i, (loc, val) in enumerate(pollution_by_loc.items()):
    ax2.text(val + 0.5, i, f'{val:.1f}', va='center', fontweight='bold')

# Subplot 3: WQI Distribution (Pie Chart)
ax3 = fig.add_subplot(gs[1, 0:2])
wqi_dist = df['WQI_Category'].value_counts()
colors_pie = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
explode = [0.05 if cat in ['Excellent', 'Poor'] else 0 for cat in wqi_dist.index]
ax3.pie(wqi_dist.values, labels=wqi_dist.index, autopct='%1.1f%%',
        colors=colors_pie[:len(wqi_dist)], explode=explode, startangle=90)
ax3.set_title('Overall Water Quality Distribution', fontweight='bold')

# Subplot 4: Pollution Level Distribution (Stacked Bar)
ax4 = fig.add_subplot(gs[1, 2:4])
pollution_dist = pd.crosstab(df['Sampling_Point'], df['Pollution_Level'])
pollution_dist = pollution_dist.reindex(columns=['Low', 'Moderate', 'High', 'Critical'], fill_value=0)
pollution_dist.plot(kind='barh', stacked=True, ax=ax4, 
                   color=['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad'])
ax4.set_xlabel('Number of Samples')
ax4.set_title('Pollution Level Distribution by Location', fontweight='bold')
ax4.legend(title='Pollution Level', bbox_to_anchor=(1.05, 1), loc='upper left')

# Subplot 5: Key Parameters Over Time
ax5 = fig.add_subplot(gs[2, :2])
for param in ['pH', 'DO']:
    monthly_avg = df.groupby(df['Date'].dt.to_period('M'))[param].mean()
    ax5.plot(monthly_avg.index.to_timestamp(), monthly_avg.values, 
            marker='o', label=param, linewidth=2)
ax5.set_xlabel('Date')
ax5.set_ylabel('Value')
ax5.set_title('Temporal Trends: pH and Dissolved Oxygen', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

# Subplot 6: Turbidity Over Time (with threshold)
ax6 = fig.add_subplot(gs[2, 2:])
for loc in df['Sampling_Point'].unique():
    loc_data = df[df['Sampling_Point'] == loc]
    ax6.scatter(loc_data['Date'], loc_data['Turbidity'], 
               label=loc, alpha=0.6, s=30, color=location_colors[loc])
ax6.axhline(y=100, color='red', linestyle='--', label='High Turbidity Threshold', linewidth=2)
ax6.set_xlabel('Date')
ax6.set_ylabel('Turbidity (NTU)')
ax6.set_title('Turbidity Over Time by Location', fontweight='bold')
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax6.set_yscale('log')
ax6.grid(True, alpha=0.3)
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

plt.savefig(f'{viz_dir}/executive_dashboard.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_dir}/executive_dashboard.png")
plt.close()

# ============================================================================
# 3. DETAILED PARAMETER COMPARISON
# ============================================================================
print("\n[3] CREATING DETAILED PARAMETER COMPARISON...")

params = ['pH', 'EC', 'TDS', 'TSS', 'DO', 'Turbidity', 'Hardness', 'Total_Chlorine']
fig, axes = plt.subplots(4, 2, figsize=(18, 20))
axes = axes.flatten()

for idx, param in enumerate(params):
    ax = axes[idx]
    
    # Violin plot with swarm overlay
    parts = ax.violinplot([df[df['Sampling_Point'] == loc][param].dropna() 
                           for loc in location_colors.keys()],
                          positions=range(len(location_colors)),
                          showmeans=True, showmedians=True, widths=0.7)
    
    # Color the violin plots
    for pc, loc in zip(parts['bodies'], location_colors.keys()):
        pc.set_facecolor(location_colors[loc])
        pc.set_alpha(0.6)
    
    # Add scatter points
    for i, loc in enumerate(location_colors.keys()):
        y_data = df[df['Sampling_Point'] == loc][param].dropna()
        x_data = np.random.normal(i, 0.04, size=len(y_data))
        ax.scatter(x_data, y_data, alpha=0.3, s=20, color=location_colors[loc])
    
    ax.set_xticks(range(len(location_colors)))
    ax.set_xticklabels(location_colors.keys(), rotation=45, ha='right')
    ax.set_ylabel(param)
    ax.set_title(f'{param} Distribution by Location', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Detailed Water Quality Parameter Comparison', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{viz_dir}/parameter_comparison_detailed.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_dir}/parameter_comparison_detailed.png")
plt.close()

# ============================================================================
# 4. TEMPORAL ANALYSIS DASHBOARD
# ============================================================================
print("\n[4] CREATING TEMPORAL ANALYSIS DASHBOARD...")

fig, axes = plt.subplots(3, 2, figsize=(18, 14))

# 4.1: pH Over Time by Location
ax = axes[0, 0]
for loc in location_colors.keys():
    loc_data = df[df['Sampling_Point'] == loc].sort_values('Date')
    ax.plot(loc_data['Date'], loc_data['pH'], 
           label=loc, alpha=0.7, linewidth=2, color=location_colors[loc])
ax.axhline(y=7.0, color='gray', linestyle='--', alpha=0.5, label='Neutral pH')
ax.set_ylabel('pH')
ax.set_title('pH Temporal Trends', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# 4.2: Dissolved Oxygen Over Time
ax = axes[0, 1]
for loc in location_colors.keys():
    loc_data = df[df['Sampling_Point'] == loc].sort_values('Date')
    ax.plot(loc_data['Date'], loc_data['DO'], 
           label=loc, alpha=0.7, linewidth=2, color=location_colors[loc])
ax.axhline(y=6.0, color='green', linestyle='--', alpha=0.5, label='Healthy DO (6 mg/L)')
ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Hypoxic (2 mg/L)')
ax.set_ylabel('Dissolved Oxygen (mg/L)')
ax.set_title('Dissolved Oxygen Temporal Trends', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# 4.3: Monthly Averages Heatmap
ax = axes[1, 0]
monthly_pivot = df.pivot_table(values='WQI', 
                                index='Sampling_Point', 
                                columns=df['Date'].dt.month, 
                                aggfunc='mean')
sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
           ax=ax, cbar_kws={'label': 'WQI'}, vmin=40, vmax=90)
ax.set_xlabel('Month')
ax.set_ylabel('Sampling Point')
ax.set_title('Monthly Average WQI Heatmap', fontweight='bold')

# 4.4: Seasonal Box Plots
ax = axes[1, 1]
season_order = ['Spring', 'Summer', 'Autumn']
bp = df.boxplot(column='Pollution_Score', by='Season', ax=ax, 
               patch_artist=True, return_type='dict')
ax.set_xlabel('Season')
ax.set_ylabel('Pollution Score')
ax.set_title('Seasonal Pollution Score Variation', fontweight='bold')
plt.suptitle('')  # Remove default title

# 4.5: Turbidity Events Timeline
ax = axes[2, 0]
high_turbidity = df[df['Turbidity'] > 200].sort_values('Date')
for loc in location_colors.keys():
    loc_events = high_turbidity[high_turbidity['Sampling_Point'] == loc]
    ax.scatter(loc_events['Date'], loc_events['Turbidity'], 
              label=loc, s=100, alpha=0.7, color=location_colors[loc],
              edgecolors='black', linewidth=1)
ax.set_ylabel('Turbidity (NTU)')
ax.set_title('High Turbidity Events (>200 NTU)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# 4.6: DO vs Temperature Relationship
ax = axes[2, 1]
scatter = ax.scatter(df['Sample_Temp'], df['DO'], 
                    c=df['Sampling_Point'].map({loc: i for i, loc in enumerate(location_colors.keys())}),
                    cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Sample Temperature (°C)')
ax.set_ylabel('Dissolved Oxygen (mg/L)')
ax.set_title('DO vs Temperature Relationship', fontweight='bold')
ax.grid(True, alpha=0.3)
# Add trend line
z = np.polyfit(df['Sample_Temp'].dropna(), df['DO'].dropna(), 1)
p = np.poly1d(z)
ax.plot(df['Sample_Temp'].sort_values(), p(df['Sample_Temp'].sort_values()), 
       "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
ax.legend()

plt.suptitle('Temporal Analysis Dashboard', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{viz_dir}/temporal_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_dir}/temporal_analysis_dashboard.png")
plt.close()

# ============================================================================
# 5. LOCATION COMPARISON REPORT
# ============================================================================
print("\n[5] CREATING LOCATION COMPARISON REPORT...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 5.1: Radar Chart for each location
from math import pi

categories = ['pH', 'DO', 'Turbidity_inv', 'EC_inv', 'WQI']
num_vars = len(categories)

# Normalize data for radar chart
radar_data = {}
for loc in location_colors.keys():
    loc_df = df[df['Sampling_Point'] == loc]
    radar_data[loc] = [
        (loc_df['pH'].mean() - 7) / 1.7 * 100,  # Normalized to 0-100
        loc_df['DO'].mean() / 10 * 100,
        (1 - loc_df['Turbidity'].mean() / df['Turbidity'].max()) * 100,  # Inverted
        (1 - loc_df['EC'].mean() / df['EC'].max()) * 100,  # Inverted
        loc_df['WQI'].mean()
    ]

# Create radar chart
ax = axes[0, 0]
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

for loc, color in location_colors.items():
    values = radar_data[loc]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=loc, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 100)
ax.set_title('Water Quality Profile Comparison', fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax.grid(True)

# 5.2: Average Parameter Values
ax = axes[0, 1]
param_means = df.groupby('Sampling_Point')[['pH', 'DO', 'Hardness', 'Total_Chlorine']].mean()
param_means_norm = (param_means - param_means.min()) / (param_means.max() - param_means.min())
param_means_norm.T.plot(kind='bar', ax=ax, color=[location_colors[loc] for loc in param_means.index])
ax.set_ylabel('Normalized Value (0-1)')
ax.set_title('Normalized Parameter Comparison', fontweight='bold')
ax.legend(title='Location', fontsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# 5.3: Sample Count by Location
ax = axes[0, 2]
sample_counts = df['Sampling_Point'].value_counts()
bars = ax.bar(range(len(sample_counts)), sample_counts.values,
              color=[location_colors[loc] for loc in sample_counts.index])
ax.set_xticks(range(len(sample_counts)))
ax.set_xticklabels(sample_counts.index, rotation=45, ha='right')
ax.set_ylabel('Number of Samples')
ax.set_title('Sampling Frequency by Location', fontweight='bold')
for i, (loc, count) in enumerate(sample_counts.items()):
    ax.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 5.4: DO Distribution Comparison
ax = axes[1, 0]
do_bins = [0, 2, 4, 6, 8, 10]
do_labels = ['<2\n(Hypoxic)', '2-4\n(Low)', '4-6\n(Moderate)', '6-8\n(Good)', '>8\n(Excellent)']
for loc in location_colors.keys():
    loc_do = df[df['Sampling_Point'] == loc]['DO']
    do_hist, _ = np.histogram(loc_do, bins=do_bins)
    ax.plot(range(len(do_hist)), do_hist, marker='o', label=loc, 
           linewidth=2, color=location_colors[loc])
ax.set_xticks(range(len(do_labels)))
ax.set_xticklabels(do_labels)
ax.set_ylabel('Frequency')
ax.set_title('Dissolved Oxygen Distribution by Location', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 5.5: Turbidity Exceedances
ax = axes[1, 1]
turbidity_threshold = [50, 100, 200, 500]
exceedance_data = []
for thresh in turbidity_threshold:
    exceedances = df.groupby('Sampling_Point').apply(
        lambda x: (x['Turbidity'] > thresh).sum()
    )
    exceedance_data.append(exceedances)

exceedance_df = pd.DataFrame(exceedance_data, 
                             index=[f'>{t}' for t in turbidity_threshold]).T
exceedance_df.plot(kind='bar', ax=ax, 
                  color=['#3498db', '#f39c12', '#e74c3c', '#8e44ad'])
ax.set_ylabel('Number of Exceedances')
ax.set_title('Turbidity Threshold Exceedances', fontweight='bold')
ax.legend(title='Threshold (NTU)', fontsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# 5.6: WQI Trends Over Time
ax = axes[1, 2]
for loc in location_colors.keys():
    loc_data = df[df['Sampling_Point'] == loc].sort_values('Date')
    # Calculate rolling average
    loc_data['WQI_rolling'] = loc_data['WQI'].rolling(window=3, min_periods=1).mean()
    ax.plot(loc_data['Date'], loc_data['WQI_rolling'], 
           label=loc, linewidth=2.5, color=location_colors[loc])
ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Good (70)')
ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Moderate (50)')
ax.set_ylabel('Water Quality Index')
ax.set_title('WQI Trends (3-sample moving average)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.suptitle('Location Comparison Report', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{viz_dir}/location_comparison_report.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_dir}/location_comparison_report.png")
plt.close()

# ============================================================================
# 6. POLLUTION EVENTS ANALYSIS
# ============================================================================
print("\n[6] CREATING POLLUTION EVENTS ANALYSIS...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 6.1: Critical Events Timeline
ax = axes[0, 0]
critical_events = df[df['Pollution_Level'] == 'Critical'].sort_values('Date')
if len(critical_events) > 0:
    for i, (idx, event) in enumerate(critical_events.iterrows()):
        ax.scatter(event['Date'], event['Pollution_Score'], 
                  s=300, alpha=0.7, color=location_colors[event['Sampling_Point']],
                  edgecolors='red', linewidth=3)
        ax.text(event['Date'], event['Pollution_Score'] + 1, 
               f"{event['Sampling_Point'][:10]}\n{event['Date'].strftime('%m/%d')}",
               ha='center', fontsize=7)
ax.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Critical Threshold')
ax.set_ylabel('Pollution Score')
ax.set_title('Critical Pollution Events', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# 6.2: Multi-parameter pollution events
ax = axes[0, 1]
# Define pollution event criteria
pollution_events = df[
    (df['Turbidity'] > 200) | 
    (df['DO'] < 2) | 
    (df['TSS'] > 100)
]
event_types = []
for idx, row in pollution_events.iterrows():
    types = []
    if row['Turbidity'] > 200:
        types.append('High Turbidity')
    if row['DO'] < 2:
        types.append('Low DO')
    if row['TSS'] > 100:
        types.append('High TSS')
    event_types.extend(types)

from collections import Counter
event_counts = Counter(event_types)
ax.bar(event_counts.keys(), event_counts.values(), 
      color=['#e74c3c', '#3498db', '#f39c12'])
ax.set_ylabel('Number of Events')
ax.set_title('Pollution Event Types', fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
for i, (event, count) in enumerate(event_counts.items()):
    ax.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 6.3: Events by Location
ax = axes[1, 0]
events_by_loc = pollution_events.groupby('Sampling_Point').size()
bars = ax.barh(events_by_loc.index, events_by_loc.values,
              color=[location_colors[loc] for loc in events_by_loc.index])
ax.set_xlabel('Number of Pollution Events')
ax.set_title('Pollution Events by Location', fontweight='bold')
for i, (loc, count) in enumerate(events_by_loc.items()):
    ax.text(count + 0.5, i, str(count), va='center', fontweight='bold')

# 6.4: Anomaly Score Distribution
ax = axes[1, 1]
if 'Anomaly_Score' in df.columns:
    for loc in location_colors.keys():
        loc_data = df[df['Sampling_Point'] == loc]['Anomaly_Score']
        ax.hist(loc_data, alpha=0.5, label=loc, bins=30, 
               color=location_colors[loc])
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Anomaly Score Distribution', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
else:
    # If anomaly scores not available, show pollution score distribution
    for loc in location_colors.keys():
        loc_data = df[df['Sampling_Point'] == loc]['Pollution_Score']
        ax.hist(loc_data, alpha=0.5, label=loc, bins=20,
               color=location_colors[loc])
    ax.set_xlabel('Pollution Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Pollution Score Distribution', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Pollution Events Analysis', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{viz_dir}/pollution_events_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_dir}/pollution_events_analysis.png")
plt.close()

# ============================================================================
# 7. STATISTICAL SUMMARY VISUALIZATION
# ============================================================================
print("\n[7] CREATING STATISTICAL SUMMARY VISUALIZATION...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 7.1: Correlation Network
ax = axes[0, 0]
params_corr = ['pH', 'DO', 'Turbidity', 'EC', 'TSS']
corr_matrix = df[params_corr].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
           cmap='coolwarm', center=0, square=True, linewidths=2,
           cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
ax.set_title('Parameter Correlation Matrix', fontweight='bold')

# 7.2: Coefficient of Variation
ax = axes[0, 1]
params = ['pH', 'EC', 'TDS', 'TSS', 'DO', 'Turbidity', 'Hardness', 'Total_Chlorine']
cv_values = [(df[param].std() / df[param].mean() * 100) for param in params]
colors_cv = ['#27ae60' if cv < 25 else '#f39c12' if cv < 50 else '#e74c3c' 
            for cv in cv_values]
bars = ax.barh(params, cv_values, color=colors_cv)
ax.set_xlabel('Coefficient of Variation (%)')
ax.set_title('Parameter Variability (CV)', fontweight='bold')
ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='High Variability')
ax.axvline(x=25, color='orange', linestyle='--', alpha=0.5, label='Moderate Variability')
ax.legend()
for i, (param, cv) in enumerate(zip(params, cv_values)):
    ax.text(cv + 2, i, f'{cv:.1f}%', va='center', fontweight='bold')

# 7.3: Parameter Ranges by Location
ax = axes[1, 0]
param_display = 'DO'
ranges_data = []
for loc in location_colors.keys():
    loc_data = df[df['Sampling_Point'] == loc][param_display]
    ranges_data.append([loc_data.min(), loc_data.quantile(0.25), 
                       loc_data.median(), loc_data.quantile(0.75), loc_data.max()])

positions = range(len(location_colors))
for i, (loc, data) in enumerate(zip(location_colors.keys(), ranges_data)):
    ax.plot([i, i], [data[0], data[4]], color=location_colors[loc], linewidth=8, alpha=0.3)
    ax.plot([i, i], [data[1], data[3]], color=location_colors[loc], linewidth=12, alpha=0.6)
    ax.scatter(i, data[2], color=location_colors[loc], s=150, zorder=5, 
              edgecolors='black', linewidth=2)

ax.set_xticks(positions)
ax.set_xticklabels(location_colors.keys(), rotation=45, ha='right')
ax.set_ylabel(f'{param_display} (mg/L)')
ax.set_title(f'{param_display} Range by Location (Min-Q1-Median-Q3-Max)', fontweight='bold')
ax.axhline(y=6, color='green', linestyle='--', alpha=0.5, label='Healthy (6 mg/L)')
ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Hypoxic (2 mg/L)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 7.4: Key Statistics Table
ax = axes[1, 1]
ax.axis('off')

# Create summary table
summary_stats = df.groupby('Sampling_Point').agg({
    'WQI': 'mean',
    'Pollution_Score': 'mean',
    'DO': 'mean',
    'Turbidity': 'mean',
    'pH': 'mean'
}).round(1)

# Sort by WQI descending
summary_stats = summary_stats.sort_values('WQI', ascending=False)

# Create table
table_data = []
table_data.append(['Location', 'WQI', 'Poll.\nScore', 'DO\n(mg/L)', 'Turb.\n(NTU)', 'pH'])
for loc, row in summary_stats.iterrows():
    table_data.append([
        loc[:15],  # Truncate long names
        f"{row['WQI']:.1f}",
        f"{row['Pollution_Score']:.1f}",
        f"{row['DO']:.1f}",
        f"{row['Turbidity']:.1f}",
        f"{row['pH']:.2f}"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.14, 0.14, 0.14, 0.14, 0.14])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows by WQI
for i in range(1, len(table_data)):
    wqi = float(table_data[i][1])
    if wqi >= 70:
        color = '#d5f4e6'  # Light green
    elif wqi >= 50:
        color = '#fff3cd'  # Light yellow
    else:
        color = '#f8d7da'  # Light red
    for j in range(6):
        table[(i, j)].set_facecolor(color)

ax.set_title('Summary Statistics by Location', fontweight='bold', pad=20)

plt.suptitle('Statistical Summary Visualization', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{viz_dir}/statistical_summary_visualization.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_dir}/statistical_summary_visualization.png")
plt.close()

# ============================================================================
# 8. GENERATE HTML REPORT
# ============================================================================
print("\n[8] GENERATING HTML REPORT...")

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>River Water Quality Monitoring Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .metric-card {{
            display: inline-block;
            width: 22%;
            margin: 1%;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .location-card {{
            margin: 20px 0;
            padding: 15px;
            border-left: 5px solid;
            background-color: #ecf0f1;
        }}
        .excellent {{ border-color: #27ae60; }}
        .good {{ border-color: #2ecc71; }}
        .moderate {{ border-color: #f39c12; }}
        .poor {{ border-color: #e74c3c; }}
        .critical {{ border-color: #c0392b; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .visualization {{
            margin: 30px 0;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .alert {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .alert-danger {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }}
        .alert-warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }}
        .alert-info {{
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌊 River Water Quality Monitoring Report</h1>
        <p><strong>Report Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>
        <p><strong>Monitoring Period:</strong> {df['Date'].min().strftime('%B %d, %Y')} - {df['Date'].max().strftime('%B %d, %Y')}</p>
        <p><strong>Total Samples:</strong> {len(df)} | <strong>Sampling Locations:</strong> {df['Sampling_Point'].nunique()}</p>
        
        <div class="alert alert-info">
            <strong>Executive Summary:</strong> This report presents a comprehensive analysis of river water quality 
            across {df['Sampling_Point'].nunique()} sampling locations over {df['Month'].nunique()} months. 
            Analysis reveals significant spatial variations in water quality parameters.
        </div>

        <h2>📊 Key Performance Indicators</h2>
        <div style="text-align: center;">
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="metric-label">Average WQI</div>
                <div class="metric-value">{df['WQI'].mean():.1f}</div>
                <div class="metric-label">out of 100</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="metric-label">Pollution Score</div>
                <div class="metric-value">{df['Pollution_Score'].mean():.1f}</div>
                <div class="metric-label">average</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="metric-label">Avg DO</div>
                <div class="metric-value">{df['DO'].mean():.1f}</div>
                <div class="metric-label">mg/L</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <div class="metric-label">Samples</div>
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">total</div>
            </div>
        </div>

        <h2>📍 Location-Specific Analysis</h2>
        
"""

# Add location cards
loc_summary = df.groupby('Sampling_Point').agg({
    'WQI': 'mean',
    'Pollution_Score': 'mean',
    'DO': 'mean',
    'pH': 'mean',
    'Turbidity': 'mean'
}).round(2)

loc_summary = loc_summary.sort_values('WQI', ascending=False)

for loc, row in loc_summary.iterrows():
    wqi = row['WQI']
    if wqi >= 70:
        status = 'good'
        status_text = 'Good Quality'
    elif wqi >= 50:
        status = 'moderate'
        status_text = 'Moderate Quality'
    else:
        status = 'poor'
        status_text = 'Poor Quality'
    
    html_content += f"""
        <div class="location-card {status}">
            <h3>{loc}</h3>
            <p><strong>Status:</strong> {status_text} | <strong>WQI:</strong> {wqi:.1f}/100</p>
            <p>
                <strong>DO:</strong> {row['DO']:.2f} mg/L | 
                <strong>pH:</strong> {row['pH']:.2f} | 
                <strong>Turbidity:</strong> {row['Turbidity']:.1f} NTU | 
                <strong>Pollution Score:</strong> {row['Pollution_Score']:.1f}
            </p>
        </div>
    """

html_content += f"""
        <h2>⚠️ Critical Findings</h2>
        
        <div class="alert alert-danger">
            <strong>High Priority:</strong> Puente Bilbao shows consistently poor water quality with 
            {len(df[(df['Sampling_Point'] == 'Puente Bilbao') & (df['Pollution_Level'] == 'Critical')])} critical pollution events. 
            Immediate investigation and remediation recommended.
        </div>
        
        <div class="alert alert-warning">
            <strong>Moderate Concern:</strong> {len(df[df['Turbidity'] > 200])} instances of high turbidity (>200 NTU) 
            detected across all locations, indicating periodic pollution events.
        </div>
        
        <h2>📈 Visualizations</h2>
        
        <div class="visualization">
            <h3>Executive Dashboard</h3>
            <img src="executive_dashboard.png" alt="Executive Dashboard">
        </div>
        
        <div class="visualization">
            <h3>Temporal Analysis</h3>
            <img src="temporal_analysis_dashboard.png" alt="Temporal Analysis">
        </div>
        
        <div class="visualization">
            <h3>Location Comparison</h3>
            <img src="location_comparison_report.png" alt="Location Comparison">
        </div>
        
        <div class="visualization">
            <h3>Parameter Details</h3>
            <img src="parameter_comparison_detailed.png" alt="Parameter Comparison">
        </div>
        
        <div class="visualization">
            <h3>Pollution Events</h3>
            <img src="pollution_events_analysis.png" alt="Pollution Events">
        </div>
        
        <div class="visualization">
            <h3>Statistical Summary</h3>
            <img src="statistical_summary_visualization.png" alt="Statistical Summary">
        </div>
        
        <h2>📋 Detailed Statistics</h2>
        <table>
            <tr>
                <th>Location</th>
                <th>WQI</th>
                <th>Pollution Score</th>
                <th>DO (mg/L)</th>
                <th>pH</th>
                <th>Turbidity (NTU)</th>
                <th>Samples</th>
            </tr>
"""

for loc, row in loc_summary.iterrows():
    sample_count = len(df[df['Sampling_Point'] == loc])
    html_content += f"""
            <tr>
                <td>{loc}</td>
                <td>{row['WQI']:.1f}</td>
                <td>{row['Pollution_Score']:.1f}</td>
                <td>{row['DO']:.2f}</td>
                <td>{row['pH']:.2f}</td>
                <td>{row['Turbidity']:.1f}</td>
                <td>{sample_count}</td>
            </tr>
    """

html_content += f"""
        </table>
        
        <h2>💡 Recommendations</h2>
        <ol>
            <li><strong>Immediate Action:</strong> Investigate pollution sources at Puente Bilbao (worst water quality)</li>
            <li><strong>Enhanced Monitoring:</strong> Increase sampling frequency during high-risk periods</li>
            <li><strong>Pollution Control:</strong> Implement targeted measures to reduce turbidity and improve dissolved oxygen</li>
            <li><strong>Long-term Strategy:</strong> Develop watershed management plan focusing on identified hotspots</li>
            <li><strong>Stakeholder Engagement:</strong> Coordinate with upstream industrial/agricultural operators</li>
        </ol>
        
        <h2>📊 Data Quality Notes</h2>
        <ul>
            <li>Total samples analyzed: {len(df)}</li>
            <li>Sampling period: {(df['Date'].max() - df['Date'].min()).days} days</li>
            <li>Parameters monitored: pH, EC, TDS, TSS, DO, Turbidity, Hardness, Total Chlorine</li>
            <li>All statistical tests performed at α = 0.05 significance level</li>
            <li>Non-parametric methods used due to non-normal distributions</li>
        </ul>
        
        <div class="footer">
            <p><strong>River Water Quality Monitoring System</strong></p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
            <p>For questions or concerns, contact the environmental monitoring team</p>
        </div>
    </div>
</body>
</html>
"""

# Save HTML report
with open(f'{viz_dir}/water_quality_report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✓ Saved: {viz_dir}/water_quality_report.html")

# ============================================================================
# 9. GENERATE SUMMARY STATISTICS CSV
# ============================================================================
print("\n[9] GENERATING SUMMARY STATISTICS CSV...")

summary_df = df.groupby('Sampling_Point').agg({
    'WQI': ['mean', 'std', 'min', 'max'],
    'Pollution_Score': ['mean', 'std', 'min', 'max'],
    'pH': ['mean', 'std'],
    'DO': ['mean', 'std'],
    'Turbidity': ['mean', 'std', 'max'],
    'EC': ['mean', 'std'],
    'Hardness': ['mean', 'std'],
    'Total_Chlorine': ['mean', 'std']
}).round(2)

summary_df.to_csv(f'{viz_dir}/location_summary_statistics.csv')
print(f"✓ Saved: {viz_dir}/location_summary_statistics.csv")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DATA VISUALIZATION & REPORTING COMPLETE!")
print("=" * 80)

print(f"""
✓ ALL VISUALIZATIONS AND REPORTS GENERATED SUCCESSFULLY!

📁 Output Directory: {viz_dir}/

📊 Visualizations Created:
   1. executive_dashboard.png - High-level overview for stakeholders
   2. parameter_comparison_detailed.png - Detailed parameter analysis
   3. temporal_analysis_dashboard.png - Time-series trends
   4. location_comparison_report.png - Spatial comparison
   5. pollution_events_analysis.png - Pollution event tracking
   6. statistical_summary_visualization.png - Statistical insights

📄 Reports Generated:
   1. water_quality_report.html - Interactive HTML report (OPEN THIS!)
   2. location_summary_statistics.csv - Summary statistics table

🎯 Key Findings Highlighted:
   • Best Location: {loc_summary.index[0]} (WQI: {loc_summary.iloc[0]['WQI']:.1f})
   • Worst Location: {loc_summary.index[-1]} (WQI: {loc_summary.iloc[-1]['WQI']:.1f})
   • Overall WQI: {df['WQI'].mean():.1f}/100 (Moderate)
   • Critical Events: {len(df[df['Pollution_Level'] == 'Critical'])}
   • High Turbidity Events: {len(df[df['Turbidity'] > 200])}

💡 Next Steps:
   1. Open water_quality_report.html in your browser
   2. Review all visualizations
   3. Share with stakeholders
   4. Develop action plan based on findings
""")

print("=" * 80)
