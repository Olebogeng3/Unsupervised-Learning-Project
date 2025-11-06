"""
Exploratory Data Analysis and Data Preparation
for Anime Recommender System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("ANIME RECOMMENDER SYSTEM - EDA & DATA PREPARATION")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")
try:
    anime_df = pd.read_csv('anime.csv')
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print("✓ Data loaded successfully!")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Please ensure anime.csv, train.csv, and test.csv are in the same directory.")
    exit()

# ============================================================================
# 2. DATASET OVERVIEW
# ============================================================================
print("\n[2] DATASET OVERVIEW")
print("-" * 80)

print("\n📊 ANIME.CSV:")
print(f"Shape: {anime_df.shape}")
print(f"Columns: {list(anime_df.columns)}")
print(f"\nFirst few rows:")
print(anime_df.head())

print("\n📊 TRAIN.CSV:")
print(f"Shape: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")
print(f"Unique users: {train_df['user_id'].nunique()}")
print(f"Unique anime: {train_df['anime_id'].nunique()}")
print(f"\nFirst few rows:")
print(train_df.head())

print("\n📊 TEST.CSV:")
print(f"Shape: {test_df.shape}")
print(f"Columns: {list(test_df.columns)}")
print(f"Unique users: {test_df['user_id'].nunique()}")
print(f"Unique anime: {test_df['anime_id'].nunique()}")

# ============================================================================
# 3. DATA QUALITY ASSESSMENT
# ============================================================================
print("\n[3] DATA QUALITY ASSESSMENT")
print("-" * 80)

print("\n🔍 Missing Values in anime.csv:")
anime_missing = anime_df.isnull().sum()
anime_missing_pct = (anime_missing / len(anime_df)) * 100
missing_anime = pd.DataFrame({
    'Missing_Count': anime_missing,
    'Percentage': anime_missing_pct
})
print(missing_anime[missing_anime['Missing_Count'] > 0])

print("\n🔍 Missing Values in train.csv:")
train_missing = train_df.isnull().sum()
train_missing_pct = (train_missing / len(train_df)) * 100
missing_train = pd.DataFrame({
    'Missing_Count': train_missing,
    'Percentage': train_missing_pct
})
print(missing_train[missing_train['Missing_Count'] > 0])

print("\n🔍 Data Types:")
print("\nAnime.csv dtypes:")
print(anime_df.dtypes)
print("\nTrain.csv dtypes:")
print(train_df.dtypes)

print("\n🔍 Duplicate Rows:")
print(f"Duplicates in anime.csv: {anime_df.duplicated().sum()}")
print(f"Duplicates in train.csv: {train_df.duplicated().sum()}")

# ============================================================================
# 4. STATISTICAL SUMMARY
# ============================================================================
print("\n[4] STATISTICAL SUMMARY")
print("-" * 80)

print("\n📈 Anime.csv Statistics:")
print(anime_df.describe())

print("\n📈 Train.csv Statistics:")
print(train_df.describe())

# Rating distribution in training data
print("\n📊 Rating Distribution in Training Data:")
rating_counts = train_df['rating'].value_counts().sort_index()
print(rating_counts)

# Count of -1 ratings (watched but not rated)
unrated_count = (train_df['rating'] == -1).sum()
print(f"\n⚠️  Users who watched but didn't rate: {unrated_count} ({unrated_count/len(train_df)*100:.2f}%)")

# ============================================================================
# 5. ANIME CHARACTERISTICS ANALYSIS
# ============================================================================
print("\n[5] ANIME CHARACTERISTICS ANALYSIS")
print("-" * 80)

# Anime types distribution
print("\n📺 Anime Types Distribution:")
type_dist = anime_df['type'].value_counts()
print(type_dist)

# Episodes analysis
print("\n📼 Episodes Analysis:")
print(f"Mean episodes: {anime_df['episodes'].mean():.2f}")
print(f"Median episodes: {anime_df['episodes'].median()}")
print(f"Max episodes: {anime_df['episodes'].max()}")
print(f"Min episodes: {anime_df['episodes'].min()}")

# Rating distribution
print("\n⭐ Anime Rating Distribution:")
print(f"Mean rating: {anime_df['rating'].mean():.2f}")
print(f"Median rating: {anime_df['rating'].median():.2f}")
print(f"Std rating: {anime_df['rating'].std():.2f}")

# Members distribution
print("\n👥 Members Distribution:")
print(f"Mean members: {anime_df['members'].mean():.2f}")
print(f"Median members: {anime_df['members'].median():.2f}")
print(f"Max members: {anime_df['members'].max()}")

# Genre analysis (if genre column exists and is not all NaN)
if 'genre' in anime_df.columns and anime_df['genre'].notna().sum() > 0:
    print("\n🎭 Genre Analysis:")
    # Extract all genres
    all_genres = []
    for genres in anime_df['genre'].dropna():
        all_genres.extend([g.strip() for g in str(genres).split(',')])
    
    genre_counts = pd.Series(all_genres).value_counts().head(20)
    print(f"Total unique genres: {len(pd.Series(all_genres).unique())}")
    print(f"\nTop 20 genres:")
    print(genre_counts)

# ============================================================================
# 6. USER BEHAVIOR ANALYSIS
# ============================================================================
print("\n[6] USER BEHAVIOR ANALYSIS")
print("-" * 80)

# Filter out -1 ratings for meaningful analysis
train_rated = train_df[train_df['rating'] != -1].copy()

# Ratings per user
ratings_per_user = train_df.groupby('user_id').size()
print(f"\n📊 Ratings per User:")
print(f"Mean: {ratings_per_user.mean():.2f}")
print(f"Median: {ratings_per_user.median():.2f}")
print(f"Min: {ratings_per_user.min()}")
print(f"Max: {ratings_per_user.max()}")
print(f"Std: {ratings_per_user.std():.2f}")

# Ratings per anime
ratings_per_anime = train_df.groupby('anime_id').size()
print(f"\n📊 Ratings per Anime:")
print(f"Mean: {ratings_per_anime.mean():.2f}")
print(f"Median: {ratings_per_anime.median():.2f}")
print(f"Min: {ratings_per_anime.min()}")
print(f"Max: {ratings_per_anime.max()}")
print(f"Std: {ratings_per_anime.std():.2f}")

# Rating statistics (excluding -1)
print(f"\n⭐ User Rating Statistics (excluding -1):")
print(f"Mean rating: {train_rated['rating'].mean():.2f}")
print(f"Median rating: {train_rated['rating'].median():.2f}")
print(f"Mode rating: {train_rated['rating'].mode().values[0]}")
print(f"Std rating: {train_rated['rating'].std():.2f}")

# ============================================================================
# 7. SPARSITY ANALYSIS
# ============================================================================
print("\n[7] SPARSITY ANALYSIS")
print("-" * 80)

n_users = train_df['user_id'].nunique()
n_anime = train_df['anime_id'].nunique()
n_ratings = len(train_df)
matrix_size = n_users * n_anime
sparsity = 1 - (n_ratings / matrix_size)

print(f"\nUser-Item Matrix Dimensions: {n_users} users × {n_anime} anime")
print(f"Possible ratings: {matrix_size:,}")
print(f"Actual ratings: {n_ratings:,}")
print(f"Sparsity: {sparsity*100:.4f}%")
print(f"Density: {(1-sparsity)*100:.4f}%")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n[8] CREATING VISUALIZATIONS...")
print("-" * 80)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# 1. Rating distribution in training data
ax1 = plt.subplot(3, 3, 1)
train_rated['rating'].hist(bins=10, edgecolor='black', alpha=0.7)
plt.title('Rating Distribution (excluding -1)', fontsize=12, fontweight='bold')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)

# 2. Anime type distribution
ax2 = plt.subplot(3, 3, 2)
anime_df['type'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Anime Type Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# 3. Top 20 most rated anime
ax3 = plt.subplot(3, 3, 3)
top_rated_anime = train_df['anime_id'].value_counts().head(20)
top_rated_anime.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Top 20 Most Rated Anime', fontsize=12, fontweight='bold')
plt.xlabel('Anime ID')
plt.ylabel('Number of Ratings')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# 4. Ratings per user distribution
ax4 = plt.subplot(3, 3, 4)
ratings_per_user_plot = ratings_per_user[ratings_per_user <= ratings_per_user.quantile(0.95)]
plt.hist(ratings_per_user_plot, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
plt.title('Ratings per User (95th percentile)', fontsize=12, fontweight='bold')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.grid(axis='y', alpha=0.3)

# 5. Anime rating distribution
ax5 = plt.subplot(3, 3, 5)
anime_df['rating'].dropna().hist(bins=30, edgecolor='black', alpha=0.7, color='purple')
plt.title('Anime Average Rating Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)

# 6. Members distribution (log scale)
ax6 = plt.subplot(3, 3, 6)
plt.hist(np.log10(anime_df['members'] + 1), bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.title('Anime Members Distribution (log10)', fontsize=12, fontweight='bold')
plt.xlabel('Log10(Members + 1)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)

# 7. Episodes distribution (filtered)
ax7 = plt.subplot(3, 3, 7)
episodes_filtered = anime_df[anime_df['episodes'] <= 100]['episodes']
plt.hist(episodes_filtered, bins=50, edgecolor='black', alpha=0.7, color='teal')
plt.title('Episodes Distribution (≤100)', fontsize=12, fontweight='bold')
plt.xlabel('Number of Episodes')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)

# 8. User rating behavior
ax8 = plt.subplot(3, 3, 8)
user_avg_ratings = train_rated.groupby('user_id')['rating'].mean()
plt.hist(user_avg_ratings, bins=30, edgecolor='black', alpha=0.7, color='pink')
plt.title('Average Rating per User', fontsize=12, fontweight='bold')
plt.xlabel('Average Rating')
plt.ylabel('Number of Users')
plt.grid(axis='y', alpha=0.3)

# 9. Correlation heatmap (anime features)
ax9 = plt.subplot(3, 3, 9)
anime_numeric = anime_df[['rating', 'members', 'episodes']].dropna()
correlation_matrix = anime_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation: Anime Features', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'eda_visualizations.png'")

# ============================================================================
# 9. DATA PREPROCESSING
# ============================================================================
print("\n[9] DATA PREPROCESSING")
print("-" * 80)

# Create clean copies
anime_clean = anime_df.copy()
train_clean = train_df.copy()
test_clean = test_df.copy()

# Handle missing values in anime data
print("\n🔧 Handling missing values in anime data...")

# Fill missing genres
if anime_clean['genre'].isnull().sum() > 0:
    anime_clean['genre'] = anime_clean['genre'].fillna('Unknown')
    print(f"  - Filled {anime_df['genre'].isnull().sum()} missing genres with 'Unknown'")

# Fill missing types
if anime_clean['type'].isnull().sum() > 0:
    anime_clean['type'] = anime_clean['type'].fillna('Unknown')
    print(f"  - Filled {anime_df['type'].isnull().sum()} missing types with 'Unknown'")

# Fill missing episodes with median
if anime_clean['episodes'].isnull().sum() > 0:
    median_episodes = anime_clean['episodes'].median()
    anime_clean['episodes'] = anime_clean['episodes'].fillna(median_episodes)
    print(f"  - Filled missing episodes with median: {median_episodes}")

# Fill missing ratings with mean
if anime_clean['rating'].isnull().sum() > 0:
    mean_rating = anime_clean['rating'].mean()
    anime_clean['rating'] = anime_clean['rating'].fillna(mean_rating)
    print(f"  - Filled missing ratings with mean: {mean_rating:.2f}")

# Handle -1 ratings in training data
print("\n🔧 Handling -1 ratings in training data...")
n_negative_ratings = (train_clean['rating'] == -1).sum()
print(f"  - Found {n_negative_ratings} ratings with value -1")
print(f"  - Options: Keep as -1, Remove, or Impute with user/anime average")

# Option 1: Remove -1 ratings (recommended for most models)
train_no_neg = train_clean[train_clean['rating'] != -1].copy()
print(f"  - Created train_no_neg: {len(train_no_neg)} rows (removed -1 ratings)")

# Option 2: Keep original (for models that can handle it)
# train_with_neg = train_clean.copy()

# Create feature engineering
print("\n🔧 Feature Engineering...")

# Add popularity score for anime
anime_clean['popularity_score'] = np.log10(anime_clean['members'] + 1)
print("  - Added popularity_score (log10 of members)")

# Add genre count
anime_clean['genre_count'] = anime_clean['genre'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) and x != 'Unknown' else 0
)
print("  - Added genre_count")

# Merge anime features with training data
print("\n🔧 Merging anime features with training data...")
train_enriched = train_no_neg.merge(
    anime_clean[['anime_id', 'type', 'episodes', 'rating', 'members', 'popularity_score', 'genre_count']],
    on='anime_id',
    how='left',
    suffixes=('_user', '_anime')
)
print(f"  - Created train_enriched: {train_enriched.shape}")

# User statistics
print("\n🔧 Computing user statistics...")
user_stats = train_no_neg.groupby('user_id').agg({
    'rating': ['count', 'mean', 'std', 'min', 'max']
}).reset_index()
user_stats.columns = ['user_id', 'user_rating_count', 'user_mean_rating', 
                      'user_std_rating', 'user_min_rating', 'user_max_rating']
user_stats['user_std_rating'] = user_stats['user_std_rating'].fillna(0)
print(f"  - Computed statistics for {len(user_stats)} users")

# Anime statistics
print("\n🔧 Computing anime statistics...")
anime_stats = train_no_neg.groupby('anime_id').agg({
    'rating': ['count', 'mean', 'std', 'min', 'max']
}).reset_index()
anime_stats.columns = ['anime_id', 'anime_rating_count', 'anime_mean_rating',
                       'anime_std_rating', 'anime_min_rating', 'anime_max_rating']
anime_stats['anime_std_rating'] = anime_stats['anime_std_rating'].fillna(0)
print(f"  - Computed statistics for {len(anime_stats)} anime")

# ============================================================================
# 10. SAVE PREPROCESSED DATA
# ============================================================================
print("\n[10] SAVING PREPROCESSED DATA")
print("-" * 80)

# Save cleaned data
anime_clean.to_csv('anime_clean.csv', index=False)
print("✓ Saved: anime_clean.csv")

train_clean.to_csv('train_clean.csv', index=False)
print("✓ Saved: train_clean.csv")

train_no_neg.to_csv('train_no_negative.csv', index=False)
print("✓ Saved: train_no_negative.csv")

train_enriched.to_csv('train_enriched.csv', index=False)
print("✓ Saved: train_enriched.csv")

user_stats.to_csv('user_statistics.csv', index=False)
print("✓ Saved: user_statistics.csv")

anime_stats.to_csv('anime_statistics.csv', index=False)
print("✓ Saved: anime_statistics.csv")

test_clean.to_csv('test_clean.csv', index=False)
print("✓ Saved: test_clean.csv")

# ============================================================================
# 11. SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

print(f"""
📊 DATA OVERVIEW:
  - Total anime: {len(anime_clean)}
  - Total users: {train_df['user_id'].nunique()}
  - Total ratings: {len(train_df)}
  - Ratings (excluding -1): {len(train_no_neg)}
  - Test samples: {len(test_df)}

🎯 DATA QUALITY:
  - Missing values handled: ✓
  - Duplicates checked: ✓
  - Feature engineering completed: ✓
  - Statistical analysis completed: ✓

📁 OUTPUT FILES:
  - anime_clean.csv
  - train_clean.csv
  - train_no_negative.csv
  - train_enriched.csv
  - user_statistics.csv
  - anime_statistics.csv
  - test_clean.csv
  - eda_visualizations.png

🚀 NEXT STEPS:
  1. Build baseline models (popularity-based, user/item average)
  2. Implement collaborative filtering (Matrix Factorization, SVD)
  3. Implement content-based filtering
  4. Build hybrid recommender system
  5. Evaluate and tune models
  6. Generate predictions for test.csv
""")

print("="*80)
print("EDA AND PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*80)
