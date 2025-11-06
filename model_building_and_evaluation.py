"""
Model Building and Evaluation
for Anime Recommender System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("ANIME RECOMMENDER SYSTEM - MODEL BUILDING & EVALUATION")
print("="*80)

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("\n[1] LOADING PREPROCESSED DATA...")
try:
    anime_clean = pd.read_csv('anime_clean.csv')
    train_data = pd.read_csv('train_no_negative.csv')
    test_data = pd.read_csv('test_clean.csv')
    user_stats = pd.read_csv('user_statistics.csv')
    anime_stats = pd.read_csv('anime_statistics.csv')
    print("✓ Data loaded successfully!")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Test samples: {len(test_data)}")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Please run eda_and_preprocessing.py first to generate the required files.")
    exit()

# ============================================================================
# 2. PREPARE EVALUATION FRAMEWORK
# ============================================================================
print("\n[2] PREPARING EVALUATION FRAMEWORK...")

# Split training data for validation
train_set, val_set = train_test_split(train_data, test_size=0.2, random_state=42)
print(f"  - Training set: {len(train_set)} samples")
print(f"  - Validation set: {len(val_set)} samples")

def evaluate_predictions(y_true, y_pred, model_name):
    """Calculate evaluation metrics for predictions"""
    # Ensure predictions are in valid range [1, 10]
    y_pred_clipped = np.clip(y_pred, 1, 10)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    mae = mean_absolute_error(y_true, y_pred_clipped)
    
    # Calculate R-squared
    ss_res = np.sum((y_true - y_pred_clipped) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"\n📊 {model_name} Metrics:")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAE:  {mae:.4f}")
    print(f"  - R²:   {r2:.4f}")
    
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Store results
results = []

# ============================================================================
# 3. BASELINE MODEL 1: GLOBAL AVERAGE
# ============================================================================
print("\n[3] BUILDING BASELINE MODEL 1: GLOBAL AVERAGE")
print("-" * 80)

global_mean = train_set['rating'].mean()
print(f"  - Global average rating: {global_mean:.4f}")

# Predict on validation set
val_pred_global = np.full(len(val_set), global_mean)

# Evaluate
result = evaluate_predictions(val_set['rating'].values, val_pred_global, "Global Average")
results.append(result)

# ============================================================================
# 4. BASELINE MODEL 2: USER AVERAGE
# ============================================================================
print("\n[4] BUILDING BASELINE MODEL 2: USER AVERAGE")
print("-" * 80)

# Calculate user averages from training set
user_avg = train_set.groupby('user_id')['rating'].mean().to_dict()
print(f"  - Calculated averages for {len(user_avg)} users")

# Predict on validation set
val_pred_user = val_set['user_id'].map(user_avg).fillna(global_mean).values

# Evaluate
result = evaluate_predictions(val_set['rating'].values, val_pred_user, "User Average")
results.append(result)

# ============================================================================
# 5. BASELINE MODEL 3: ANIME AVERAGE
# ============================================================================
print("\n[5] BUILDING BASELINE MODEL 3: ANIME AVERAGE")
print("-" * 80)

# Calculate anime averages from training set
anime_avg = train_set.groupby('anime_id')['rating'].mean().to_dict()
print(f"  - Calculated averages for {len(anime_avg)} anime")

# Predict on validation set
val_pred_anime = val_set['anime_id'].map(anime_avg).fillna(global_mean).values

# Evaluate
result = evaluate_predictions(val_set['rating'].values, val_pred_anime, "Anime Average")
results.append(result)

# ============================================================================
# 6. BASELINE MODEL 4: COMBINED (USER + ANIME - GLOBAL)
# ============================================================================
print("\n[6] BUILDING BASELINE MODEL 4: COMBINED BASELINE")
print("-" * 80)

# Predict using combined baseline
val_pred_combined = (
    val_set['user_id'].map(user_avg).fillna(global_mean) +
    val_set['anime_id'].map(anime_avg).fillna(global_mean) -
    global_mean
)
val_pred_combined = val_pred_combined.values

# Evaluate
result = evaluate_predictions(val_set['rating'].values, val_pred_combined, "Combined Baseline")
results.append(result)

# ============================================================================
# 7. COLLABORATIVE FILTERING: MATRIX FACTORIZATION (SVD)
# ============================================================================
print("\n[7] BUILDING COLLABORATIVE FILTERING MODEL: SVD")
print("-" * 80)

# Create user-item matrix
print("  - Creating user-item rating matrix...")

# Map user and anime IDs to indices
user_id_map = {uid: idx for idx, uid in enumerate(train_set['user_id'].unique())}
anime_id_map = {aid: idx for idx, aid in enumerate(train_set['anime_id'].unique())}

# Reverse maps for prediction
idx_to_user = {idx: uid for uid, idx in user_id_map.items()}
idx_to_anime = {idx: aid for aid, idx in anime_id_map.items()}

# Create sparse matrix
n_users = len(user_id_map)
n_anime = len(anime_id_map)

print(f"  - Matrix shape: {n_users} users × {n_anime} anime")

# Build sparse matrix from training set
row_indices = train_set['user_id'].map(user_id_map).values
col_indices = train_set['anime_id'].map(anime_id_map).values
ratings = train_set['rating'].values

user_anime_matrix = csr_matrix(
    (ratings, (row_indices, col_indices)),
    shape=(n_users, n_anime)
)

print(f"  - Matrix sparsity: {1 - (user_anime_matrix.nnz / (n_users * n_anime)):.4%}")

# Apply SVD
print("  - Applying SVD with 50 components...")
n_components = 50
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(user_anime_matrix)
anime_factors = svd.components_.T

print(f"  - Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

# Predict on validation set
print("  - Making predictions on validation set...")

val_pred_svd = []
for _, row in val_set.iterrows():
    user_id = row['user_id']
    anime_id = row['anime_id']
    
    # Check if user and anime are in training data
    if user_id in user_id_map and anime_id in anime_id_map:
        user_idx = user_id_map[user_id]
        anime_idx = anime_id_map[anime_id]
        
        # Predict rating
        pred = np.dot(user_factors[user_idx], anime_factors[anime_idx])
        val_pred_svd.append(pred)
    else:
        # Use global mean for cold start
        val_pred_svd.append(global_mean)

val_pred_svd = np.array(val_pred_svd)

# Evaluate
result = evaluate_predictions(val_set['rating'].values, val_pred_svd, "SVD (50 components)")
results.append(result)

# ============================================================================
# 8. USER-BASED COLLABORATIVE FILTERING
# ============================================================================
print("\n[8] BUILDING USER-BASED COLLABORATIVE FILTERING")
print("-" * 80)

print("  - Computing user similarity matrix (using subset for efficiency)...")

# Use a sample for efficiency (top users with most ratings)
top_n_users = 1000
top_users = train_set['user_id'].value_counts().head(top_n_users).index.tolist()

# Filter training data for top users
train_subset = train_set[train_set['user_id'].isin(top_users)].copy()

# Create user-item matrix for subset
user_id_map_sub = {uid: idx for idx, uid in enumerate(train_subset['user_id'].unique())}
anime_id_map_sub = {aid: idx for idx, aid in enumerate(train_subset['anime_id'].unique())}

n_users_sub = len(user_id_map_sub)
n_anime_sub = len(anime_id_map_sub)

row_indices_sub = train_subset['user_id'].map(user_id_map_sub).values
col_indices_sub = train_subset['anime_id'].map(anime_id_map_sub).values
ratings_sub = train_subset['rating'].values

user_anime_matrix_sub = csr_matrix(
    (ratings_sub, (row_indices_sub, col_indices_sub)),
    shape=(n_users_sub, n_anime_sub)
).toarray()

print(f"  - Subset matrix shape: {n_users_sub} users × {n_anime_sub} anime")

# Compute user similarity (cosine)
print("  - Computing cosine similarity between users...")
user_similarity = cosine_similarity(user_anime_matrix_sub)

# Make predictions for validation set
print("  - Making predictions...")
k_neighbors = 30

val_pred_user_cf = []
for _, row in val_set.iterrows():
    user_id = row['user_id']
    anime_id = row['anime_id']
    
    if user_id in user_id_map_sub and anime_id in anime_id_map_sub:
        user_idx = user_id_map_sub[user_id]
        anime_idx = anime_id_map_sub[anime_id]
        
        # Get similar users
        user_sims = user_similarity[user_idx]
        
        # Find users who rated this anime
        anime_ratings_mask = user_anime_matrix_sub[:, anime_idx] > 0
        
        # Get weighted ratings from similar users
        weighted_ratings = user_sims * user_anime_matrix_sub[:, anime_idx] * anime_ratings_mask
        sim_sum = np.sum(user_sims * anime_ratings_mask)
        
        if sim_sum > 0:
            pred = np.sum(weighted_ratings) / sim_sum
            val_pred_user_cf.append(pred)
        else:
            val_pred_user_cf.append(user_avg.get(user_id, global_mean))
    else:
        val_pred_user_cf.append(user_avg.get(user_id, global_mean))

val_pred_user_cf = np.array(val_pred_user_cf)

# Evaluate
result = evaluate_predictions(val_set['rating'].values, val_pred_user_cf, 
                              f"User-Based CF (k={k_neighbors})")
results.append(result)

# ============================================================================
# 9. ITEM-BASED COLLABORATIVE FILTERING
# ============================================================================
print("\n[9] BUILDING ITEM-BASED COLLABORATIVE FILTERING")
print("-" * 80)

print("  - Computing anime similarity matrix (using subset)...")

# Use subset of most popular anime
top_n_anime = 1000
top_anime = train_set['anime_id'].value_counts().head(top_n_anime).index.tolist()

# Filter training data
train_anime_subset = train_set[train_set['anime_id'].isin(top_anime)].copy()

# Create user-item matrix
user_id_map_anime = {uid: idx for idx, uid in enumerate(train_anime_subset['user_id'].unique())}
anime_id_map_anime = {aid: idx for idx, aid in enumerate(train_anime_subset['anime_id'].unique())}

n_users_anime = len(user_id_map_anime)
n_anime_anime = len(anime_id_map_anime)

row_indices_anime = train_anime_subset['user_id'].map(user_id_map_anime).values
col_indices_anime = train_anime_subset['anime_id'].map(anime_id_map_anime).values
ratings_anime = train_anime_subset['rating'].values

user_anime_matrix_anime = csr_matrix(
    (ratings_anime, (row_indices_anime, col_indices_anime)),
    shape=(n_users_anime, n_anime_anime)
).toarray()

print(f"  - Subset matrix shape: {n_users_anime} users × {n_anime_anime} anime")

# Compute anime similarity (transpose for item-based)
print("  - Computing cosine similarity between anime...")
anime_similarity = cosine_similarity(user_anime_matrix_anime.T)

# Make predictions
print("  - Making predictions...")
k_neighbors_item = 30

val_pred_item_cf = []
for _, row in val_set.iterrows():
    user_id = row['user_id']
    anime_id = row['anime_id']
    
    if user_id in user_id_map_anime and anime_id in anime_id_map_anime:
        user_idx = user_id_map_anime[user_id]
        anime_idx = anime_id_map_anime[anime_id]
        
        # Get similar anime
        anime_sims = anime_similarity[anime_idx]
        
        # Find anime rated by this user
        user_ratings_mask = user_anime_matrix_anime[user_idx] > 0
        
        # Get weighted ratings from similar anime
        weighted_ratings = anime_sims * user_anime_matrix_anime[user_idx] * user_ratings_mask
        sim_sum = np.sum(anime_sims * user_ratings_mask)
        
        if sim_sum > 0:
            pred = np.sum(weighted_ratings) / sim_sum
            val_pred_item_cf.append(pred)
        else:
            val_pred_item_cf.append(anime_avg.get(anime_id, global_mean))
    else:
        val_pred_item_cf.append(anime_avg.get(anime_id, global_mean))

val_pred_item_cf = np.array(val_pred_item_cf)

# Evaluate
result = evaluate_predictions(val_set['rating'].values, val_pred_item_cf,
                              f"Item-Based CF (k={k_neighbors_item})")
results.append(result)

# ============================================================================
# 10. HYBRID MODEL: WEIGHTED ENSEMBLE
# ============================================================================
print("\n[10] BUILDING HYBRID MODEL: WEIGHTED ENSEMBLE")
print("-" * 80)

# Combine best performing models
print("  - Combining SVD, User-Based CF, and Item-Based CF...")

# Weights (can be tuned)
w_svd = 0.4
w_user_cf = 0.3
w_item_cf = 0.3

val_pred_hybrid = (
    w_svd * val_pred_svd +
    w_user_cf * val_pred_user_cf +
    w_item_cf * val_pred_item_cf
)

print(f"  - Weights: SVD={w_svd}, User-CF={w_user_cf}, Item-CF={w_item_cf}")

# Evaluate
result = evaluate_predictions(val_set['rating'].values, val_pred_hybrid, "Hybrid Ensemble")
results.append(result)

# ============================================================================
# 11. MODEL COMPARISON
# ============================================================================
print("\n[11] MODEL COMPARISON")
print("-" * 80)

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE')

print("\n📊 MODEL PERFORMANCE COMPARISON:")
print(results_df.to_string(index=False))

# Find best model
best_model = results_df.iloc[0]
print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"   - RMSE: {best_model['RMSE']:.4f}")
print(f"   - MAE:  {best_model['MAE']:.4f}")
print(f"   - R²:   {best_model['R2']:.4f}")

# ============================================================================
# 12. VISUALIZE RESULTS
# ============================================================================
print("\n[12] CREATING VISUALIZATIONS...")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: RMSE Comparison
ax1 = axes[0, 0]
bars1 = ax1.barh(results_df['Model'], results_df['RMSE'], color='skyblue', edgecolor='black')
ax1.set_xlabel('RMSE (Lower is Better)', fontsize=12)
ax1.set_title('Model Comparison: RMSE', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
# Add value labels
for i, bar in enumerate(bars1):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}', ha='left', va='center', fontsize=10)

# Plot 2: MAE Comparison
ax2 = axes[0, 1]
bars2 = ax2.barh(results_df['Model'], results_df['MAE'], color='lightcoral', edgecolor='black')
ax2.set_xlabel('MAE (Lower is Better)', fontsize=12)
ax2.set_title('Model Comparison: MAE', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
# Add value labels
for i, bar in enumerate(bars2):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2,
             f'{width:.4f}', ha='left', va='center', fontsize=10)

# Plot 3: R² Comparison
ax3 = axes[1, 0]
bars3 = ax3.barh(results_df['Model'], results_df['R2'], color='lightgreen', edgecolor='black')
ax3.set_xlabel('R² Score (Higher is Better)', fontsize=12)
ax3.set_title('Model Comparison: R² Score', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
# Add value labels
for i, bar in enumerate(bars3):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2,
             f'{width:.4f}', ha='left', va='center', fontsize=10)

# Plot 4: Prediction vs Actual (Best Model - SVD)
ax4 = axes[1, 1]
sample_size = min(1000, len(val_set))
indices = np.random.choice(len(val_set), sample_size, replace=False)
ax4.scatter(val_set['rating'].values[indices], val_pred_svd[indices], 
           alpha=0.3, s=10, color='blue')
ax4.plot([1, 10], [1, 10], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Rating', fontsize=12)
ax4.set_ylabel('Predicted Rating', fontsize=12)
ax4.set_title('Prediction vs Actual (SVD Model)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'model_evaluation_results.png'")

# ============================================================================
# 13. ERROR ANALYSIS
# ============================================================================
print("\n[13] ERROR ANALYSIS")
print("-" * 80)

# Analyze errors for best model (SVD)
errors = val_set['rating'].values - val_pred_svd
abs_errors = np.abs(errors)

print("\n📊 ERROR DISTRIBUTION:")
print(f"  - Mean Error: {np.mean(errors):.4f}")
print(f"  - Std Error: {np.std(errors):.4f}")
print(f"  - Max Overestimation: {np.max(errors):.4f}")
print(f"  - Max Underestimation: {np.min(errors):.4f}")
print(f"  - 95th Percentile Error: {np.percentile(abs_errors, 95):.4f}")

# Error by rating value
print("\n📊 ERROR BY ACTUAL RATING:")
val_set_with_error = val_set.copy()
val_set_with_error['prediction'] = val_pred_svd
val_set_with_error['error'] = abs_errors

error_by_rating = val_set_with_error.groupby('rating')['error'].agg(['mean', 'std', 'count'])
print(error_by_rating)

# ============================================================================
# 14. GENERATE PREDICTIONS FOR TEST SET
# ============================================================================
print("\n[14] GENERATING PREDICTIONS FOR TEST SET")
print("-" * 80)

print("  - Using SVD model for final predictions...")

test_predictions = []
for _, row in test_data.iterrows():
    user_id = row['user_id']
    anime_id = row['anime_id']
    
    # SVD prediction
    if user_id in user_id_map and anime_id in anime_id_map:
        user_idx = user_id_map[user_id]
        anime_idx = anime_id_map[anime_id]
        pred_svd = np.dot(user_factors[user_idx], anime_factors[anime_idx])
    else:
        pred_svd = global_mean
    
    # Fallback to baseline
    pred_user = user_avg.get(user_id, global_mean)
    pred_anime = anime_avg.get(anime_id, global_mean)
    pred_combined = pred_user + pred_anime - global_mean
    
    # Weighted combination
    final_pred = 0.7 * pred_svd + 0.3 * pred_combined
    
    # Clip to valid range
    final_pred = np.clip(final_pred, 1, 10)
    
    test_predictions.append(final_pred)

# Create submission file
test_data['predicted_rating'] = test_predictions

print(f"  - Generated {len(test_predictions)} predictions")
print(f"  - Prediction statistics:")
print(f"    Mean: {np.mean(test_predictions):.4f}")
print(f"    Min: {np.min(test_predictions):.4f}")
print(f"    Max: {np.max(test_predictions):.4f}")
print(f"    Std: {np.std(test_predictions):.4f}")

# ============================================================================
# 15. SAVE RESULTS
# ============================================================================
print("\n[15] SAVING RESULTS")
print("-" * 80)

# Save results
results_df.to_csv('model_performance_comparison.csv', index=False)
print("✓ Saved: model_performance_comparison.csv")

# Save predictions
test_data[['user_id', 'anime_id', 'predicted_rating']].to_csv('test_predictions.csv', index=False)
print("✓ Saved: test_predictions.csv")

# Save error analysis
error_analysis = pd.DataFrame({
    'actual': val_set['rating'].values,
    'predicted': val_pred_svd,
    'error': errors,
    'abs_error': abs_errors
})
error_analysis.to_csv('error_analysis.csv', index=False)
print("✓ Saved: error_analysis.csv")

# ============================================================================
# 16. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"""
🎯 MODELS EVALUATED:
  1. Global Average Baseline
  2. User Average Baseline
  3. Anime Average Baseline
  4. Combined Baseline
  5. SVD Matrix Factorization
  6. User-Based Collaborative Filtering
  7. Item-Based Collaborative Filtering
  8. Hybrid Ensemble

🏆 BEST PERFORMING MODEL:
  - Name: {best_model['Model']}
  - RMSE: {best_model['RMSE']:.4f}
  - MAE:  {best_model['MAE']:.4f}
  - R²:   {best_model['R2']:.4f}

📊 KEY INSIGHTS:
  - Matrix factorization (SVD) typically outperforms simple baselines
  - Collaborative filtering captures user preferences effectively
  - Hybrid models can improve robustness
  - Cold start problem exists for new users/anime

📁 OUTPUT FILES:
  - model_performance_comparison.csv
  - test_predictions.csv
  - error_analysis.csv
  - model_evaluation_results.png

🚀 NEXT STEPS:
  1. Hyperparameter tuning (SVD components, k-neighbors)
  2. Advanced models (Deep Learning, Neural CF)
  3. Feature engineering (genre embedding, temporal features)
  4. Cross-validation for more robust evaluation
  5. Production deployment considerations
""")

print("="*80)
print("MODEL BUILDING AND EVALUATION COMPLETED SUCCESSFULLY!")
print("="*80)
