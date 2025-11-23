"""
Weighting Schemes Module.

This module implements 5 different schemes for combining centrality scores
into overall city importance scores:
1. Equal Weighting - Treat all metrics equally
2. Correlation-Based - Down-weight correlated metrics
3. Category Importance - Weight by economic significance
4. Entropy-Based - Weight by information content
5. Interactive - User-defined custom weights

Functions:
    - apply_equal_weighting: Equal weights (0.25 each)
    - apply_correlation_weighting: Inverse correlation weights
    - apply_category_weighting: Category importance weights
    - apply_entropy_weighting: Information entropy weights
    - apply_interactive_weighting: Custom user weights
    - apply_all_weighting_schemes: Apply all 5 schemes at once
    - aggregate_scores_by_year: Aggregate to year-level
    - get_rankings: Get city rankings by scheme
    - compare_schemes: Compare rankings across schemes
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

from src.config import CATEGORY_WEIGHTS_IMPORTANCE, DEFAULT_WEIGHTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_equal_weighting(df_centrality: pd.DataFrame) -> pd.DataFrame:
    """
    Apply equal weighting to all centrality metrics (baseline approach).

    All metrics receive equal weight (0.25 each). This serves as a neutral
    baseline with no assumptions about relative importance.

    Algorithm:
        Score = 0.25 * D_i + 0.25 * C_i + 0.25 * B_i + 0.25 * E_i

    Args:
        df_centrality: DataFrame with columns [Year, Category, City, D_i, C_i, B_i, E_i]

    Returns:
        Same DataFrame with added column: Score_Equal

    Example:
        >>> df = apply_equal_weighting(df_centrality)
        >>> df[['City', 'Score_Equal']].head()
    """
    df = df_centrality.copy()
    
    logger.info("Applying equal weighting (0.25 each)")
    
    df['Score_Equal'] = 0.25 * (df['D_i'] + df['C_i'] + df['B_i'] + df['E_i'])
    
    return df


def apply_correlation_weighting(df_centrality: pd.DataFrame) -> pd.DataFrame:
    """
    Weight metrics inversely to their correlation (reduce redundancy).

    Down-weights metrics that are highly correlated with others,
    emphasizing unique information from each metric.

    Algorithm:
        1. Group by (Year, Category)
        2. For each group:
           - Calculate correlation matrix
           - For each metric: w_k = 1 / (1 + sum of |correlations|)
           - Normalize weights to sum to 1
           - Calculate weighted score

    Args:
        df_centrality: Centrality DataFrame

    Returns:
        DataFrame with added column: Score_Correlation

    Edge Cases:
        - Zero correlations: Falls back to equal weights
        - Constant values: Handled gracefully

    Example:
        >>> df = apply_correlation_weighting(df_centrality)
    """
    df = df_centrality.copy()
    
    logger.info("Applying correlation-based weighting")
    
    metrics = ['D_i', 'C_i', 'B_i', 'E_i']
    scores = []
    
    for (year, category), group in df.groupby(['Year', 'Category']):
        # Extract metric values
        metric_values = group[metrics]
        
        # Calculate correlation matrix
        try:
            corr_matrix = metric_values.corr().abs()
            
            # Calculate weights for each metric
            weights = {}
            for metric in metrics:
                # Sum correlations with other metrics (exclude self)
                corr_sum = corr_matrix.loc[metric, [m for m in metrics if m != metric]].sum()
                weights[metric] = 1.0 / (1.0 + corr_sum)
            
            # Normalize to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            else:
                # Fall back to equal weights
                weights = {k: 0.25 for k in metrics}
                
        except Exception as e:
            logger.warning(f"Correlation calculation failed for {year}-{category}: {e}. Using equal weights.")
            weights = {k: 0.25 for k in metrics}
        
        # Calculate weighted score
        group_scores = sum(group[metric] * weights[metric] for metric in metrics)
        scores.extend(group_scores.tolist())
    
    df['Score_Correlation'] = scores
    
    return df


def apply_category_weighting(
    df_centrality: pd.DataFrame,
    category_weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Weight by category economic importance.

    Two-stage weighting:
    1. Equal combination within category (0.25 each)
    2. Apply category importance weights (e.g., Food=0.25, Utilities=0.35)

    Algorithm:
        1. Calculate within-category score (equal weights)
        2. Apply category weight
        3. Aggregate across categories by year

    Args:
        df_centrality: Centrality DataFrame
        category_weights: Optional dict of category weights (default: from config)

    Returns:
        DataFrame with added columns:
        - Score_Category: Within-category score
        - Score_Category_Weighted: Category-weighted score

    Example:
        >>> df = apply_category_weighting(df_centrality)
    """
    df = df_centrality.copy()
    
    # Use default weights if not provided
    if category_weights is None:
        category_weights = CATEGORY_WEIGHTS_IMPORTANCE
    
    logger.info("Applying category importance weighting")
    
    # Stage 1: Equal combination within category
    df['Score_Category'] = 0.25 * (df['D_i'] + df['C_i'] + df['B_i'] + df['E_i'])
    
    # Stage 2: Apply category weight
    df['Score_Category_Weighted'] = df.apply(
        lambda row: row['Score_Category'] * category_weights.get(row['Category'], 0.1),
        axis=1
    )
    
    return df


def apply_entropy_weighting(df_centrality: pd.DataFrame) -> pd.DataFrame:
    """
    Weight by information content (Shannon entropy).

    Metrics with higher variance/entropy receive higher weights,
    emphasizing discriminating factors.

    Algorithm:
        1. Group by (Year, Category)
        2. For each metric:
           - Normalize to probability distribution
           - Calculate entropy: H = -sum(p * log(p))
           - Weight = H / sum(all H)
        3. Calculate weighted score

    Args:
        df_centrality: Centrality DataFrame

    Returns:
        DataFrame with added column: Score_Entropy

    Implementation Notes:
        - Adds epsilon (1e-10) to avoid log(0)
        - Falls back to equal weights if all entropies are zero

    Example:
        >>> df = apply_entropy_weighting(df_centrality)
    """
    df = df_centrality.copy()
    
    logger.info("Applying entropy-based weighting")
    
    metrics = ['D_i', 'C_i', 'B_i', 'E_i']
    scores = []
    epsilon = 1e-10
    
    for (year, category), group in df.groupby(['Year', 'Category']):
        entropies = {}
        
        for metric in metrics:
            values = group[metric].values
            
            # Normalize to probability distribution
            total = values.sum()
            if total > epsilon:
                probs = values / total
            else:
                # All zeros, use uniform distribution
                probs = np.ones(len(values)) / len(values)
            
            # Calculate Shannon entropy
            probs = probs + epsilon  # Avoid log(0)
            entropy = -np.sum(probs * np.log(probs))
            entropies[metric] = entropy
        
        # Calculate weights from entropies
        total_entropy = sum(entropies.values())
        if total_entropy > epsilon:
            weights = {k: v / total_entropy for k, v in entropies.items()}
        else:
            # All entropies zero (constant values), use equal weights
            weights = {k: 0.25 for k in metrics}
        
        # Calculate weighted score
        group_scores = sum(group[metric] * weights[metric] for metric in metrics)
        scores.extend(group_scores.tolist())
    
    df['Score_Entropy'] = scores
    
    return df


def apply_interactive_weighting(
    df_centrality: pd.DataFrame,
    weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Apply user-defined custom weights.

    Allows stakeholders to specify their own priorities for different
    centrality metrics.

    Algorithm:
        1. Validate weights (sum to 1, all non-negative, all keys present)
        2. Apply: Score = w_D * D_i + w_C * C_i + w_B * B_i + w_E * E_i

    Args:
        df_centrality: Centrality DataFrame
        weights: Dict like {'D_i': 0.4, 'C_i': 0.3, 'B_i': 0.2, 'E_i': 0.1}

    Returns:
        DataFrame with added column: Score_Interactive

    Raises:
        ValueError: If weights don't sum to 1 or are negative
        KeyError: If required metric missing

    Example:
        >>> custom_weights = {'D_i': 0.5, 'C_i': 0.3, 'B_i': 0.1, 'E_i': 0.1}
        >>> df = apply_interactive_weighting(df_centrality, custom_weights)
    """
    df = df_centrality.copy()
    
    # Validate weights
    required_keys = ['D_i', 'C_i', 'B_i', 'E_i']
    for key in required_keys:
        if key not in weights:
            raise KeyError(f"Missing required weight: {key}")
    
    for key, value in weights.items():
        if value < 0:
            raise ValueError(f"Weight for {key} must be non-negative, got {value}")
    
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1, got {weight_sum}")
    
    logger.info(f"Applying interactive weighting: {weights}")
    
    # Apply weights
    df['Score_Interactive'] = (
        weights['D_i'] * df['D_i'] +
        weights['C_i'] * df['C_i'] +
        weights['B_i'] * df['B_i'] +
        weights['E_i'] * df['E_i']
    )
    
    return df


def apply_all_weighting_schemes(df_centrality: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all 5 weighting schemes at once.

    Convenience function that applies all weighting schemes and returns
    a DataFrame with all score columns.

    Args:
        df_centrality: DataFrame with centrality metrics

    Returns:
        DataFrame with all score columns:
        - Score_Equal
        - Score_Correlation
        - Score_Category_Weighted
        - Score_Entropy
        - Score_Interactive (using equal weights 0.25 each)

    Example:
        >>> df_all = apply_all_weighting_schemes(df_centrality)
        >>> df_all.columns
    """
    logger.info("="*80)
    logger.info("Applying all weighting schemes")
    logger.info("="*80)
    
    df = df_centrality.copy()
    
    # Apply each scheme
    df = apply_equal_weighting(df)
    df = apply_correlation_weighting(df)
    df = apply_category_weighting(df)
    df = apply_entropy_weighting(df)
    df = apply_interactive_weighting(df, DEFAULT_WEIGHTS)
    
    logger.info("All weighting schemes applied")
    
    return df


def aggregate_scores_by_year(df_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate category-level scores to year-level.

    Different aggregation methods for different schemes:
    - Score_Equal, Score_Correlation, Score_Entropy, Score_Interactive: mean
    - Score_Category_Weighted: sum (already weighted by category importance)

    Algorithm:
        1. Group by (Year, City)
        2. Aggregate each score column appropriately
        3. Sort by year and score

    Args:
        df_scores: DataFrame with all score columns

    Returns:
        Year-level DataFrame with columns:
        [Year, City, Score_Equal, Score_Correlation, Score_Entropy,
         Score_Category_Weighted, Score_Interactive]

    Example:
        >>> df_year = aggregate_scores_by_year(df_scores)
    """
    logger.info("Aggregating scores by year")
    
    # Define aggregation methods
    agg_dict = {}
    if 'Score_Equal' in df_scores.columns:
        agg_dict['Score_Equal'] = 'mean'
    if 'Score_Correlation' in df_scores.columns:
        agg_dict['Score_Correlation'] = 'mean'
    if 'Score_Entropy' in df_scores.columns:
        agg_dict['Score_Entropy'] = 'mean'
    if 'Score_Category_Weighted' in df_scores.columns:
        agg_dict['Score_Category_Weighted'] = 'sum'  # Already weighted
    if 'Score_Interactive' in df_scores.columns:
        agg_dict['Score_Interactive'] = 'mean'
    
    # Group and aggregate
    df_year = df_scores.groupby(['Year', 'City']).agg(agg_dict).reset_index()
    
    # Sort by year
    df_year = df_year.sort_values(['Year', 'Score_Equal'], ascending=[True, False])
    
    logger.info(f"Aggregated to {len(df_year)} city-year combinations")
    
    return df_year


def get_rankings(
    df_scores: pd.DataFrame,
    year: int,
    scheme: str
) -> pd.DataFrame:
    """
    Get city rankings for a specific year and weighting scheme.

    Args:
        df_scores: Aggregated scores DataFrame
        year: Year to analyze
        scheme: One of ['Score_Equal', 'Score_Correlation', 'Score_Entropy',
                        'Score_Category_Weighted', 'Score_Interactive']

    Returns:
        DataFrame with columns: [Rank, City, Score]

    Example:
        >>> rankings = get_rankings(df_year, 2024, 'Score_Equal')
        >>> rankings.head()
    """
    # Validate scheme
    valid_schemes = ['Score_Equal', 'Score_Correlation', 'Score_Entropy',
                    'Score_Category_Weighted', 'Score_Interactive']
    if scheme not in valid_schemes:
        raise ValueError(f"Scheme must be one of {valid_schemes}, got '{scheme}'")
    
    # Filter to year
    df_year = df_scores[df_scores['Year'] == year].copy()
    
    if df_year.empty:
        logger.warning(f"No data for year {year}")
        return pd.DataFrame(columns=['Rank', 'City', scheme])
    
    # Sort by score
    df_year = df_year.sort_values(scheme, ascending=False).reset_index(drop=True)
    
    # Add rank
    df_year['Rank'] = range(1, len(df_year) + 1)
    
    # Return relevant columns
    return df_year[['Rank', 'City', scheme]]


def compare_schemes(
    df_scores: pd.DataFrame,
    year: int,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Compare how different schemes rank the same cities.

    Shows rankings and scores for top N cities under each scheme,
    highlighting where rankings diverge.

    Args:
        df_scores: Aggregated scores DataFrame
        year: Year to analyze
        top_n: Number of top cities to show

    Returns:
        Comparison DataFrame with rankings and scores for each scheme

    Example:
        >>> comparison = compare_schemes(df_year, 2024, top_n=10)
        >>> print(comparison)
    """
    schemes = ['Score_Equal', 'Score_Correlation', 'Score_Entropy',
              'Score_Category_Weighted', 'Score_Interactive']
    
    # Get rankings for each scheme
    all_rankings = []
    for scheme in schemes:
        if scheme in df_scores.columns:
            rankings = get_rankings(df_scores, year, scheme)
            rankings = rankings.head(top_n)
            rankings = rankings.rename(columns={
                'Rank': f'Rank_{scheme}',
                scheme: f'Score_{scheme}'
            })
            all_rankings.append(rankings[['City', f'Rank_{scheme}', f'Score_{scheme}']])
    
    # Merge all rankings
    if all_rankings:
        comparison = all_rankings[0]
        for df in all_rankings[1:]:
            comparison = comparison.merge(df, on='City', how='outer')
        
        return comparison
    else:
        return pd.DataFrame()


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    """
    Basic testing of weighting module functions.
    """
    print("="*80)
    print("WEIGHTING MODULE TESTS")
    print("="*80)
    
    # Create sample centrality data
    np.random.seed(42)
    cities = ['Islamabad', 'Karachi', 'Lahore', 'Quetta', 'Peshawar']
    categories = ['1. Food Staples & Grains', '2. Meat, Poultry & Dairy']
    years = [2023, 2024]
    
    data = []
    for year in years:
        for category in categories:
            for city in cities:
                data.append({
                    'Year': year,
                    'Category': category,
                    'City': city,
                    'D_i': np.random.rand(),
                    'C_i': np.random.rand(),
                    'B_i': np.random.rand(),
                    'E_i': np.random.rand()
                })
    
    df_test = pd.DataFrame(data)
    
    # Test 1: Equal weighting
    print("\n[Test 1] Equal Weighting")
    df_equal = apply_equal_weighting(df_test)
    assert 'Score_Equal' in df_equal.columns
    assert df_equal['Score_Equal'].min() >= 0
    assert df_equal['Score_Equal'].max() <= 1
    print(f"Score range: [{df_equal['Score_Equal'].min():.3f}, {df_equal['Score_Equal'].max():.3f}]")
    print("Equal weighting test PASSED")
    
    # Test 2: Correlation weighting
    print("\n[Test 2] Correlation Weighting")
    df_corr = apply_correlation_weighting(df_test)
    assert 'Score_Correlation' in df_corr.columns
    print("Correlation weighting test PASSED")
    
    # Test 3: Category weighting
    print("\n[Test 3] Category Weighting")
    df_cat = apply_category_weighting(df_test)
    assert 'Score_Category' in df_cat.columns
    assert 'Score_Category_Weighted' in df_cat.columns
    print("Category weighting test PASSED")
    
    # Test 4: Entropy weighting
    print("\n[Test 4] Entropy Weighting")
    df_entropy = apply_entropy_weighting(df_test)
    assert 'Score_Entropy' in df_entropy.columns
    print("Entropy weighting test PASSED")
    
    # Test 5: Interactive weighting
    print("\n[Test 5] Interactive Weighting")
    custom_weights = {'D_i': 0.4, 'C_i': 0.3, 'B_i': 0.2, 'E_i': 0.1}
    df_interactive = apply_interactive_weighting(df_test, custom_weights)
    assert 'Score_Interactive' in df_interactive.columns
    print("Interactive weighting test PASSED")
    
    # Test 6: Validation - weights must sum to 1
    print("\n[Test 6] Weight Validation")
    try:
        bad_weights = {'D_i': 0.5, 'C_i': 0.3, 'B_i': 0.2, 'E_i': 0.2}  # Sum = 1.2
        apply_interactive_weighting(df_test, bad_weights)
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Correctly caught invalid weights: {e}")
        print("Weight validation test PASSED")
    
    # Test 7: All schemes
    print("\n[Test 7] All Weighting Schemes")
    df_all = apply_all_weighting_schemes(df_test)
    expected_cols = ['Score_Equal', 'Score_Correlation', 'Score_Category_Weighted',
                     'Score_Entropy', 'Score_Interactive']
    for col in expected_cols:
        assert col in df_all.columns, f"Missing column: {col}"
    print("All schemes test PASSED")
    
    # Test 8: Aggregation
    print("\n[Test 8] Year Aggregation")
    df_year = aggregate_scores_by_year(df_all)
    assert len(df_year) == len(cities) * len(years)
    print(f"Aggregated to {len(df_year)} city-year combinations")
    print("Aggregation test PASSED")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80)
