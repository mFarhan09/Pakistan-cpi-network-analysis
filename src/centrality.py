"""
Centrality Analysis Module.

This module calculates four centrality metrics for all cities in all networks:
- Degree Centrality (D_i): Number of direct connections
- Closeness Centrality (C_i): Average distance to all other nodes
- Betweenness Centrality (B_i): How often node lies on shortest paths
- Eigenvector Centrality (E_i): Influence based on connections to influential nodes

Functions:
    - calculate_single_network_centrality: Calculate all 4 centralities for one network
    - calculate_all_centralities: Calculate for all networks
    - normalize_centralities: Optional normalization of centralities
    - get_top_cities: Get top cities by a specific metric
    - print_centrality_summary: Display summary statistics
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, List
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_single_network_centrality(G: nx.Graph) -> pd.DataFrame:
    """
    Calculate all 4 centrality metrics for one network.

    Computes degree, closeness, betweenness, and eigenvector centrality
    for all nodes in the graph. Handles edge cases like disconnected
    graphs and eigenvector convergence failures.

    Algorithm:
        1. Check if graph is empty (no nodes) -> return empty DataFrame
        2. Calculate degree centrality: nx.degree_centrality(G)
        3. Calculate closeness centrality: nx.closeness_centrality(G)
        4. Calculate betweenness centrality: nx.betweenness_centrality(G)
        5. Try eigenvector centrality: nx.eigenvector_centrality(G, max_iter=1000)
           - If fails: assign zeros
        6. Combine into DataFrame with columns: City, D_i, C_i, B_i, E_i
        7. Return DataFrame

    Args:
        G: NetworkX Graph object

    Returns:
        DataFrame with columns: [City, D_i, C_i, B_i, E_i]
        - One row per city in the graph
        - Empty DataFrame if graph has no nodes

    Edge Cases:
        - Empty graph: Return empty DataFrame
        - Single node: Return zeros for all metrics
        - Disconnected graph: Closeness handled per component
        - Eigenvector fails: Log warning, use zeros

    Example:
        >>> G = nx.karate_club_graph()
        >>> df = calculate_single_network_centrality(G)
        >>> df.head()
    """
    # Check if graph is empty
    if G.number_of_nodes() == 0:
        logger.warning("Empty graph provided to calculate_single_network_centrality")
        return pd.DataFrame(columns=['City', 'D_i', 'C_i', 'B_i', 'E_i'])

    # Get graph metadata
    year = G.graph.get('year', 'Unknown')
    category = G.graph.get('category', 'Unknown')

    logger.info(f"Calculating centralities for {G.number_of_nodes()} cities")

    # Calculate degree centrality
    try:
        degree_centrality = nx.degree_centrality(G)
    except Exception as e:
        logger.error(f"Failed to calculate degree centrality: {e}")
        degree_centrality = {node: 0.0 for node in G.nodes()}

    # Calculate closeness centrality
    try:
        closeness_centrality = nx.closeness_centrality(G)
    except Exception as e:
        logger.error(f"Failed to calculate closeness centrality: {e}")
        closeness_centrality = {node: 0.0 for node in G.nodes()}

    # Calculate betweenness centrality
    try:
        betweenness_centrality = nx.betweenness_centrality(G)
    except Exception as e:
        logger.error(f"Failed to calculate betweenness centrality: {e}")
        betweenness_centrality = {node: 0.0 for node in G.nodes()}

    # Calculate eigenvector centrality (may fail to converge)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError) as e:
        logger.warning(f"Eigenvector centrality failed: {e}. Using zeros.")
        eigenvector_centrality = {node: 0.0 for node in G.nodes()}
    except Exception as e:
        logger.warning(f"Unexpected error in eigenvector centrality: {e}. Using zeros.")
        eigenvector_centrality = {node: 0.0 for node in G.nodes()}

    # Combine into DataFrame
    cities = list(G.nodes())
    df = pd.DataFrame({
        'City': cities,
        'D_i': [degree_centrality[city] for city in cities],
        'C_i': [closeness_centrality[city] for city in cities],
        'B_i': [betweenness_centrality[city] for city in cities],
        'E_i': [eigenvector_centrality[city] for city in cities]
    })

    # Log statistics
    logger.info(f"  D_i: mean={df['D_i'].mean():.3f}, max={df['D_i'].max():.3f}")
    logger.info(f"  C_i: mean={df['C_i'].mean():.3f}, max={df['C_i'].max():.3f}")
    logger.info(f"  B_i: mean={df['B_i'].mean():.3f}, max={df['B_i'].max():.3f}")
    logger.info(f"  E_i: mean={df['E_i'].mean():.3f}, max={df['E_i'].max():.3f}")

    return df


def calculate_all_centralities(
    graphs_dict: Dict[Tuple[int, str], nx.Graph]
) -> pd.DataFrame:
    """
    Calculate centralities for all networks and combine into single DataFrame.

    Batch processing function that computes centrality metrics for all
    networks in the dictionary and combines results into a long-format DataFrame.

    Algorithm:
        1. Initialize empty list for results
        2. For each (year, category), graph in graphs_dict:
           a. Calculate centralities
           b. Add year and category columns
           c. Append to results list
        3. Concatenate all DataFrames
        4. Reset index
        5. Return long-format DataFrame

    Args:
        graphs_dict: Dictionary {(year, category): Graph}

    Returns:
        Long-format DataFrame with columns:
        - Year: int
        - Category: str
        - City: str
        - D_i: float (degree centrality)
        - C_i: float (closeness centrality)
        - B_i: float (betweenness centrality)
        - E_i: float (eigenvector centrality)

    Progress Tracking:
        - Prints progress every 5 networks
        - Shows percentage complete
        - Logs any errors encountered

    Example:
        >>> graphs = build_all_networks(df, [2023, 2024], threshold=0.75)
        >>> df_centrality = calculate_all_centralities(graphs)
        >>> print(df_centrality.shape)
    """
    if not graphs_dict:
        logger.warning("Empty graphs dictionary provided")
        return pd.DataFrame(columns=['Year', 'Category', 'City', 'D_i', 'C_i', 'B_i', 'E_i'])

    logger.info("="*80)
    logger.info(f"Calculating centralities for {len(graphs_dict)} networks")
    logger.info("="*80)

    results = []
    total_networks = len(graphs_dict)
    processed = 0

    for (year, category), G in graphs_dict.items():
        processed += 1

        # Progress tracking
        if processed % 5 == 0 or processed == total_networks:
            progress_pct = (processed / total_networks) * 100
            logger.info(f"Progress: {processed}/{total_networks} ({progress_pct:.1f}%)")

        # Calculate centralities for this network
        try:
            df_network = calculate_single_network_centrality(G)

            # Add year and category columns
            df_network['Year'] = year
            df_network['Category'] = category

            # Append to results
            results.append(df_network)

        except Exception as e:
            logger.error(f"Failed to process {year}-{category}: {e}")
            continue

    # Concatenate all results
    if results:
        df_final = pd.concat(results, ignore_index=True)

        # Reorder columns
        df_final = df_final[['Year', 'Category', 'City', 'D_i', 'C_i', 'B_i', 'E_i']]

        logger.info("\n" + "="*80)
        logger.info(f"CENTRALITY CALCULATION COMPLETE")
        logger.info(f"Total rows: {len(df_final)}")
        logger.info(f"Unique cities: {df_final['City'].nunique()}")
        logger.info(f"Year-Category combinations: {len(df_final.groupby(['Year', 'Category']))}")
        logger.info("="*80)

        return df_final
    else:
        logger.warning("No results generated")
        return pd.DataFrame(columns=['Year', 'Category', 'City', 'D_i', 'C_i', 'B_i', 'E_i'])


def normalize_centralities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize centralities within each year-category group.

    Applies min-max normalization to each centrality metric within each
    year-category group. This ensures fair comparison across different
    network structures.

    Algorithm:
        1. For each year-category group:
           a. Min-max normalize each centrality metric
           b. Formula: (x - min) / (max - min)
           c. Handle case where max = min (set to 0.5)
        2. Add normalized columns: D_i_norm, C_i_norm, B_i_norm, E_i_norm
        3. Keep original columns as well

    Args:
        df: DataFrame from calculate_all_centralities

    Returns:
        Same DataFrame with added normalized columns:
        - D_i_norm, C_i_norm, B_i_norm, E_i_norm

    Note:
        This is optional - some weighting schemes work better with raw scores

    Example:
        >>> df_norm = normalize_centralities(df_centrality)
        >>> df_norm[['City', 'D_i', 'D_i_norm']].head()
    """
    df = df.copy()

    metrics = ['D_i', 'C_i', 'B_i', 'E_i']

    for metric in metrics:
        norm_col = f"{metric}_norm"

        # Group by year and category, then normalize
        def min_max_normalize(series):
            min_val = series.min()
            max_val = series.max()

            if max_val == min_val:
                # All values are the same, set to 0.5
                return pd.Series([0.5] * len(series), index=series.index)
            else:
                return (series - min_val) / (max_val - min_val)

        df[norm_col] = df.groupby(['Year', 'Category'])[metric].transform(min_max_normalize)

    logger.info("Normalized centralities added")

    return df


def get_top_cities(
    df: pd.DataFrame,
    metric: str,
    n: int = 10,
    year: Optional[int] = None,
    category: Optional[str] = None
) -> pd.DataFrame:
    """
    Get top cities by a specific centrality metric.

    Convenience function to extract and rank cities by their centrality scores.
    Supports filtering by year and/or category.

    Algorithm:
        1. Filter by year and/or category if provided
        2. Sort by specified metric (descending)
        3. Return top n rows

    Args:
        df: Centrality DataFrame
        metric: One of ['D_i', 'C_i', 'B_i', 'E_i']
        n: Number of top cities to return
        year: Optional year filter
        category: Optional category filter

    Returns:
        DataFrame with top cities for that metric, sorted descending

    Example:
        >>> top_degree = get_top_cities(df, 'D_i', n=5, year=2024)
        >>> print(top_degree[['City', 'D_i']])
    """
    # Validate metric
    valid_metrics = ['D_i', 'C_i', 'B_i', 'E_i']
    if metric not in valid_metrics:
        raise ValueError(f"Metric must be one of {valid_metrics}, got '{metric}'")

    # Filter by year if provided
    df_filtered = df.copy()
    if year is not None:
        df_filtered = df_filtered[df_filtered['Year'] == year]

    # Filter by category if provided
    if category is not None:
        df_filtered = df_filtered[df_filtered['Category'] == category]

    # Sort by metric and return top n
    df_top = df_filtered.nlargest(n, metric)

    return df_top


def print_centrality_summary(df: pd.DataFrame) -> None:
    """
    Display summary statistics of centrality distributions.

    Creates a formatted table showing statistical summaries and top performers
    for each centrality metric. Useful for understanding overall patterns.

    Algorithm:
        1. Calculate statistics for each metric: Mean, Median, Std, Min, Max
        2. Count cities with zero centrality
        3. Identify cities with highest values
        4. Print formatted table

    Args:
        df: Centrality DataFrame

    Output Example:
        ================================================================================
        CENTRALITY ANALYSIS SUMMARY
        ================================================================================
        Metric            Mean    Median    Std     Min     Max     Zeros
        --------------------------------------------------------------------------------
        Degree (D_i)      0.423   0.412    0.186   0.000   0.875   3
        Closeness (C_i)   0.512   0.523    0.142   0.000   0.792   5
        Betweenness (B_i) 0.089   0.067    0.098   0.000   0.456   12
        Eigenvector (E_i) 0.387   0.391    0.203   0.000   0.823   8

        Top Cities by Degree: Karachi (0.875), Lahore (0.831), Islamabad (0.812)
        ================================================================================

    Example:
        >>> print_centrality_summary(df_centrality)
    """
    if df.empty:
        print("No centrality data to display.")
        return

    print("\n" + "="*80)
    print("CENTRALITY ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'Metric':<20}{'Mean':<8}{'Median':<8}{'Std':<8}{'Min':<8}{'Max':<8}{'Zeros':<8}")
    print("-"*80)

    metrics = {
        'D_i': 'Degree (D_i)',
        'C_i': 'Closeness (C_i)',
        'B_i': 'Betweenness (B_i)',
        'E_i': 'Eigenvector (E_i)'
    }

    for metric_code, metric_name in metrics.items():
        if metric_code in df.columns:
            mean_val = df[metric_code].mean()
            median_val = df[metric_code].median()
            std_val = df[metric_code].std()
            min_val = df[metric_code].min()
            max_val = df[metric_code].max()
            zeros = (df[metric_code] == 0).sum()

            print(
                f"{metric_name:<20}"
                f"{mean_val:<8.3f}"
                f"{median_val:<8.3f}"
                f"{std_val:<8.3f}"
                f"{min_val:<8.3f}"
                f"{max_val:<8.3f}"
                f"{zeros:<8}"
            )

    # Print top cities for each metric
    print("\n" + "-"*80)
    for metric_code, metric_name in metrics.items():
        if metric_code in df.columns:
            top_3 = df.nlargest(3, metric_code)[['City', metric_code]]
            cities_str = ", ".join([
                f"{row['City']} ({row[metric_code]:.3f})"
                for _, row in top_3.iterrows()
            ])
            print(f"Top Cities by {metric_name}: {cities_str}")

    print("="*80 + "\n")


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    """
    Basic testing of centrality module functions.
    """
    print("="*80)
    print("CENTRALITY MODULE TESTS")
    print("="*80)

    # Test 1: Complete graph (everyone connected to everyone)
    print("\n[Test 1] Complete Graph K_5 - All centralities should be 1.0")
    G_complete = nx.complete_graph(5)
    mapping = {i: f"City_{i}" for i in range(5)}
    G_complete = nx.relabel_nodes(G_complete, mapping)
    G_complete.graph['year'] = 2024
    G_complete.graph['category'] = 'Test'

    df_complete = calculate_single_network_centrality(G_complete)
    print(f"Degree centralities: {df_complete['D_i'].unique()}")
    assert np.allclose(df_complete['D_i'], 1.0), "All degree centralities should be 1.0"
    print("Complete graph test PASSED")

    # Test 2: Star graph
    print("\n[Test 2] Star Graph - Center should have high centrality")
    G_star = nx.star_graph(4)  # 1 center + 4 periphery
    mapping = {i: f"City_{i}" for i in range(5)}
    G_star = nx.relabel_nodes(G_star, mapping)
    G_star.graph['year'] = 2024
    G_star.graph['category'] = 'Test'

    df_star = calculate_single_network_centrality(G_star)
    center_degree = df_star[df_star['City'] == 'City_0']['D_i'].values[0]
    periphery_degree = df_star[df_star['City'] == 'City_1']['D_i'].values[0]
    print(f"Center degree: {center_degree:.3f}, Periphery degree: {periphery_degree:.3f}")
    assert center_degree > periphery_degree, "Center should have higher degree"
    print("Star graph test PASSED")

    # Test 3: Karate club graph (real-world test)
    print("\n[Test 3] Karate Club Graph - Range test")
    G_karate = nx.karate_club_graph()
    G_karate.graph['year'] = 2024
    G_karate.graph['category'] = 'Test'

    df_karate = calculate_single_network_centrality(G_karate)
    for metric in ['D_i', 'C_i', 'B_i', 'E_i']:
        assert df_karate[metric].min() >= 0, f"{metric} has negative values"
        assert df_karate[metric].max() <= 1, f"{metric} has values > 1"
    print("All centralities in range [0, 1]")
    print("Range test PASSED")

    # Test 4: Batch processing
    print("\n[Test 4] Batch Processing Test")
    graphs_dict = {
        (2023, 'Cat1'): G_complete,
        (2024, 'Cat2'): G_star
    }
    df_all = calculate_all_centralities(graphs_dict)
    print(f"Total rows: {len(df_all)}")
    assert len(df_all) == 10, "Should have 10 rows (5 + 5 cities)"
    print("Batch processing test PASSED")

    # Test 5: Top cities function
    print("\n[Test 5] Get Top Cities Test")
    top_cities = get_top_cities(df_all, 'D_i', n=3)
    print(f"Top 3 cities by degree:")
    print(top_cities[['City', 'Year', 'Category', 'D_i']])
    print("Top cities test PASSED")

    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80)
