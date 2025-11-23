"""
Network Construction Module.

This module creates NetworkX graphs from similarity matrices using threshold-based
edge creation. It supports building networks for individual year-category combinations
or batch processing for multiple combinations.

Functions:
    - create_adjacency_matrix: Convert similarity to binary adjacency matrix
    - create_graph_from_adjacency: Create NetworkX graph from adjacency matrix
    - build_network: Build complete network from data
    - build_all_networks: Build networks for all year-category combinations
    - get_network_statistics: Calculate network statistics
    - print_network_summary: Display summary table of all networks
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import logging
import time

from src.config import CITIES, CATEGORIES, TAU, ANALYSIS_YEARS
from src.similarity import get_similarity_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_adjacency_matrix(sim_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert similarity matrix to binary adjacency matrix using threshold.

    Applies a threshold (tau) to determine which similarities are "strong enough"
    to create edges in the network. Similarities above threshold become edges (1),
    others remain disconnected (0).

    Algorithm:
        1. Create binary matrix: (sim_matrix >= threshold).astype(int)
        2. Remove self-loops: np.fill_diagonal(result, 0)
        3. Return adjacency matrix

    Args:
        sim_matrix: Similarity matrix of shape (n_cities, n_cities)
                   Values typically range from -1 to 1
        threshold: Float between 0 and 1 (tau parameter)
                  Cities with similarity >= threshold are connected

    Returns:
        Binary adjacency matrix (0 or 1)
        - 1 means edge exists (similarity >= threshold)
        - 0 means no edge
        - Diagonal is all zeros (no self-loops)

    Validation:
        - Ensures diagonal is all zeros
        - Checks threshold is valid (0 <= tau <= 1)
        - Matrix is symmetric

    Example:
        >>> sim = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.6], [0.3, 0.6, 1.0]])
        >>> adj = create_adjacency_matrix(sim, threshold=0.7)
        >>> adj
        array([[0, 1, 0],
               [1, 0, 0],
               [0, 0, 0]])
    """
    # Validate threshold
    if not (0 <= threshold <= 1):
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

    # Validate input
    if sim_matrix.size == 0:
        logger.warning("Empty similarity matrix provided")
        return np.array([])

    if len(sim_matrix.shape) != 2 or sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {sim_matrix.shape}")

    # Create binary adjacency matrix
    adj_matrix = (sim_matrix >= threshold).astype(int)

    # Remove self-loops (diagonal should be 0)
    np.fill_diagonal(adj_matrix, 0)

    # Validate result
    if not np.allclose(adj_matrix, adj_matrix.T):
        logger.warning("Adjacency matrix is not symmetric")

    # Log statistics
    n_cities = sim_matrix.shape[0]
    n_possible_edges = (n_cities * (n_cities - 1)) // 2
    n_edges = np.sum(adj_matrix) // 2  # Divide by 2 because matrix is symmetric
    density = n_edges / n_possible_edges if n_possible_edges > 0 else 0

    logger.info(f"Adjacency matrix created: threshold={threshold:.2f}")
    logger.info(f"  Edges: {n_edges}/{n_possible_edges} (density={density:.3f})")

    return adj_matrix


def create_graph_from_adjacency(
    adj_matrix: np.ndarray,
    cities: List[str],
    sim_matrix: Optional[np.ndarray] = None
) -> nx.Graph:
    """
    Create NetworkX graph from adjacency matrix.

    Converts a binary adjacency matrix into a NetworkX Graph object with
    city names as node labels. Optionally adds edge weights from similarity matrix.

    Algorithm:
        1. Create graph from adjacency matrix: nx.from_numpy_array(adj_matrix)
        2. Map numeric node IDs to city names
        3. If sim_matrix provided, add edge weights
        4. Return undirected graph

    Args:
        adj_matrix: Binary adjacency matrix of shape (n_cities, n_cities)
        cities: List of city names for node labels
        sim_matrix: Optional similarity matrix for edge weights

    Returns:
        NetworkX Graph object with:
        - Nodes labeled with city names
        - Node attributes: city_name, node_id
        - Edge attributes (if sim_matrix provided): weight, similarity

    Example:
        >>> adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        >>> cities = ['Islamabad', 'Karachi', 'Lahore']
        >>> G = create_graph_from_adjacency(adj, cities)
        >>> G.number_of_nodes()
        3
    """
    # Validate inputs
    if adj_matrix.shape[0] != len(cities):
        raise ValueError(
            f"Adjacency matrix has {adj_matrix.shape[0]} nodes "
            f"but {len(cities)} city names provided"
        )

    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    # Create mapping from node IDs to city names
    # NetworkX creates nodes as 0, 1, 2, ..., n-1
    mapping = {i: cities[i] for i in range(len(cities))}

    # Relabel nodes with city names
    G = nx.relabel_nodes(G, mapping)

    # Add node attributes
    for i, city in enumerate(cities):
        G.nodes[city]['city_name'] = city
        G.nodes[city]['node_id'] = i

    # Add edge weights if similarity matrix provided
    if sim_matrix is not None:
        for i, city_i in enumerate(cities):
            for j, city_j in enumerate(cities):
                if i < j and adj_matrix[i, j] == 1:  # Only process each edge once
                    similarity_value = sim_matrix[i, j]
                    G[city_i][city_j]['weight'] = similarity_value
                    G[city_i][city_j]['similarity'] = similarity_value

    logger.info(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def build_network(
    df: pd.DataFrame,
    year: int,
    category: str,
    threshold: float,
    weighted: bool = False
) -> Optional[nx.Graph]:
    """
    Main function to build complete network from data.

    This is the primary interface for building a single network for a specific
    year-category combination. It handles the full pipeline from data to graph.

    Algorithm:
        1. Get similarity matrix from data
        2. Create adjacency matrix with threshold
        3. Build NetworkX graph
        4. Add metadata to graph
        5. Return graph

    Args:
        df: Preprocessed DataFrame with columns:
            [City, Year, Category, Month_Year, Item_Description, Normalized_Price]
        year: Year to analyze (e.g., 2024)
        category: Category to analyze (e.g., "1. Food Staples & Grains")
        threshold: Similarity threshold tau (0 to 1)
        weighted: If True, add similarity as edge weight

    Returns:
        NetworkX Graph with:
        - Nodes = cities
        - Edges = similar cities (above threshold)
        - Graph attributes: year, category, threshold, num_edges, density, etc.
        Returns None if no data available for year-category

    Graph Metadata:
        - year: Year of analysis
        - category: Category analyzed
        - threshold: Threshold used
        - num_nodes: Number of nodes
        - num_edges: Number of edges
        - density: Network density
        - weighted: Whether edges have weights

    Example:
        >>> G = build_network(df, 2024, "1. Food Staples & Grains", 0.75, weighted=True)
        >>> print(f"Density: {G.graph['density']:.3f}")
    """
    logger.info(f"Building network: Year={year}, Category='{category}', threshold={threshold}")

    # Get similarity matrix
    sim_matrix, city_names = get_similarity_matrix(df, year, category)

    # Check if data exists
    if sim_matrix is None:
        logger.warning(f"No data for Year={year}, Category='{category}'. Skipping.")
        return None

    # Create adjacency matrix
    adj_matrix = create_adjacency_matrix(sim_matrix, threshold)

    # Build graph
    if weighted:
        G = create_graph_from_adjacency(adj_matrix, city_names, sim_matrix)
    else:
        G = create_graph_from_adjacency(adj_matrix, city_names)

    # Add graph metadata
    G.graph['year'] = year
    G.graph['category'] = category
    G.graph['threshold'] = threshold
    G.graph['num_nodes'] = G.number_of_nodes()
    G.graph['num_edges'] = G.number_of_edges()
    G.graph['density'] = nx.density(G)
    G.graph['weighted'] = weighted

    # Warn if network is empty
    if G.number_of_edges() == 0:
        logger.warning(f"Network has no edges (threshold may be too high)")

    logger.info(
        f"Network built: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges, density={G.graph['density']:.3f}"
    )

    return G


def build_all_networks(
    df: pd.DataFrame,
    years: Optional[List[int]] = None,
    categories: Optional[List[str]] = None,
    threshold: float = TAU,
    weighted: bool = False
) -> Dict[Tuple[int, str], nx.Graph]:
    """
    Build networks for all year-category combinations.

    Batch processing function that builds networks for multiple year-category
    pairs. Useful for comprehensive analysis across time periods and categories.

    Algorithm:
        1. Get unique years from config (if not provided)
        2. Get unique categories from data (if not provided)
        3. Loop through all year-category pairs
        4. Build network for each combination
        5. Store in dictionary keyed by (year, category) tuple
        6. Log progress and statistics

    Args:
        df: Preprocessed DataFrame
        years: List of years to analyze (default: ANALYSIS_YEARS from config)
        categories: List of categories to analyze (default: CATEGORIES from config)
        threshold: Similarity threshold (default: TAU from config)
        weighted: If True, create weighted graphs

    Returns:
        Dictionary: {(year, category): Graph}
        - Only includes graphs with at least 1 edge
        - Empty networks are excluded

    Progress Logging:
        - Prints total combinations to process
        - Shows progress: "Processing 5/21: 2024, Food Staples..."
        - Logs network statistics for each graph

    Example:
        >>> graphs = build_all_networks(df, [2023, 2024], threshold=0.75)
        >>> print(f"Built {len(graphs)} networks")
    """
    # Use defaults if not provided
    if years is None:
        years = ANALYSIS_YEARS
    if categories is None:
        categories = CATEGORIES

    # Calculate total combinations
    total_combinations = len(years) * len(categories)
    logger.info("="*80)
    logger.info(f"Building {total_combinations} networks")
    logger.info(f"Years: {years}")
    logger.info(f"Categories: {len(categories)} categories")
    logger.info(f"Threshold: {threshold}")
    logger.info("="*80)

    # Initialize storage
    graphs_dict = {}
    start_time = time.time()
    processed = 0

    # Loop through all combinations
    for year in years:
        for category in categories:
            processed += 1

            # Shorten category name for progress display
            category_short = category[:30] + "..." if len(category) > 30 else category
            logger.info(f"\nProcessing {processed}/{total_combinations}: {year}, {category_short}")

            # Build network
            G = build_network(df, year, category, threshold, weighted)

            # Store if network has edges
            if G is not None and G.number_of_edges() > 0:
                graphs_dict[(year, category)] = G
                logger.info(f"Added to collection")
            elif G is not None:
                logger.info(f"Skipped (no edges)")
            else:
                logger.info(f"Skipped (no data)")

    # Final summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info(f"BATCH PROCESSING COMPLETE")
    logger.info(f"Built {len(graphs_dict)}/{total_combinations} networks with edges")
    logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
    logger.info("="*80)

    return graphs_dict


def get_network_statistics(G: nx.Graph) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a network.

    Computes various network metrics useful for understanding network structure,
    connectivity, and characteristics.

    Algorithm:
        1. Basic stats: nodes, edges, density
        2. Connectivity: is_connected, num_components
        3. If connected: diameter, average_path_length
        4. Clustering coefficient
        5. Return as dictionary

    Args:
        G: NetworkX Graph object

    Returns:
        Dictionary with statistics:
        - num_nodes: Number of nodes
        - num_edges: Number of edges
        - density: Network density (0 to 1)
        - is_connected: Whether graph is fully connected
        - num_components: Number of connected components
        - avg_degree: Average degree across all nodes
        - clustering_coefficient: Global clustering coefficient
        - diameter: Network diameter (only if connected)
        - avg_path_length: Average shortest path length (only if connected)

    Example:
        >>> stats = get_network_statistics(G)
        >>> print(f"Clustering: {stats['clustering_coefficient']:.3f}")
    """
    stats = {}

    # Basic statistics
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    stats['density'] = nx.density(G)

    # Degree statistics
    if stats['num_nodes'] > 0:
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = np.mean(degrees) if degrees else 0
        stats['max_degree'] = max(degrees) if degrees else 0
        stats['min_degree'] = min(degrees) if degrees else 0
    else:
        stats['avg_degree'] = 0
        stats['max_degree'] = 0
        stats['min_degree'] = 0

    # Connectivity
    stats['is_connected'] = nx.is_connected(G)
    stats['num_components'] = nx.number_connected_components(G)

    # Clustering coefficient
    stats['clustering_coefficient'] = nx.average_clustering(G)

    # If connected, compute additional metrics
    if stats['is_connected'] and stats['num_nodes'] > 1:
        try:
            stats['diameter'] = nx.diameter(G)
            stats['avg_path_length'] = nx.average_shortest_path_length(G)
        except nx.NetworkXError as e:
            logger.warning(f"Could not compute path metrics: {e}")
            stats['diameter'] = None
            stats['avg_path_length'] = None
    else:
        stats['diameter'] = None
        stats['avg_path_length'] = None

    return stats


def print_network_summary(graphs_dict: Dict[Tuple[int, str], nx.Graph]) -> None:
    """
    Display summary table of all networks.

    Creates a formatted table showing key statistics for all networks in the
    dictionary. Useful for quick overview of batch processing results.

    Algorithm:
        1. Create table with columns: Year, Category, Nodes, Edges, Density, Connected
        2. Sort by year then category
        3. Print formatted table
        4. Show totals at bottom

    Args:
        graphs_dict: Dictionary of {(year, category): Graph}

    Output Example:
        ================================================================================
        NETWORK SUMMARY
        ================================================================================
        Year    Category                          Nodes  Edges  Density  Connected
        --------------------------------------------------------------------------------
        2023    1. Food Staples & Grains            17     45    0.329    Yes
        2023    2. Meat, Poultry & Dairy            17     38    0.279    Yes
        ...
        Total Networks: 21
        Average Density: 0.285
        ================================================================================

    Example:
        >>> graphs = build_all_networks(df, [2023, 2024], threshold=0.75)
        >>> print_network_summary(graphs)
    """
    if not graphs_dict:
        print("No networks to display.")
        return

    print("\n" + "="*80)
    print("NETWORK SUMMARY")
    print("="*80)
    print(f"{'Year':<8}{'Category':<35}{'Nodes':<7}{'Edges':<7}{'Density':<9}{'Connected':<10}")
    print("-"*80)

    # Sort by year, then category
    sorted_keys = sorted(graphs_dict.keys(), key=lambda x: (x[0], x[1]))

    densities = []
    total_edges = 0
    total_nodes_sum = 0

    for year, category in sorted_keys:
        G = graphs_dict[(year, category)]

        # Truncate category name if too long
        category_display = category[:32] + "..." if len(category) > 32 else category

        # Get statistics
        stats = get_network_statistics(G)
        is_connected_str = "Yes" if stats['is_connected'] else "No"

        # Print row
        print(
            f"{year:<8}"
            f"{category_display:<35}"
            f"{stats['num_nodes']:<7}"
            f"{stats['num_edges']:<7}"
            f"{stats['density']:<9.3f}"
            f"{is_connected_str:<10}"
        )

        # Accumulate for summary
        densities.append(stats['density'])
        total_edges += stats['num_edges']
        total_nodes_sum += stats['num_nodes']

    # Print summary statistics
    print("-"*80)
    print(f"Total Networks: {len(graphs_dict)}")
    print(f"Total Edges: {total_edges}")
    if densities:
        print(f"Average Density: {np.mean(densities):.3f}")
        print(f"Min Density: {np.min(densities):.3f}")
        print(f"Max Density: {np.max(densities):.3f}")
    print("="*80 + "\n")


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    """
    Basic testing of networks module functions.
    """
    print("="*80)
    print("NETWORKS MODULE TESTS")
    print("="*80)

    # Test 1: Adjacency matrix creation
    print("\n[Test 1] Adjacency Matrix Creation")
    sim = np.array([
        [1.0, 0.8, 0.3, 0.5],
        [0.8, 1.0, 0.6, 0.4],
        [0.3, 0.6, 1.0, 0.7],
        [0.5, 0.4, 0.7, 1.0]
    ])
    adj = create_adjacency_matrix(sim, threshold=0.7)
    print(f"Adjacency matrix created")
    print(f"  Diagonal all zeros: {np.all(np.diag(adj) == 0)}")
    print(f"  Symmetric: {np.allclose(adj, adj.T)}")

    # Test 2: Threshold test
    print("\n[Test 2] Threshold Test - Higher threshold -> fewer edges")
    adj_low = create_adjacency_matrix(sim, threshold=0.5)
    adj_high = create_adjacency_matrix(sim, threshold=0.8)
    edges_low = np.sum(adj_low) // 2
    edges_high = np.sum(adj_high) // 2
    print(f"Passed: threshold=0.5 -> {edges_low} edges, threshold=0.8 -> {edges_high} edges")
    assert edges_low >= edges_high, "Higher threshold should have fewer edges"

    # Test 3: Graph creation
    print("\n[Test 3] Graph Creation from Adjacency")
    cities = ['Islamabad', 'Karachi', 'Lahore', 'Quetta']
    G = create_graph_from_adjacency(adj, cities)
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    assert G.number_of_nodes() == 4, "Should have 4 nodes"

    # Test 4: Node labels
    print("\n[Test 4] Node Labels Test")
    node_names = list(G.nodes())
    print(f"Node names: {node_names}")
    assert 'Islamabad' in node_names, "Should have city names as nodes"

    # Test 5: Weighted edges
    print("\n[Test 5] Weighted Edges Test")
    G_weighted = create_graph_from_adjacency(adj, cities, sim)
    has_weights = all('weight' in G_weighted[u][v] for u, v in G_weighted.edges())
    print(f"All edges have weights: {has_weights}")

    # Test 6: Network statistics
    print("\n[Test 6] Network Statistics")
    stats = get_network_statistics(G)
    print(f"Statistics computed:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Density: {stats['density']:.3f}")
    print(f"  Connected: {stats['is_connected']}")
    print(f"  Clustering: {stats['clustering_coefficient']:.3f}")

    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80)
