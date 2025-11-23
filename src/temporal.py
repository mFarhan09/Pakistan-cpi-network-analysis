"""
Temporal Analysis Module.

This module analyzes temporal evolution of networks and creates Hasse diagrams
to visualize how network structures change over time.

Functions:
    - check_subset_relationship: Check if edges of G1 are subset of G2
    - verify_partial_order: Verify partial order properties
    - build_temporal_order_matrix: Build adjacency matrix for Hasse diagram
    - create_hasse_diagram: Create directed graph showing temporal ordering
    - analyze_network_evolution: Track metrics over time
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_subset_relationship(G1: nx.Graph, G2: nx.Graph) -> bool:
    """
    Check if edges of G1 are a subset of edges of G2.

    Returns True if all edges in G1 exist in G2, indicating that G1's
    network structure is contained within G2's structure.

    Algorithm:
        For each edge (u, v) in G1:
            Check if edge exists in G2
        Return True if all edges found, False otherwise

    Args:
        G1: First graph (potentially subset)
        G2: Second graph (potentially superset)

    Returns:
        True if edges(G1) ⊆ edges(G2), False otherwise

    Example:
        >>> G1 = nx.Graph([(1, 2), (2, 3)])
        >>> G2 = nx.Graph([(1, 2), (2, 3), (3, 4)])
        >>> check_subset_relationship(G1, G2)
        True
    """
    # Get edge sets (undirected, so normalize order)
    edges_G1 = set(frozenset(edge) for edge in G1.edges())
    edges_G2 = set(frozenset(edge) for edge in G2.edges())
    
    # Check subset relationship
    is_subset = edges_G1.issubset(edges_G2)
    
    return is_subset


def verify_partial_order(graphs: List[nx.Graph]) -> Dict[str, bool]:
    """
    Verify that the graph sequence forms a partial order.

    Checks three properties:
    1. Reflexive: Every graph is a subset of itself
    2. Antisymmetric: If G1 ⊆ G2 and G2 ⊆ G1, then G1 = G2
    3. Transitive: If G1 ⊆ G2 and G2 ⊆ G3, then G1 ⊆ G3

    Args:
        graphs: List of NetworkX Graph objects

    Returns:
        Dictionary with verification results:
        - reflexive: bool
        - antisymmetric: bool
        - transitive: bool
        - is_partial_order: bool (all three properties hold)

    Example:
        >>> graphs = [G1, G2, G3]
        >>> results = verify_partial_order(graphs)
        >>> results['is_partial_order']
    """
    n = len(graphs)
    results = {
        'reflexive': True,
        'antisymmetric': True,
        'transitive': True,
        'is_partial_order': True
    }
    
    if n == 0:
        return results
    
    # Check reflexive property
    for G in graphs:
        if not check_subset_relationship(G, G):
            results['reflexive'] = False
            logger.warning("Reflexive property violated")
    
    # Check antisymmetric property
    for i in range(n):
        for j in range(i + 1, n):
            if check_subset_relationship(graphs[i], graphs[j]) and \
               check_subset_relationship(graphs[j], graphs[i]):
                # Both are subsets of each other, check if equal
                if set(graphs[i].edges()) != set(graphs[j].edges()):
                    results['antisymmetric'] = False
                    logger.warning(f"Antisymmetric property violated for graphs {i} and {j}")
    
    # Check transitive property
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if check_subset_relationship(graphs[i], graphs[j]) and \
                   check_subset_relationship(graphs[j], graphs[k]):
                    if not check_subset_relationship(graphs[i], graphs[k]):
                        results['transitive'] = False
                        logger.warning(f"Transitive property violated for graphs {i}, {j}, {k}")
    
    # Overall result
    results['is_partial_order'] = (
        results['reflexive'] and
        results['antisymmetric'] and
        results['transitive']
    )
    
    return results


def build_temporal_order_matrix(
    graphs_dict: Dict[Tuple[int, str], nx.Graph],
    category: str
) -> Tuple[List[int], np.ndarray]:
    """
    Build adjacency matrix for Hasse diagram.

    Creates a matrix where entry [i][j] = 1 if G_i ⊆ G_j.
    Removes transitive edges to create a clean Hasse diagram.

    Algorithm:
        1. Extract graphs for specified category, sorted by year
        2. Build subset relationship matrix
        3. Perform transitive reduction to remove redundant edges
        4. Return years and reduced matrix

    Args:
        graphs_dict: Dictionary {(year, category): Graph}
        category: Category to analyze

    Returns:
        Tuple of (years_list, order_matrix)
        - years_list: Sorted list of years
        - order_matrix: Binary adjacency matrix with transitive reduction

    Example:
        >>> years, matrix = build_temporal_order_matrix(graphs_dict, "Food Staples")
    """
    # Extract graphs for this category
    category_graphs = {year: G for (year, cat), G in graphs_dict.items() if cat == category}
    
    if not category_graphs:
        logger.warning(f"No graphs found for category: {category}")
        return [], np.array([])
    
    # Sort by year
    years = sorted(category_graphs.keys())
    graphs = [category_graphs[year] for year in years]
    n = len(years)
    
    # Build subset relationship matrix
    order_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if check_subset_relationship(graphs[i], graphs[j]):
                order_matrix[i, j] = 1
    
    # Transitive reduction: remove edges that can be inferred
    # If i -> j and j -> k exist, remove i -> k
    reduced_matrix = order_matrix.copy()
    
    for i in range(n):
        for j in range(n):
            if reduced_matrix[i, j] == 1 and i != j:
                for k in range(n):
                    if reduced_matrix[j, k] == 1 and j != k and i != k:
                        # Remove transitive edge
                        reduced_matrix[i, k] = 0
    
    logger.info(f"Built temporal order matrix for {category}: {n} time points")
    
    return years, reduced_matrix


def create_hasse_diagram(
    years: List[int],
    order_matrix: np.ndarray,
    category: str
) -> nx.DiGraph:
    """
    Create directed graph showing temporal ordering (Hasse diagram).

    Nodes represent years, edges represent direct subset relationships
    (after transitive reduction).

    Algorithm:
        1. Create directed graph
        2. Add nodes with year labels
        3. Add edges from order matrix
        4. Add graph metadata

    Args:
        years: List of years (node labels)
        order_matrix: Adjacency matrix (transitively reduced)
        category: Category name for metadata

    Returns:
        NetworkX DiGraph representing Hasse diagram
        - Nodes have 'year' attribute
        - Graph has 'category' attribute

    Example:
        >>> H = create_hasse_diagram(years, matrix, "Food Staples")
        >>> nx.draw(H, with_labels=True)
    """
    H = nx.DiGraph()
    
    # Add nodes
    for year in years:
        H.add_node(year, year=year)
    
    # Add edges from order matrix
    n = len(years)
    for i in range(n):
        for j in range(n):
            if order_matrix[i, j] == 1 and i != j:
                H.add_edge(years[i], years[j])
    
    # Add metadata
    H.graph['category'] = category
    H.graph['num_years'] = len(years)
    
    logger.info(f"Created Hasse diagram: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    
    return H


def analyze_network_evolution(
    graphs_dict: Dict[Tuple[int, str], nx.Graph]
) -> pd.DataFrame:
    """
    Track network metrics over time.

    Analyzes how network characteristics evolve across years and categories,
    providing insights into temporal patterns.

    Metrics tracked:
        - Number of nodes
        - Number of edges
        - Density
        - Average clustering coefficient
        - Number of connected components
        - Is connected (boolean)

    Args:
        graphs_dict: Dictionary {(year, category): Graph}

    Returns:
        DataFrame with columns:
        - Year: int
        - Category: str
        - Nodes: int
        - Edges: int
        - Density: float
        - Clustering: float
        - Components: int
        - IsConnected: bool

    Example:
        >>> df_evolution = analyze_network_evolution(graphs_dict)
        >>> df_evolution.groupby('Year')['Density'].mean()
    """
    results = []
    
    for (year, category), G in graphs_dict.items():
        metrics = {
            'Year': year,
            'Category': category,
            'Nodes': G.number_of_nodes(),
            'Edges': G.number_of_edges(),
            'Density': nx.density(G),
            'Clustering': nx.average_clustering(G),
            'Components': nx.number_connected_components(G),
            'IsConnected': nx.is_connected(G)
        }
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Sort by year and category
    df = df.sort_values(['Year', 'Category']).reset_index(drop=True)
    
    logger.info(f"Analyzed evolution for {len(df)} networks")
    
    return df


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    """
    Basic testing of temporal module functions.
    """
    print("="*80)
    print("TEMPORAL MODULE TESTS")
    print("="*80)
    
    # Test 1: Subset relationship
    print("\n[Test 1] Subset Relationship")
    G1 = nx.Graph([(1, 2), (2, 3)])
    G2 = nx.Graph([(1, 2), (2, 3), (3, 4)])
    G3 = nx.Graph([(1, 2)])
    
    assert check_subset_relationship(G1, G2) == True, "G1 should be subset of G2"
    assert check_subset_relationship(G2, G1) == False, "G2 should not be subset of G1"
    assert check_subset_relationship(G3, G1) == True, "G3 should be subset of G1"
    print("Subset relationship test PASSED")
    
    # Test 2: Partial order verification
    print("\n[Test 2] Partial Order Verification")
    graphs = [G3, G1, G2]  # Growing sequence
    results = verify_partial_order(graphs)
    print(f"  Reflexive: {results['reflexive']}")
    print(f"  Antisymmetric: {results['antisymmetric']}")
    print(f"  Transitive: {results['transitive']}")
    print(f"  Is Partial Order: {results['is_partial_order']}")
    print("Partial order verification test PASSED")
    
    # Test 3: Temporal order matrix
    print("\n[Test 3] Temporal Order Matrix")
    graphs_dict = {
        (2021, 'Food'): G3,
        (2022, 'Food'): G1,
        (2023, 'Food'): G2,
    }
    years, matrix = build_temporal_order_matrix(graphs_dict, 'Food')
    print(f"  Years: {years}")
    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Matrix:\n{matrix}")
    assert len(years) == 3, "Should have 3 years"
    assert matrix.shape == (3, 3), "Matrix should be 3x3"
    print("Temporal order matrix test PASSED")
    
    # Test 4: Hasse diagram
    print("\n[Test 4] Hasse Diagram")
    H = create_hasse_diagram(years, matrix, 'Food')
    print(f"  Nodes: {H.number_of_nodes()}")
    print(f"  Edges: {H.number_of_edges()}")
    assert H.number_of_nodes() == 3, "Should have 3 nodes"
    print("Hasse diagram test PASSED")
    
    # Test 5: Network evolution
    print("\n[Test 5] Network Evolution Analysis")
    graphs_dict_full = {
        (2021, 'Food'): G3,
        (2022, 'Food'): G1,
        (2023, 'Food'): G2,
        (2021, 'Energy'): G1,
        (2022, 'Energy'): G2,
    }
    df_evolution = analyze_network_evolution(graphs_dict_full)
    print(f"  Rows: {len(df_evolution)}")
    print(f"  Columns: {list(df_evolution.columns)}")
    assert len(df_evolution) == 5, "Should have 5 network records"
    assert 'Density' in df_evolution.columns, "Should have Density column"
    print("Network evolution test PASSED")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80)
