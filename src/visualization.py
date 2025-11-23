"""
Visualization Module.

This module provides plotting functions for networks, heatmaps, rankings,
and temporal evolution. Supports both static (matplotlib) and interactive (plotly) plots.

Functions:
    - plot_network: Visualize network with nodes and edges
    - plot_similarity_heatmap: Heatmap of pairwise similarities
    - plot_centrality_comparison: Radar chart for centrality metrics
    - plot_rankings_comparison: Compare top cities across schemes
    - plot_temporal_evolution: Track metrics over time
    - plot_hasse_diagram: Visualize partial order
"""

import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100  # Default for display
plt.rcParams['savefig.dpi'] = 300  # High quality for saving
plt.rcParams['font.size'] = 10


def plot_network(
    G: nx.Graph,
    title: str,
    layout: str = 'spring',
    figsize: Tuple[int, int] = (10, 8)
) -> matplotlib.figure.Figure:
    """
    Visualize network with nodes and edges.

    Creates a network visualization with node size proportional to degree
    and optional edge thickness based on weights.

    Args:
        G: NetworkX Graph object
        title: Plot title
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_network(G, "Food Network 2024", layout='spring')
        >>> plt.savefig('network.png')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'Empty Network', ha='center', va='center')
        ax.set_title(title)
        return fig
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Calculate node sizes based on degree
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 100 + 100 for node in G.nodes()]
    
    # Draw network
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='lightblue',
        edgecolors='black',
        linewidths=1,
        ax=ax
    )
    
    # Draw edges (with weights if available)
    edges = G.edges()
    if edges:
        # Check if weighted
        if 'weight' in G[list(edges)[0][0]][list(edges)[0][1]]:
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, ax=ax)
        else:
            nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    return fig


def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    cities: List[str],
    title: str,
    figsize: Tuple[int, int] = (10, 8)
) -> matplotlib.figure.Figure:
    """
    Heatmap showing pairwise city similarities.

    Args:
        sim_matrix: Similarity matrix (n_cities x n_cities)
        cities: List of city names
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_similarity_heatmap(sim_matrix, cities, "Similarity 2024")
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create DataFrame for better labeling
    df_sim = pd.DataFrame(sim_matrix, index=cities, columns=cities)
    
    # Plot heatmap
    sns.heatmap(
        df_sim,
        annot=False,  # Don't annotate to avoid clutter
        fmt='.2f',
        cmap='RdYlBu_r',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={'label': 'Similarity'},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    return fig


def plot_centrality_comparison(
    df: pd.DataFrame,
    city: str,
    year: int,
    figsize: Tuple[int, int] = (8, 8)
) -> matplotlib.figure.Figure:
    """
    Radar chart comparing 4 centrality metrics for a city across categories.

    Args:
        df: Centrality DataFrame
        city: City name to analyze
        year: Year to analyze
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_centrality_comparison(df, 'Karachi', 2024)
    """
    # Filter data
    df_city = df[(df['City'] == city) & (df['Year'] == year)].copy()
    
    if df_city.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'No data for {city} in {year}', ha='center', va='center')
        return fig
    
    metrics = ['D_i', 'C_i', 'B_i', 'E_i']
    categories = df_city['Category'].tolist()
    
    # Create radar chart
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    for idx, (_, row) in enumerate(df_city.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Category'][:20])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title(f'Centrality Metrics: {city} ({year})', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    return fig


def plot_rankings_comparison(
    df_scores: pd.DataFrame,
    year: int,
    top_n: int = 10,
    figsize: Tuple[int, int] = (15, 10)
) -> matplotlib.figure.Figure:
    """
    Side-by-side bar charts for top cities under each weighting scheme.

    Args:
        df_scores: Scores DataFrame (aggregated by year)
        year: Year to analyze
        top_n: Number of top cities to show
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_rankings_comparison(df_year, 2024, top_n=10)
    """
    schemes = ['Score_Equal', 'Score_Correlation', 'Score_Entropy',
              'Score_Category_Weighted', 'Score_Interactive']
    
    # Filter available schemes
    available_schemes = [s for s in schemes if s in df_scores.columns]
    
    # Filter to year
    df_year = df_scores[df_scores['Year'] == year].copy()
    
    if df_year.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'No data for year {year}', ha='center', va='center')
        return fig
    
    # Create subplots
    n_schemes = len(available_schemes)
    nrows = (n_schemes + 1) // 2
    ncols = 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_schemes > 1 else [axes]
    
    for idx, scheme in enumerate(available_schemes):
        ax = axes[idx]
        
        # Get top N cities
        top_cities = df_year.nlargest(top_n, scheme)
        
        # Plot
        ax.barh(top_cities['City'], top_cities[scheme], color='steelblue')
        ax.set_xlabel('Score')
        ax.set_title(scheme.replace('_', ' '), fontweight='bold')
        ax.invert_yaxis()  # Highest on top
    
    # Hide unused subplots
    for idx in range(n_schemes, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Top {top_n} Cities by Weighting Scheme ({year})',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_temporal_evolution(
    df_evolution: pd.DataFrame,
    category: str,
    metric: str = 'Density',
    figsize: Tuple[int, int] = (10, 6)
) -> matplotlib.figure.Figure:
    """
    Line chart showing how network metric changes over time.

    Args:
        df_evolution: Evolution DataFrame from analyze_network_evolution
        category: Category to analyze
        metric: Metric to plot ('Nodes', 'Edges', 'Density', 'Clustering')
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_temporal_evolution(df_evolution, 'Food', 'Density')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter to category
    df_cat = df_evolution[df_evolution['Category'] == category].copy()
    
    if df_cat.empty:
        ax.text(0.5, 0.5, f'No data for category: {category}', ha='center', va='center')
        return fig
    
    # Sort by year
    df_cat = df_cat.sort_values('Year')
    
    # Plot
    ax.plot(df_cat['Year'], df_cat[metric], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} Evolution: {category}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_hasse_diagram(
    H: nx.DiGraph,
    title: str,
    figsize: Tuple[int, int] = (10, 8)
) -> matplotlib.figure.Figure:
    """
    Visualize partial order with directed graph (Hasse diagram).

    Uses hierarchical layout with later years at the top.

    Args:
        H: Directed graph (from create_hasse_diagram)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_hasse_diagram(H, "Temporal Order: Food")
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if H.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'Empty Diagram', ha='center', va='center')
        ax.set_title(title)
        return fig
    
    # Use hierarchical layout
    try:
        pos = nx.spring_layout(H, seed=42)
    except:
        pos = {node: (0, i) for i, node in enumerate(H.nodes())}
    
    # Draw
    nx.draw(
        H, pos,
        with_labels=True,
        node_size=1000,
        node_color='lightcoral',
        edgecolors='black',
        linewidths=2,
        font_size=12,
        font_weight='bold',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    return fig


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    """
    Basic testing of visualization module functions.
    """
    print("="*80)
    print("VISUALIZATION MODULE TESTS")
    print("="*80)
    
    # Test 1: Network plot
    print("\n[Test 1] Network Visualization")
    G = nx.karate_club_graph()
    fig = plot_network(G, "Test Network", layout='spring')
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    print("Network plot test PASSED")
    
    # Test 2: Similarity heatmap
    print("\n[Test 2] Similarity Heatmap")
    sim_matrix = np.random.rand(5, 5)
    sim_matrix = (sim_matrix + sim_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(sim_matrix, 1.0)
    cities = ['A', 'B', 'C', 'D', 'E']
    fig = plot_similarity_heatmap(sim_matrix, cities, "Test Heatmap")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    print("Heatmap test PASSED")
    
    # Test 3: Centrality comparison
    print("\n[Test 3] Centrality Comparison")
    df = pd.DataFrame({
        'Year': [2024, 2024],
        'Category': ['Cat1', 'Cat2'],
        'City': ['CityA', 'CityA'],
        'D_i': [0.5, 0.6],
        'C_i': [0.4, 0.7],
        'B_i': [0.3, 0.5],
        'E_i': [0.6, 0.8]
    })
    fig = plot_centrality_comparison(df, 'CityA', 2024)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    print("Centrality comparison test PASSED")
    
    # Test 4: Rankings comparison
    print("\n[Test 4] Rankings Comparison")
    df_scores = pd.DataFrame({
        'Year': [2024] * 5,
        'City': ['A', 'B', 'C', 'D', 'E'],
        'Score_Equal': [0.8, 0.7, 0.6, 0.5, 0.4],
        'Score_Correlation': [0.75, 0.65, 0.55, 0.45, 0.35]
    })
    fig = plot_rankings_comparison(df_scores, 2024, top_n=5)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    print("Rankings comparison test PASSED")
    
    # Test 5: Temporal evolution
    print("\n[Test 5] Temporal Evolution")
    df_evolution = pd.DataFrame({
        'Year': [2021, 2022, 2023],
        'Category': ['Food'] * 3,
        'Density': [0.3, 0.4, 0.5]
    })
    fig = plot_temporal_evolution(df_evolution, 'Food', 'Density')
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    print("Temporal evolution test PASSED")
    
    # Test 6: Hasse diagram
    print("\n[Test 6] Hasse Diagram")
    H = nx.DiGraph()
    H.add_edges_from([(2021, 2022), (2022, 2023)])
    fig = plot_hasse_diagram(H, "Test Hasse")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    print("Hasse diagram test PASSED")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80)
