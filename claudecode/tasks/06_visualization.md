# Task 6: Visualization Module

## Objective
Build `src/visualization.py` with all plotting functions for networks, heatmaps, and rankings.

## Core Functions

### 1. `plot_network(G: nx.Graph, title: str, layout: str = 'spring') -> matplotlib.figure.Figure`
- Visualize network with nodes and edges
- Support different layouts: spring, circular, kamada_kawai
- Node size proportional to degree
- Edge thickness by weight (if weighted)

### 2. `plot_similarity_heatmap(sim_matrix: np.ndarray, cities: List[str], title: str) -> matplotlib.figure.Figure`
- Heatmap showing pairwise city similarities
- Annotate with similarity values
- Use diverging colormap (e.g., RdYlBu)

### 3. `plot_centrality_comparison(df: pd.DataFrame, city: str, year: int) -> matplotlib.figure.Figure`
- Spider/radar chart comparing 4 centrality metrics for a city
- Show across different categories

### 4. `plot_rankings_comparison(df_scores: pd.DataFrame, year: int) -> matplotlib.figure.Figure`
- Side-by-side bar charts for top 10 cities under each weighting scheme
- 2x2 or 2x3 subplot layout

### 5. `plot_temporal_evolution(graphs_dict: Dict, category: str, metric: str) -> matplotlib.figure.Figure`
- Line chart showing how network metric changes over time
- Metrics: nodes, edges, density, clustering

### 6. `plot_hasse_diagram(H: nx.DiGraph, title: str) -> matplotlib.figure.Figure`
- Visualize partial order with directed graph
- Hierarchical layout (top = later years)

### 7. Interactive Plotly versions of above
- `plot_network_interactive()`
- `plot_heatmap_interactive()`
- etc.

## Styling
- Use consistent color scheme from config
- Professional fonts and labels
- Clear titles and legends
- Save with high DPI (300)

## Success Criteria
✅ All plots are clear and professional
✅ Support both static (matplotlib) and interactive (plotly)
✅ Consistent styling across plots
✅ Handle edge cases (empty networks, missing data)

## Files to Create
- `src/visualization.py`
