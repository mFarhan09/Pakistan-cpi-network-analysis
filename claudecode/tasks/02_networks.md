# Task 2: Network Construction Module

## Objective
Build `src/networks.py` that creates NetworkX graphs from similarity matrices using threshold-based edge creation.

## Context
We have similarity matrices showing how similar each city pair is. Now we need to:
1. Apply a threshold (τ) to determine which similarities are "strong enough" to create edges
2. Build NetworkX graphs G_{y,c} for each year-category combination
3. Support both weighted and unweighted graphs
4. Manage multiple graphs efficiently

## Requirements

### Function 1: `create_adjacency_matrix(sim_matrix: np.ndarray, threshold: float) -> np.ndarray`

**Purpose:** Convert similarity matrix to binary adjacency matrix using threshold

**Algorithm:**
```
1. Create binary matrix: (sim_matrix >= threshold).astype(int)
2. Remove self-loops: np.fill_diagonal(result, 0)
3. Return adjacency matrix
```

**Input:**
- sim_matrix: Similarity matrix (n_cities × n_cities)
- threshold: Float between 0 and 1 (τ parameter)

**Output:**
- Binary adjacency matrix (0 or 1)
- 1 means edge exists (similarity >= threshold)
- 0 means no edge

**Validation:**
- Ensure diagonal is all zeros
- Check threshold is valid (0 <= τ <= 1)
- Matrix should be symmetric

---

### Function 2: `create_graph_from_adjacency(adj_matrix: np.ndarray, cities: List[str], sim_matrix: Optional[np.ndarray] = None) -> nx.Graph`

**Purpose:** Create NetworkX graph from adjacency matrix

**Algorithm:**
```
1. Create graph from adjacency matrix: nx.from_numpy_array(adj_matrix)
2. Map numeric node IDs to city names
3. If sim_matrix provided, add edge weights
4. Return undirected graph
```

**Input:**
- adj_matrix: Binary adjacency matrix
- cities: List of city names (for node labels)
- sim_matrix: Optional similarity matrix for edge weights

**Output:**
- NetworkX Graph object
- Nodes labeled with city names
- Edges optionally weighted by similarity

**Node Attributes:**
- city_name: String name of city
- node_id: Original numeric ID

**Edge Attributes (if sim_matrix provided):**
- weight: Similarity score
- similarity: Copy of similarity score

---

### Function 3: `build_network(df: pd.DataFrame, year: int, category: str, threshold: float, weighted: bool = False) -> nx.Graph`

**Purpose:** Main function to build complete network from data

**Algorithm:**
```
1. Import similarity module
2. Get similarity matrix: similarity.get_similarity_matrix(df, year, category)
3. Create adjacency matrix with threshold
4. Build NetworkX graph
5. Add metadata to graph
6. Return graph
```

**Input:**
- df: Preprocessed DataFrame
- year: Year to analyze
- category: Category to analyze  
- threshold: Similarity threshold (τ)
- weighted: If True, add similarity as edge weight

**Output:**
- NetworkX Graph with:
  - Nodes = cities
  - Edges = similar cities (above threshold)
  - Graph attributes: year, category, threshold, num_edges, density

**Graph Metadata:**
```python
G.graph['year'] = year
G.graph['category'] = category
G.graph['threshold'] = threshold
G.graph['num_nodes'] = G.number_of_nodes()
G.graph['num_edges'] = G.number_of_edges()
G.graph['density'] = nx.density(G)
G.graph['weighted'] = weighted
```

---

### Function 4: `build_all_networks(df: pd.DataFrame, years: List[int], categories: List[str], threshold: float) -> Dict[Tuple[int, str], nx.Graph]`

**Purpose:** Build networks for all year-category combinations

**Algorithm:**
```
1. Get unique years from config (if not provided)
2. Get unique categories from data
3. Loop through all year-category pairs
4. Build network for each combination
5. Store in dictionary keyed by (year, category) tuple
6. Log progress and statistics
```

**Input:**
- df: Preprocessed DataFrame
- years: List of years to analyze
- categories: List of categories to analyze
- threshold: Similarity threshold

**Output:**
- Dictionary: {(year, category): Graph}
- Only includes graphs with at least 1 edge

**Progress Logging:**
- Print total combinations to process
- Show progress: "Processing 5/21: 2024, Food Staples..."
- Log network statistics for each graph

---

### Function 5: `get_network_statistics(G: nx.Graph) -> Dict[str, any]`

**Purpose:** Calculate comprehensive statistics for a network

**Algorithm:**
```
1. Basic stats: nodes, edges, density
2. Connectivity: is_connected, num_components
3. If connected: diameter, average_path_length
4. Clustering coefficient
5. Return as dictionary
```

**Output:**
```python
{
    'num_nodes': int,
    'num_edges': int,
    'density': float,
    'is_connected': bool,
    'num_components': int,
    'avg_degree': float,
    'clustering_coefficient': float,
    # If connected:
    'diameter': int,
    'avg_path_length': float
}
```

---

### Function 6: `print_network_summary(graphs_dict: Dict[Tuple[int, str], nx.Graph])`

**Purpose:** Display summary table of all networks

**Algorithm:**
```
1. Create table with columns: Year, Category, Nodes, Edges, Density, Connected
2. Sort by year then category
3. Print formatted table
4. Show totals at bottom
```

**Output Example:**
```
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
```

---

## Implementation Details

### Imports Needed
```python
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
from src.config import CITIES, CATEGORIES, TAU, ANALYSIS_YEARS
from src.similarity import get_similarity_matrix
```

### Performance Considerations
- Cache similarity matrices if building multiple networks with different thresholds
- Use sparse matrices for large networks (if needed)
- Parallel processing for multiple networks (optional)

### Logging
- Log each network as it's created
- Warn if network is empty (no edges)
- Report total processing time

---

## Testing Criteria

1. **Threshold Test**: Higher threshold → fewer edges
2. **Symmetry Test**: Undirected graphs should have symmetric adjacency
3. **Node Count Test**: All graphs should have 17 nodes (all cities)
4. **Metadata Test**: Graph attributes correctly set
5. **Edge Weight Test**: If weighted=True, edges have 'weight' attribute

---

## Example Usage

```python
from src.networks import build_network, build_all_networks, print_network_summary
from src.preprocessing import run_preprocessing_pipeline

# Load data
df = run_preprocessing_pipeline()

# Build single network
G = build_network(
    df,
    year=2024,
    category='1. Food Staples & Grains',
    threshold=0.75,
    weighted=True
)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.3f}")

# Build all networks
graphs = build_all_networks(df, [2023, 2024], ['1. Food Staples & Grains'], 0.75)
print_network_summary(graphs)
```

---

## Success Criteria

✅ Creates valid NetworkX graphs
✅ Applies threshold correctly
✅ Node labels are city names
✅ Supports both weighted and unweighted graphs
✅ Handles empty networks gracefully
✅ Efficient for batch processing
✅ Comprehensive statistics and logging

---

## Files to Create
- `src/networks.py`

## Files to Reference
- `src/similarity.py` (for similarity matrices)
- `src/config.py` (for CITIES, CATEGORIES, TAU)
