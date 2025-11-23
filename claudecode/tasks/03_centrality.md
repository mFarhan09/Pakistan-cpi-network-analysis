# Task 3: Centrality Analysis Module

## Objective
Build `src/centrality.py` that calculates four centrality metrics for all cities in all networks.

## Context
We have NetworkX graphs G_{y,c} for different year-category combinations. Now we need to:
1. Calculate 4 centrality measures for each city in each network
2. Handle disconnected graphs gracefully
3. Return results in a structured DataFrame ready for weighting analysis

## Requirements

### The 4 Centrality Metrics

#### 1. Degree Centrality (D_i)
**What it measures:** Number of direct connections
**Interpretation:** How many cities share similar price patterns?
**Formula:** Degree / (N-1) where N = total nodes
**Range:** [0, 1]

#### 2. Closeness Centrality (C_i)
**What it measures:** Average distance to all other nodes
**Interpretation:** How quickly does this city's price signal reach others?
**Formula:** (N-1) / sum(shortest_path_distances)
**Range:** [0, 1]
**Note:** Only defined for connected components

#### 3. Betweenness Centrality (B_i)
**What it measures:** How often this node lies on shortest paths
**Interpretation:** Is this city a "bridge" between regions?
**Formula:** Fraction of shortest paths passing through node
**Range:** [0, 1]

#### 4. Eigenvector Centrality (E_i)
**What it measures:** Influence based on connections to influential nodes
**Interpretation:** Is this city connected to other important cities?
**Formula:** Eigenvector of adjacency matrix (power iteration)
**Range:** [0, 1]
**Note:** May fail to converge for some graphs

---

### Function 1: `calculate_single_network_centrality(G: nx.Graph) -> pd.DataFrame`

**Purpose:** Calculate all 4 centralities for one network

**Algorithm:**
```
1. Check if graph is empty (no nodes) → return empty DataFrame
2. Calculate degree centrality: nx.degree_centrality(G)
3. Calculate closeness centrality: nx.closeness_centrality(G)
4. Calculate betweenness centrality: nx.betweenness_centrality(G)
5. Try eigenvector centrality: nx.eigenvector_centrality(G, max_iter=1000)
   - If fails: assign zeros or fallback to degree centrality
6. Combine into DataFrame with columns: City, D_i, C_i, B_i, E_i
7. Return DataFrame
```

**Input:**
- G: NetworkX Graph

**Output:**
- DataFrame with columns: [City, D_i, C_i, B_i, E_i]
- One row per city in the graph

**Edge Cases:**
- **Empty graph**: Return empty DataFrame
- **Single node**: Return zeros for all metrics
- **Disconnected graph**: Closeness handled per component
- **Eigenvector fails**: Log warning, use zeros

---

### Function 2: `calculate_all_centralities(graphs_dict: Dict[Tuple[int, str], nx.Graph]) -> pd.DataFrame`

**Purpose:** Calculate centralities for all networks and combine

**Algorithm:**
```
1. Initialize empty list for results
2. For each (year, category), graph in graphs_dict:
   a. Calculate centralities
   b. Add year and category columns
   c. Append to results list
3. Concatenate all DataFrames
4. Reset index
5. Return long-format DataFrame
```

**Input:**
- graphs_dict: Dictionary {(year, category): Graph}

**Output:**
- Long-format DataFrame with columns:
  - Year: int
  - Category: str
  - City: str
  - D_i: float (degree centrality)
  - C_i: float (closeness centrality)
  - B_i: float (betweenness centrality)
  - E_i: float (eigenvector centrality)

**Progress Tracking:**
- Print progress every 5 networks
- Show percentage complete
- Log any errors encountered

---

### Function 3: `normalize_centralities(df: pd.DataFrame) -> pd.DataFrame`

**Purpose:** Optionally normalize centralities within each year-category group

**Algorithm:**
```
1. For each year-category group:
   a. Min-max normalize each centrality metric
   b. Formula: (x - min) / (max - min)
   c. Handle case where max = min (set to 0.5)
2. Add normalized columns: D_i_norm, C_i_norm, B_i_norm, E_i_norm
3. Keep original columns as well
```

**Input:**
- df: DataFrame from calculate_all_centralities

**Output:**
- Same DataFrame with added normalized columns

**Note:** This is optional - some weighting schemes work better with raw scores

---

### Function 4: `get_top_cities(df: pd.DataFrame, metric: str, n: int = 10, year: Optional[int] = None, category: Optional[str] = None) -> pd.DataFrame`

**Purpose:** Convenience function to get top cities by a specific metric

**Algorithm:**
```
1. Filter by year and/or category if provided
2. Sort by specified metric (descending)
3. Return top n rows
```

**Input:**
- df: Centrality DataFrame
- metric: One of ['D_i', 'C_i', 'B_i', 'E_i']
- n: Number of top cities to return
- year: Optional year filter
- category: Optional category filter

**Output:**
- DataFrame with top cities for that metric

---

### Function 5: `print_centrality_summary(df: pd.DataFrame)`

**Purpose:** Display summary statistics of centrality distributions

**Algorithm:**
```
1. Calculate statistics for each metric:
   - Mean, Median, Std, Min, Max
2. Count cities with zero centrality
3. Identify cities with highest values
4. Print formatted table
```

**Output Example:**
```
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
Top Cities by Betweenness: Multan (0.456), Faisalabad (0.389), Rawalpindi (0.321)
================================================================================
```

---

## Implementation Details

### Imports Needed
```python
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, List
import logging
import warnings
```

### Error Handling
- Wrap eigenvector calculation in try-except
- Handle disconnected graphs for closeness centrality
- Log warnings for failed calculations
- Never crash - always return some result

### Performance
- NetworkX functions are already optimized
- For large networks, consider using parallel processing
- Cache results to avoid recalculation

### Logging
```python
logger.info(f"Calculating centralities for {len(graphs_dict)} networks")
logger.warning(f"Eigenvector centrality failed for {year}-{category}")
logger.info(f"Completed: {len(df)} city-network combinations")
```

---

## Testing Criteria

1. **Complete Network Test**: Well-connected graph should have high centralities
2. **Star Network Test**: Center node should have max centrality
3. **Isolated Node Test**: Isolated nodes should have zero centrality
4. **Symmetry Test**: Isomorphic positions should have same centrality
5. **Range Test**: All values between 0 and 1

---

## Example Usage

```python
from src.centrality import calculate_all_centralities, print_centrality_summary, get_top_cities
from src.networks import build_all_networks
from src.preprocessing import run_preprocessing_pipeline

# Load data and build networks
df_data = run_preprocessing_pipeline()
graphs = build_all_networks(df_data, [2023, 2024], threshold=0.75)

# Calculate centralities
df_centrality = calculate_all_centralities(graphs)

# Display summary
print_centrality_summary(df_centrality)

# Get top cities by degree centrality in 2024
top_cities = get_top_cities(
    df_centrality,
    metric='D_i',
    n=10,
    year=2024
)
print(top_cities)

# Save results
df_centrality.to_csv('data/results/centrality_scores.csv', index=False)
```

---

## Success Criteria

✅ Calculates all 4 centrality metrics correctly
✅ Handles disconnected graphs without errors
✅ Handles eigenvector convergence failures gracefully
✅ Returns properly structured DataFrame
✅ All centrality values in range [0, 1]
✅ Results are reproducible
✅ Efficient for batch processing
✅ Comprehensive logging and error messages

---

## Mathematical Validation

To verify correctness, test on known graph structures:

**Complete Graph K_5:**
- All centralities should be 1.0 (everyone connected to everyone)

**Star Graph:**
- Center: high degree, high betweenness, high closeness
- Periphery: low everything

**Path Graph:**
- Middle nodes: higher betweenness
- End nodes: lower closeness

---

## Files to Create
- `src/centrality.py`

## Files to Reference
- `src/networks.py` (for graph dictionary structure)
- `src/config.py` (for CITIES list)
