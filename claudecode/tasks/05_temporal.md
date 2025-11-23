# Task 5: Temporal Analysis Module

## Objective
Build `src/temporal.py` for analyzing temporal evolution of networks and creating Hasse diagrams.

## Core Functions

### 1. `check_subset_relationship(G1: nx.Graph, G2: nx.Graph) -> bool`
- Check if edges of G1 ⊆ edges of G2
- Return True if all edges in G1 exist in G2

### 2. `verify_partial_order(graphs: List[nx.Graph]) -> Dict[str, bool]`
- Verify reflexive, antisymmetric, transitive properties
- Return dictionary with verification results

### 3. `build_temporal_order_matrix(graphs_dict: Dict, category: str) -> Tuple[List[int], np.ndarray]`
- Build adjacency matrix for Hasse diagram
- Matrix[i][j] = 1 if G_i ⊆ G_j
- Remove transitive edges for clean Hasse diagram

### 4. `create_hasse_diagram(years: List[int], order_matrix: np.ndarray, category: str) -> nx.DiGraph`
- Create directed graph showing temporal ordering
- Nodes = year labels
- Edges = direct subset relationships (no transitive)

### 5. `analyze_network_evolution(graphs_dict: Dict) -> pd.DataFrame`
- Track metrics over time: nodes, edges, density, clustering
- Return DataFrame with temporal trends

## Success Criteria
✅ Correctly identifies subset relationships
✅ Verifies partial order properties
✅ Generates clean Hasse diagrams
✅ Provides temporal evolution statistics

## Files to Create
- `src/temporal.py`
