# Task 4: Weighting Schemes Module

## Objective
Build `src/weighting.py` that implements 5 different schemes for combining centrality scores into an overall city importance score.

## Context
We have 4 centrality metrics (D_i, C_i, B_i, E_i) for each city in each year-category. Now we need to:
1. Combine these 4 metrics into a single importance score
2. Implement 5 different weighting strategies
3. Aggregate scores across categories to get year-level city rankings
4. Enable comparison between different weighting approaches

## The 5 Weighting Schemes

### 1. Equal Weighting (Baseline)
**Philosophy:** Treat all metrics equally
**Weights:** w_D = w_C = w_B = w_E = 0.25
**Use Case:** Neutral baseline, no assumptions about importance

### 2. Correlation-Based Weighting
**Philosophy:** Reduce redundancy - down-weight correlated metrics
**Algorithm:** 
```
For each metric k:
  corr_sum = sum of |correlations| with other metrics
  w_k = 1 / (1 + corr_sum)
Normalize weights to sum to 1
```
**Use Case:** When metrics provide overlapping information

### 3. Category Importance Weighting
**Philosophy:** Weight by economic significance
**Algorithm:**
```
First: Equal combination within category (0.25 each)
Then: Weight by category importance (from config)
  - Food: 0.25
  - Utilities: 0.35  
  - Others: 0.10
```
**Use Case:** Policy-relevant rankings emphasizing essentials

### 4. Entropy-Based Weighting
**Philosophy:** Value metrics with high information content (variance)
**Algorithm:**
```
For each metric k:
  Normalize values to probability distribution
  Calculate Shannon entropy: H_k = -Σ p_i log(p_i)
  w_k = H_k / Σ H_j
```
**Use Case:** Highlight discriminating factors

### 5. Interactive Weighting
**Philosophy:** User-defined weights
**Algorithm:**
```
Accept custom weights as input
Validate: all weights >= 0 and sum to 1
Apply weights
```
**Use Case:** Stakeholder-specific priorities, sensitivity analysis

---

## Requirements

### Function 1: `apply_equal_weighting(df_centrality: pd.DataFrame) -> pd.DataFrame`

**Purpose:** Baseline approach - equal weights

**Algorithm:**
```
Score = 0.25 * D_i + 0.25 * C_i + 0.25 * B_i + 0.25 * E_i
```

**Input:**
- df_centrality: DataFrame with columns [Year, Category, City, D_i, C_i, B_i, E_i]

**Output:**
- Same DataFrame with added column: Score_Equal

---

### Function 2: `apply_correlation_weighting(df_centrality: pd.DataFrame) -> pd.DataFrame`

**Purpose:** Weight inversely to correlation

**Algorithm:**
```
1. Group by (Year, Category)
2. For each group:
   a. Extract metrics: [D_i, C_i, B_i, E_i]
   b. Calculate correlation matrix
   c. For each metric:
      - Sum absolute correlations with other metrics (exclude self)
      - w_k = 1 / (1 + corr_sum)
   d. Normalize weights to sum to 1
   e. Calculate weighted score
3. Combine all groups
```

**Input:**
- df_centrality: Centrality DataFrame

**Output:**
- DataFrame with added column: Score_Correlation

**Edge Cases:**
- Handle groups with constant values (zero correlation)
- If all correlations are 1, fall back to equal weights

---

### Function 3: `apply_category_weighting(df_centrality: pd.DataFrame, category_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame`

**Purpose:** Weight by category importance

**Algorithm:**
```
1. If category_weights not provided, use CATEGORY_WEIGHTS_IMPORTANCE from config
2. First stage: Equal combination within category
   Score_category = 0.25 * D_i + 0.25 * C_i + 0.25 * B_i + 0.25 * E_i
3. Second stage: Apply category weight
   Score_weighted = Score_category * category_weight[category]
4. Group by (Year, City) and sum across categories
```

**Input:**
- df_centrality: Centrality DataFrame
- category_weights: Optional dict mapping category names to weights

**Output:**
- DataFrame with added columns: 
  - Score_Category (within-category score)
  - Score_Category_Weighted (final weighted score)

---

### Function 4: `apply_entropy_weighting(df_centrality: pd.DataFrame) -> pd.DataFrame`

**Purpose:** Weight by information content

**Algorithm:**
```
1. Group by (Year, Category)
2. For each group:
   a. For each metric (D_i, C_i, B_i, E_i):
      - Normalize to probability distribution: p_i = value_i / sum(values)
      - Calculate entropy: H = -Σ p_i * log(p_i + epsilon)
   b. Calculate weights: w_k = H_k / Σ H_j
   c. If sum of entropies is zero, use equal weights
   d. Calculate weighted score
3. Combine results
```

**Input:**
- df_centrality: Centrality DataFrame

**Output:**
- DataFrame with added column: Score_Entropy

**Implementation Notes:**
- Add small epsilon (1e-10) to avoid log(0)
- Handle case where all values are identical (entropy = 0)

---

### Function 5: `apply_interactive_weighting(df_centrality: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame`

**Purpose:** User-defined custom weights

**Algorithm:**
```
1. Validate weights:
   - Check all keys present: ['D_i', 'C_i', 'B_i', 'E_i']
   - Check all values >= 0
   - Check sum = 1 (allow small tolerance)
2. Apply weights:
   Score = w_D * D_i + w_C * C_i + w_B * B_i + w_E * E_i
```

**Input:**
- df_centrality: Centrality DataFrame
- weights: Dict like {'D_i': 0.4, 'C_i': 0.3, 'B_i': 0.2, 'E_i': 0.1}

**Output:**
- DataFrame with added column: Score_Interactive

**Validation Errors:**
- Raise ValueError if weights don't sum to 1
- Raise ValueError if any weight is negative
- Raise KeyError if required metric missing

---

### Function 6: `apply_all_weighting_schemes(df_centrality: pd.DataFrame) -> pd.DataFrame`

**Purpose:** Apply all 5 schemes at once

**Algorithm:**
```
1. Apply equal weighting
2. Apply correlation weighting
3. Apply category weighting
4. Apply entropy weighting
5. Apply interactive with example weights (0.25 each)
6. Return DataFrame with all score columns
```

**Output:**
- DataFrame with columns:
  - Original: Year, Category, City, D_i, C_i, B_i, E_i
  - Added: Score_Equal, Score_Correlation, Score_Category_Weighted, Score_Entropy, Score_Interactive

---

### Function 7: `aggregate_scores_by_year(df_scores: pd.DataFrame) -> pd.DataFrame`

**Purpose:** Aggregate category-level scores to year-level

**Algorithm:**
```
1. Group by (Year, City)
2. Aggregate each score column:
   - Score_Equal: mean
   - Score_Correlation: mean
   - Score_Entropy: mean
   - Score_Category_Weighted: sum (already weighted)
   - Score_Interactive: mean
3. Sort by year and score (descending)
```

**Output:**
- Year-level DataFrame with columns:
  [Year, City, Score_Equal, Score_Correlation, Score_Entropy, Score_Category_Weighted, Score_Interactive]

---

### Function 8: `get_rankings(df_scores: pd.DataFrame, year: int, scheme: str) -> pd.DataFrame`

**Purpose:** Get city rankings for a specific year and scheme

**Algorithm:**
```
1. Filter to specified year
2. Sort by specified score column (descending)
3. Add rank column
4. Return sorted DataFrame
```

**Input:**
- df_scores: Aggregated scores DataFrame
- year: Year to analyze
- scheme: One of ['Score_Equal', 'Score_Correlation', 'Score_Entropy', 'Score_Category_Weighted', 'Score_Interactive']

**Output:**
- DataFrame with columns: [Rank, City, Score]

---

### Function 9: `compare_schemes(df_scores: pd.DataFrame, year: int, top_n: int = 10) -> pd.DataFrame`

**Purpose:** Show how different schemes rank the same cities

**Algorithm:**
```
1. Get top N cities under each scheme
2. Create comparison table showing:
   - City name
   - Rank under each scheme
   - Score under each scheme
3. Highlight cities that change rank significantly
```

**Output:**
- Comparison DataFrame with multi-index columns

---

## Implementation Details

### Imports
```python
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from src.config import CATEGORY_WEIGHTS_IMPORTANCE, DEFAULT_WEIGHTS
```

### Validation
- Check input DataFrame has required columns
- Validate weight dictionaries
- Handle edge cases (empty DataFrames, missing categories)

### Logging
```python
logger.info("Applying equal weighting...")
logger.info("Applying correlation-based weighting...")
logger.warning("High correlation detected between metrics")
logger.info("Aggregating scores across categories...")
```

---

## Testing Criteria

1. **Weight Sum Test**: All schemes produce weights summing to 1
2. **Range Test**: All scores should be between 0 and 1
3. **Consistency Test**: Same centralities → same score
4. **Sensitivity Test**: Changing weights → different rankings
5. **Edge Case Test**: Zero centralities → zero score

---

## Example Usage

```python
from src.weighting import apply_all_weighting_schemes, aggregate_scores_by_year, compare_schemes
from src.centrality import calculate_all_centralities
from src.networks import build_all_networks
from src.preprocessing import run_preprocessing_pipeline

# Full pipeline
df_data = run_preprocessing_pipeline()
graphs = build_all_networks(df_data, [2023, 2024], threshold=0.75)
df_centrality = calculate_all_centralities(graphs)

# Apply all weighting schemes
df_scores = apply_all_weighting_schemes(df_centrality)

# Aggregate to year level
df_final = aggregate_scores_by_year(df_scores)

# Compare schemes
comparison = compare_schemes(df_final, year=2024, top_n=10)
print(comparison)

# Save
df_final.to_csv('data/results/final_city_rankings.csv', index=False)
```

---

## Success Criteria

✅ All 5 schemes implemented correctly
✅ Weights always sum to 1
✅ Scores in valid range [0, 1]
✅ Handles edge cases gracefully
✅ Results are interpretable
✅ Enables meaningful comparisons
✅ Comprehensive documentation

---

## Files to Create
- `src/weighting.py`

## Files to Reference
- `src/centrality.py` (for DataFrame structure)
- `src/config.py` (for CATEGORY_WEIGHTS_IMPORTANCE)
