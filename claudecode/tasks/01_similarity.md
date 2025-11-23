# Task 1: Similarity Calculation Module

## Objective
Build `src/similarity.py` that calculates cosine similarity between city price vectors.

## Context
We have preprocessed CPI data with normalized prices (Z-scores). Now we need to:
1. Construct price vectors for each city (features = normalized prices across items in a month)
2. Calculate pairwise cosine similarity between all city pairs
3. Return similarity matrices that can be used to build networks

## Requirements

### Function 1: `compute_cosine_similarity(M_numpy: np.ndarray) -> np.ndarray`
**Purpose:** Calculate pairwise cosine similarity for all rows in a matrix

**Algorithm:**
```
1. Normalize each row by its L2 norm (||row||)
2. Handle zero-norm rows (set to tiny value to avoid division by zero)
3. Compute similarity matrix: M_normalized @ M_normalized.T
4. Return symmetric matrix where entry (i,j) = cosine similarity between row i and row j
```

**Input:** 
- M_numpy: NumPy array of shape (n_cities, n_features)

**Output:**
- Similarity matrix of shape (n_cities, n_cities)
- Values range from -1 to 1 (1 = identical, 0 = orthogonal, -1 = opposite)

**Edge Cases:**
- Handle rows with all zeros
- Ensure diagonal is all 1.0 (city similar to itself)
- Result must be symmetric

---

### Function 2: `create_price_vector_matrix(df: pd.DataFrame, year: int, category: str, cities: List[str]) -> Tuple[pd.DataFrame, np.ndarray]`

**Purpose:** Build the city-feature matrix M_{y,c} for a specific year and category

**Algorithm:**
```
1. Filter data: df[(df['Year'] == year) & (df['Category'] == category)]
2. Create pivot table:
   - Rows = Cities
   - Columns = (Month_Year, Item_Description) combinations
   - Values = Normalized_Price
   - Fill missing with 0
3. Reindex to ensure all cities from CITIES list are present
4. Return both pandas DataFrame and NumPy array
```

**Input:**
- df: Preprocessed DataFrame with columns [City, Year, Category, Month_Year, Item_Description, Normalized_Price]
- year: Year to filter (e.g., 2024)
- category: Category to filter (e.g., "1. Food Staples & Grains")
- cities: List of city names from config

**Output:**
- Tuple of (DataFrame, numpy_array)
- DataFrame has cities as index, item-month combos as columns
- If no data found, return (None, None)

**Validation:**
- Check that output has correct number of cities
- Warn if too many missing values
- Log the shape of the resulting matrix

---

### Function 3: `get_similarity_matrix(df: pd.DataFrame, year: int, category: str) -> Tuple[np.ndarray, List[str]]`

**Purpose:** Convenience function combining vector creation and similarity calculation

**Algorithm:**
```
1. Call create_price_vector_matrix()
2. If data exists, call compute_cosine_similarity()
3. Return similarity matrix and list of cities
```

**Input:**
- Same as create_price_vector_matrix

**Output:**
- Tuple of (similarity_matrix, city_names)
- Returns (None, None) if no data available

---

## Implementation Details

### Imports Needed
```python
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import logging
from src.config import CITIES
```

### Logging
- Log shape of matrices created
- Log min/max/mean similarity values
- Warn if many zero similarities

### Error Handling
- Handle empty DataFrames gracefully
- Catch division by zero in normalization
- Validate input dimensions

---

## Testing Criteria

The module should pass these tests:

1. **Identity Test**: Cosine similarity of vector with itself = 1.0
2. **Symmetry Test**: sim(A,B) = sim(B,A)
3. **Range Test**: All similarities between -1 and 1
4. **Known Case**: Two identical vectors should have similarity 1.0
5. **Orthogonal Case**: Orthogonal vectors should have similarity ≈ 0

---

## Example Usage

```python
from src.similarity import get_similarity_matrix
from src.preprocessing import run_preprocessing_pipeline

# Load data
df = run_preprocessing_pipeline()

# Get similarity matrix for Food category in 2024
sim_matrix, cities = get_similarity_matrix(
    df, 
    year=2024, 
    category='1. Food Staples & Grains'
)

print(f"Similarity matrix shape: {sim_matrix.shape}")
print(f"Mean similarity: {np.mean(sim_matrix):.3f}")
print(f"Top similar pair: {np.max(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]):.3f}")
```

---

## Success Criteria

✅ Module imports without errors
✅ All functions have comprehensive docstrings
✅ Handles edge cases gracefully
✅ Returns correct shapes
✅ Similarity values are valid (in range [-1, 1])
✅ Results are reproducible

---

## Files to Create
- `src/similarity.py`

## Files to Reference
- `src/config.py` (for CITIES list)
- `src/preprocessing.py` (for understanding DataFrame structure)
