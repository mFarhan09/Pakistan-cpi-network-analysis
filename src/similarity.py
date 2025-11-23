"""
Similarity Calculation Module.

This module calculates cosine similarity between city price vectors to identify
cities with similar price patterns across different categories and time periods.

Functions:
    - compute_cosine_similarity: Calculate pairwise cosine similarity matrix
    - create_price_vector_matrix: Build city-feature matrix for year/category
    - get_similarity_matrix: Convenience function combining above two
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import logging

from src.config import CITIES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_cosine_similarity(M_numpy: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise cosine similarity for all rows in a matrix.

    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical). This metric is useful for
    comparing price patterns regardless of absolute magnitude.

    Algorithm:
        1. Normalize each row by its L2 norm (||row||)
        2. Handle zero-norm rows (set to tiny value to avoid division by zero)
        3. Compute similarity matrix: M_normalized @ M_normalized.T
        4. Return symmetric matrix where entry (i,j) = cosine similarity

    Args:
        M_numpy: NumPy array of shape (n_cities, n_features)
                 Each row represents a city's price vector

    Returns:
        Similarity matrix of shape (n_cities, n_cities)
        - Values range from -1 to 1
        - Diagonal elements are all 1.0 (city similar to itself)
        - Matrix is symmetric: sim(i,j) = sim(j,i)

    Edge Cases:
        - Handles rows with all zeros (returns 0 similarity with others)
        - Ensures diagonal is all 1.0
        - Result is symmetric

    Example:
        >>> M = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        >>> sim = compute_cosine_similarity(M)
        >>> sim[0, 1]  # First two rows are identical
        1.0
    """
    # Input validation
    if M_numpy.size == 0:
        logger.warning("Empty matrix provided to compute_cosine_similarity")
        return np.array([])

    if len(M_numpy.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {M_numpy.shape}")

    # Calculate L2 norm for each row
    norms = np.linalg.norm(M_numpy, axis=1, keepdims=True)

    # Handle zero-norm rows to avoid division by zero
    # Replace zero norms with tiny value (effectively makes similarity 0)
    epsilon = 1e-10
    norms = np.where(norms == 0, epsilon, norms)

    # Normalize each row by its L2 norm
    M_normalized = M_numpy / norms

    # Compute similarity matrix: M_normalized @ M_normalized.T
    similarity_matrix = M_normalized @ M_normalized.T

    # Ensure diagonal is exactly 1.0 (may have floating point errors)
    np.fill_diagonal(similarity_matrix, 1.0)

    # Clip values to [-1, 1] range (handle numerical precision issues)
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

    # Log statistics
    n_cities = M_numpy.shape[0]
    # Get upper triangle (excluding diagonal) for statistics
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

    if len(upper_triangle) > 0:
        logger.info(f"Similarity matrix computed: {n_cities}x{n_cities}")
        logger.info(f"  Min similarity: {np.min(upper_triangle):.3f}")
        logger.info(f"  Max similarity: {np.max(upper_triangle):.3f}")
        logger.info(f"  Mean similarity: {np.mean(upper_triangle):.3f}")

        # Warn if many zero similarities
        zero_count = np.sum(upper_triangle == 0)
        if zero_count > len(upper_triangle) * 0.5:
            logger.warning(f"High number of zero similarities: {zero_count}/{len(upper_triangle)}")

    return similarity_matrix


def create_price_vector_matrix(
    df: pd.DataFrame,
    year: int,
    category: str,
    cities: List[str] = CITIES
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """
    Build the city-feature matrix M_{y,c} for a specific year and category.

    Creates a matrix where:
    - Rows = Cities
    - Columns = (Month_Year, Item_Description) combinations
    - Values = Normalized_Price (Z-scores)
    - Missing values filled with 0

    Algorithm:
        1. Filter data: df[(df['Year'] == year) & (df['Category'] == category)]
        2. Create pivot table with cities as rows, item-month combos as columns
        3. Fill missing values with 0
        4. Reindex to ensure all cities from CITIES list are present
        5. Return both pandas DataFrame and NumPy array

    Args:
        df: Preprocessed DataFrame with columns:
            [City, Year, Category, Month_Year, Item_Description, Normalized_Price]
        year: Year to filter (e.g., 2024)
        category: Category to filter (e.g., "1. Food Staples & Grains")
        cities: List of city names from config (default: CITIES)

    Returns:
        Tuple of (DataFrame, numpy_array):
        - DataFrame has cities as index, item-month combos as columns
        - numpy_array is the same data in array format
        - Returns (None, None) if no data found for the year/category

    Validation:
        - Checks that output has correct number of cities
        - Warns if too many missing values
        - Logs the shape of the resulting matrix

    Example:
        >>> df_matrix, np_matrix = create_price_vector_matrix(df, 2024, "1. Food Staples & Grains")
        >>> print(df_matrix.shape)  # (17, 240) - 17 cities, 240 item-month features
    """
    # Input validation
    required_columns = ['City', 'Year', 'Category', 'Month_Year', 'Item_Description', 'Normalized_Price']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    # Filter data for specific year and category
    filtered = df[(df['Year'] == year) & (df['Category'] == category)].copy()

    # Check if data exists
    if filtered.empty:
        logger.warning(f"No data found for Year={year}, Category='{category}'")
        return None, None

    logger.info(f"Processing Year={year}, Category='{category}'")
    logger.info(f"  Filtered data: {len(filtered)} rows")

    # Create feature name by combining Month_Year and Item_Description
    filtered['Feature'] = filtered['Month_Year'].astype(str) + '_' + filtered['Item_Description'].astype(str)

    # Create pivot table
    # Rows: Cities, Columns: Features (Month_Year + Item_Description), Values: Normalized_Price
    pivot = filtered.pivot_table(
        index='City',
        columns='Feature',
        values='Normalized_Price',
        aggfunc='first'  # Use first if duplicates exist
    )

    # Fill missing values with 0
    pivot = pivot.fillna(0)

    # Reindex to ensure all cities are present (adds missing cities with all zeros)
    pivot = pivot.reindex(cities, fill_value=0)

    # Convert to numpy array
    np_matrix = pivot.values

    # Validation
    n_cities_expected = len(cities)
    n_cities_actual = pivot.shape[0]
    n_features = pivot.shape[1]

    if n_cities_actual != n_cities_expected:
        logger.warning(f"Expected {n_cities_expected} cities, got {n_cities_actual}")

    # Check for too many missing values (zeros)
    total_values = np_matrix.size
    zero_count = np.sum(np_matrix == 0)
    zero_percentage = (zero_count / total_values) * 100 if total_values > 0 else 0

    logger.info(f"  Matrix shape: {pivot.shape} ({n_cities_actual} cities, {n_features} features)")
    logger.info(f"  Zero values: {zero_count}/{total_values} ({zero_percentage:.1f}%)")

    if zero_percentage > 70:
        logger.warning(f"High percentage of missing values: {zero_percentage:.1f}%")

    return pivot, np_matrix


def get_similarity_matrix(
    df: pd.DataFrame,
    year: int,
    category: str,
    cities: List[str] = CITIES
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Convenience function combining vector creation and similarity calculation.

    This function wraps create_price_vector_matrix and compute_cosine_similarity
    to provide a simple interface for getting similarity matrices.

    Algorithm:
        1. Call create_price_vector_matrix() to build feature matrix
        2. If data exists, call compute_cosine_similarity()
        3. Return similarity matrix and list of cities

    Args:
        df: Preprocessed DataFrame with columns:
            [City, Year, Category, Month_Year, Item_Description, Normalized_Price]
        year: Year to filter (e.g., 2024)
        category: Category to filter (e.g., "1. Food Staples & Grains")
        cities: List of city names from config (default: CITIES)

    Returns:
        Tuple of (similarity_matrix, city_names):
        - similarity_matrix: NumPy array of shape (n_cities, n_cities)
        - city_names: List of city names corresponding to matrix rows/columns
        - Returns (None, None) if no data available

    Example:
        >>> sim_matrix, city_list = get_similarity_matrix(df, 2024, "1. Food Staples & Grains")
        >>> if sim_matrix is not None:
        ...     print(f"Similarity between {city_list[0]} and {city_list[1]}: {sim_matrix[0, 1]:.3f}")
    """
    # Create price vector matrix
    df_matrix, np_matrix = create_price_vector_matrix(df, year, category, cities)

    # If no data, return None
    if df_matrix is None or np_matrix is None:
        logger.warning(f"Cannot compute similarity: no data for Year={year}, Category='{category}'")
        return None, None

    # Compute similarity matrix
    similarity_matrix = compute_cosine_similarity(np_matrix)

    # Return similarity matrix and city names
    city_names = df_matrix.index.tolist()

    logger.info(f"Similarity matrix ready for {len(city_names)} cities")

    return similarity_matrix, city_names


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    """
    Basic testing of similarity module functions.
    """
    print("="*80)
    print("SIMILARITY MODULE TESTS")
    print("="*80)

    # Test 1: Identity Test
    print("\n[Test 1] Identity Test - Vector with itself should have similarity 1.0")
    vec1 = np.array([[1, 2, 3]])
    sim = compute_cosine_similarity(vec1)
    assert abs(sim[0, 0] - 1.0) < 1e-6, "Failed: diagonal should be 1.0"
    print(f" Passed: sim[0,0] = {sim[0, 0]:.6f}")

    # Test 2: Symmetry Test
    print("\n[Test 2] Symmetry Test - sim(A,B) should equal sim(B,A)")
    vec2 = np.array([[1, 2, 3], [4, 5, 6]])
    sim2 = compute_cosine_similarity(vec2)
    assert abs(sim2[0, 1] - sim2[1, 0]) < 1e-6, "Failed: matrix should be symmetric"
    print(f" Passed: sim[0,1] = sim[1,0] = {sim2[0, 1]:.6f}")

    # Test 3: Range Test
    print("\n[Test 3] Range Test - All similarities should be in [-1, 1]")
    vec3 = np.random.randn(10, 20)
    sim3 = compute_cosine_similarity(vec3)
    assert np.all(sim3 >= -1.0) and np.all(sim3 <= 1.0), "Failed: values outside [-1, 1]"
    print(f" Passed: min={np.min(sim3):.3f}, max={np.max(sim3):.3f}")

    # Test 4: Known Case - Identical vectors
    print("\n[Test 4] Known Case - Identical vectors should have similarity 1.0")
    vec4 = np.array([[1, 2, 3], [1, 2, 3]])
    sim4 = compute_cosine_similarity(vec4)
    assert abs(sim4[0, 1] - 1.0) < 1e-6, "Failed: identical vectors should have sim=1.0"
    print(f" Passed: identical vectors have sim = {sim4[0, 1]:.6f}")

    # Test 5: Orthogonal Case
    print("\n[Test 5] Orthogonal Case - Orthogonal vectors should have similarity H 0")
    vec5 = np.array([[1, 0, 0], [0, 1, 0]])
    sim5 = compute_cosine_similarity(vec5)
    assert abs(sim5[0, 1]) < 1e-6, "Failed: orthogonal vectors should have simH0"
    print(f" Passed: orthogonal vectors have sim = {sim5[0, 1]:.6f}")

    # Test 6: Zero vector edge case
    print("\n[Test 6] Edge Case - Zero vector handling")
    vec6 = np.array([[1, 2, 3], [0, 0, 0], [4, 5, 6]])
    sim6 = compute_cosine_similarity(vec6)
    print(f" Passed: handled zero vector without error")
    print(f"  sim(zero, vec1) = {sim6[1, 0]:.6f}")
    print(f"  sim(zero, zero) = {sim6[1, 1]:.6f}")

    print("\n" + "="*80)
    print("ALL TESTS PASSED ")
    print("="*80)
