"""
Configuration file for Pakistan CPI Network Analysis.

This module contains all constants and configuration parameters used throughout the project.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CITIES
# ============================================================================
# List of major Pakistani cities included in CPI analysis
CITIES = [
    'Islamabad',
    'Karachi',
    'Lahore',
    'Quetta',
    'Peshawar',
    'Faisalabad',
    'Multan',
    'Rawalpindi',
    'Gujranwala',
    'Hyderabad',
    'Sialkot',
    'Bahawalpur',
    'Sukkur',
    'Sargodha',
    'Larkana',
    'Sheikhupura',
    'Rahim Yar Khan'
]

# ============================================================================
# CATEGORIES
# ============================================================================
# CPI categories analyzed in the project
CATEGORIES = [
    '1. Food Staples & Grains',
    '2. Meat, Poultry & Dairy',
    '3. Fruits & Vegetables',
    '4. Utilities & Energy',
    '5. Housing & Transport',
    '6. Clothing & Footwear',
    '7. Miscellaneous'
]

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
# Years to analyze
ANALYSIS_YEARS = [2023, 2024]

# Similarity threshold (Ä) for creating network edges
TAU = float(os.getenv('SIMILARITY_THRESHOLD', '0.75'))

# ============================================================================
# CATEGORY IMPORTANCE WEIGHTS
# ============================================================================
# Weights for category importance weighting scheme
# These reflect economic significance of each category
CATEGORY_WEIGHTS_IMPORTANCE = {
    '1. Food Staples & Grains': 0.25,
    '2. Meat, Poultry & Dairy': 0.20,
    '3. Fruits & Vegetables': 0.10,
    '4. Utilities & Energy': 0.35,
    '5. Housing & Transport': 0.05,
    '6. Clothing & Footwear': 0.03,
    '7. Miscellaneous': 0.02
}

# ============================================================================
# METRIC WEIGHTS
# ============================================================================
# Default equal weights for centrality metrics
DEFAULT_WEIGHTS = {
    'D_i': 0.25,  # Degree centrality
    'C_i': 0.25,  # Closeness centrality
    'B_i': 0.25,  # Betweenness centrality
    'E_i': 0.25   # Eigenvector centrality
}

# ============================================================================
# PATHS
# ============================================================================
BASE_PATH = os.getenv('BASE_PATH', os.getcwd())
PDF_DIR = os.path.join(BASE_PATH, os.getenv('PDF_DIR', 'data/raw'))
PROCESSED_DIR = os.path.join(BASE_PATH, os.getenv('PROCESSED_DIR', 'data/processed'))
RESULTS_DIR = os.path.join(BASE_PATH, os.getenv('RESULTS_DIR', 'data/results'))

# ============================================================================
# WEB SCRAPING CONFIGURATION
# ============================================================================
PBS_CPI_BASE_URL = "https://www.pbs.gov.pk/publications/consumer-price-index-cpi"
DOWNLOAD_CHECK_INTERVAL = 3600  # seconds

# ============================================================================
# DATE RANGE
# ============================================================================
START_DATE = os.getenv('START_DATE', '2023-03-01')
END_DATE = os.getenv('END_DATE', '2025-05-01')

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
