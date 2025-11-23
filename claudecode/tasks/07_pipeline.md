# Task 7: Main Pipeline Script

## Objective
Build `pipeline.py` - the main orchestration script that runs the complete analysis.

## Structure

```python
#!/usr/bin/env python3
"""
Main Pipeline for Pakistan CPI Network Analysis
Orchestrates the complete analysis workflow
"""

import argparse
from datetime import datetime
from src.preprocessing import run_preprocessing_pipeline
from src.networks import build_all_networks
from src.centrality import calculate_all_centralities
from src.weighting import apply_all_weighting_schemes, aggregate_scores_by_year
from src.temporal import analyze_network_evolution
from src.visualization import *
```

## Core Functions

### 1. `run_full_analysis(config: Dict) -> Dict`
Main function that:
1. Loads and preprocesses data
2. Builds networks for all year-category combinations
3. Calculates centralities
4. Applies weighting schemes
5. Aggregates results
6. Generates visualizations
7. Saves outputs
8. Returns results dictionary

### 2. `save_results(results: Dict, output_dir: str)`
- Save DataFrames to CSV
- Save graphs to pickle/GraphML
- Save visualizations as PNG
- Generate summary report

### 3. `generate_summary_report(results: Dict) -> str`
- Create text report with key findings
- Top cities by different metrics
- Network statistics
- Temporal trends
- Save as markdown

### 4. CLI Interface using argparse
```bash
python pipeline.py --years 2023 2024 --threshold 0.75 --output results/
python pipeline.py --force-reload  # Skip cache
python pipeline.py --quick  # Just 1 year for testing
```

## Output Structure
```
data/results/
├── preprocessed_data.csv
├── networks/
│   ├── network_2023_food.graphml
│   └── ...
├── centrality_scores.csv
├── final_rankings.csv
├── visualizations/
│   ├── network_2024_food.png
│   ├── heatmap_2024_utilities.png
│   └── rankings_comparison.png
└── analysis_report.md
```

## Progress Tracking
- Print clear progress messages
- Show percentage complete
- Estimate time remaining
- Handle interruptions gracefully

## Success Criteria
✅ Runs complete analysis end-to-end
✅ Saves all outputs properly
✅ Generates comprehensive report
✅ Handles errors gracefully
✅ Supports command-line arguments
✅ Clear progress tracking

## Files to Create
- `pipeline.py`
