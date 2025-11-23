#!/usr/bin/env python3
"""
Main Pipeline for Pakistan CPI Network Analysis.

Orchestrates the complete analysis workflow from data loading to visualization.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.networks import build_all_networks, print_network_summary
from src.centrality import calculate_all_centralities, print_centrality_summary
from src.weighting import apply_all_weighting_schemes, aggregate_scores_by_year
from src.temporal import analyze_network_evolution
from src.config import ANALYSIS_YEARS, CATEGORIES, TAU, RESULTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run complete analysis pipeline.

    Steps:
        1. Load and preprocess data
        2. Build networks for all year-category combinations
        3. Calculate centralities
        4. Apply weighting schemes
        5. Aggregate results
        6. Analyze temporal evolution
        7. Generate visualizations
        8. Return results dictionary

    Args:
        config: Configuration dictionary with keys:
            - years: List of years to analyze
            - categories: List of categories
            - threshold: Similarity threshold
            - output_dir: Output directory path
            - save_plots: Whether to save plots
            - quick: Quick mode (fewer combinations)

    Returns:
        Dictionary containing all analysis results:
        - graphs_dict: Network graphs
        - df_centrality: Centrality scores
        - df_scores: Weighted scores
        - df_year: Year-level aggregated scores
        - df_evolution: Temporal evolution data

    Example:
        >>> config = {'years': [2023, 2024], 'threshold': 0.75, 'output_dir': 'results/'}
        >>> results = run_full_analysis(config)
    """
    logger.info("="*80)
    logger.info("PAKISTAN CPI NETWORK ANALYSIS PIPELINE")
    logger.info("="*80)
    logger.info(f"Analysis Years: {config['years']}")
    logger.info(f"Threshold: {config['threshold']}")
    logger.info(f"Output Directory: {config['output_dir']}")
    logger.info("="*80)

    results = {}

    # STEP 1: Load and preprocess data
    logger.info("\n[STEP 1/7] Loading and preprocessing data...")
    try:
        # For now, we'll skip this step as preprocessing module needs data
        # In production, this would call:
        # from src.preprocessing import run_preprocessing_pipeline
        # df_data = run_preprocessing_pipeline()
        logger.warning("Skipping data loading (no preprocessing module implemented yet)")
        df_data = None
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return results

    # STEP 2: Build networks
    logger.info("\n[STEP 2/7] Building networks...")
    try:
        if df_data is not None:
            graphs_dict = build_all_networks(
                df_data,
                years=config['years'],
                categories=config.get('categories'),
                threshold=config['threshold']
            )
            results['graphs_dict'] = graphs_dict
            print_network_summary(graphs_dict)
        else:
            logger.warning("Skipping network building (no data available)")
            results['graphs_dict'] = {}
    except Exception as e:
        logger.error(f"Network building failed: {e}")
        results['graphs_dict'] = {}

    # STEP 3: Calculate centralities
    logger.info("\n[STEP 3/7] Calculating centralities...")
    try:
        if results['graphs_dict']:
            df_centrality = calculate_all_centralities(results['graphs_dict'])
            results['df_centrality'] = df_centrality
            print_centrality_summary(df_centrality)
        else:
            logger.warning("Skipping centrality calculation (no networks available)")
            results['df_centrality'] = None
    except Exception as e:
        logger.error(f"Centrality calculation failed: {e}")
        results['df_centrality'] = None

    # STEP 4: Apply weighting schemes
    logger.info("\n[STEP 4/7] Applying weighting schemes...")
    try:
        if results['df_centrality'] is not None:
            df_scores = apply_all_weighting_schemes(results['df_centrality'])
            results['df_scores'] = df_scores
        else:
            logger.warning("Skipping weighting (no centrality scores available)")
            results['df_scores'] = None
    except Exception as e:
        logger.error(f"Weighting failed: {e}")
        results['df_scores'] = None

    # STEP 5: Aggregate by year
    logger.info("\n[STEP 5/7] Aggregating scores by year...")
    try:
        if results['df_scores'] is not None:
            df_year = aggregate_scores_by_year(results['df_scores'])
            results['df_year'] = df_year
        else:
            logger.warning("Skipping aggregation (no scores available)")
            results['df_year'] = None
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        results['df_year'] = None

    # STEP 6: Analyze temporal evolution
    logger.info("\n[STEP 6/7] Analyzing temporal evolution...")
    try:
        if results['graphs_dict']:
            df_evolution = analyze_network_evolution(results['graphs_dict'])
            results['df_evolution'] = df_evolution
        else:
            logger.warning("Skipping temporal analysis (no networks available)")
            results['df_evolution'] = None
    except Exception as e:
        logger.error(f"Temporal analysis failed: {e}")
        results['df_evolution'] = None

    # STEP 7: Save results
    logger.info("\n[STEP 7/7] Saving results...")
    try:
        save_results(results, config['output_dir'])
    except Exception as e:
        logger.error(f"Saving results failed: {e}")

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)

    return results


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save all results to files.

    Args:
        results: Dictionary from run_full_analysis
        output_dir: Output directory path

    Saves:
        - DataFrames to CSV files
        - Summary report as markdown
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to {output_path}")

    # Save centrality scores
    if results.get('df_centrality') is not None:
        csv_path = output_path / 'centrality_scores.csv'
        results['df_centrality'].to_csv(csv_path, index=False)
        logger.info(f"  Saved: {csv_path}")

    # Save weighted scores
    if results.get('df_scores') is not None:
        csv_path = output_path / 'weighted_scores.csv'
        results['df_scores'].to_csv(csv_path, index=False)
        logger.info(f"  Saved: {csv_path}")

    # Save year-level rankings
    if results.get('df_year') is not None:
        csv_path = output_path / 'final_rankings.csv'
        results['df_year'].to_csv(csv_path, index=False)
        logger.info(f"  Saved: {csv_path}")

    # Save temporal evolution
    if results.get('df_evolution') is not None:
        csv_path = output_path / 'network_evolution.csv'
        results['df_evolution'].to_csv(csv_path, index=False)
        logger.info(f"  Saved: {csv_path}")

    # Generate and save summary report
    report = generate_summary_report(results)
    report_path = output_path / 'analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"  Saved: {report_path}")


def generate_summary_report(results: Dict[str, Any]) -> str:
    """
    Generate markdown summary report.

    Args:
        results: Dictionary from run_full_analysis

    Returns:
        Markdown-formatted report string
    """
    report = []
    report.append("# Pakistan CPI Network Analysis Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("="*80 + "\n\n")

    # Summary statistics
    if results.get('graphs_dict'):
        report.append("## Network Summary\n")
        report.append(f"- Total Networks: {len(results['graphs_dict'])}\n")

    if results.get('df_centrality') is not None:
        report.append("\n## Centrality Analysis\n")
        report.append(f"- Total City-Network Combinations: {len(results['df_centrality'])}\n")
        report.append(f"- Unique Cities: {results['df_centrality']['City'].nunique()}\n")

    if results.get('df_year') is not None:
        report.append("\n## Final Rankings\n")
        for year in sorted(results['df_year']['Year'].unique()):
            report.append(f"\n### Year {year}\n")
            df_year = results['df_year'][results['df_year']['Year'] == year]
            top_5 = df_year.nlargest(5, 'Score_Equal')
            report.append("\nTop 5 Cities (Equal Weighting):\n")
            for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                report.append(f"{idx}. {row['City']}: {row['Score_Equal']:.3f}\n")

    if results.get('df_evolution') is not None:
        report.append("\n## Temporal Evolution\n")
        report.append(f"- Average Density: {results['df_evolution']['Density'].mean():.3f}\n")
        report.append(f"- Average Clustering: {results['df_evolution']['Clustering'].mean():.3f}\n")

    report.append("\n" + "="*80 + "\n")
    report.append("End of Report\n")

    return ''.join(report)


def main():
    """
    Main entry point with CLI interface.
    """
    parser = argparse.ArgumentParser(
        description='Pakistan CPI Network Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        default=ANALYSIS_YEARS,
        help='Years to analyze (default: from config)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=TAU,
        help=f'Similarity threshold (default: {TAU})'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=RESULTS_DIR,
        help='Output directory (default: data/results/)'
    )

    parser.add_argument(
        '--categories',
        nargs='+',
        default=None,
        help='Specific categories to analyze (default: all)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: analyze only first year'
    )

    parser.add_argument(
        '--help-full',
        action='store_true',
        help='Show full help and exit'
    )

    args = parser.parse_args()

    if args.help_full:
        print(__doc__)
        print("\nUsage Examples:")
        print("  python pipeline.py --years 2023 2024 --threshold 0.75")
        print("  python pipeline.py --quick  # Test with one year")
        print("  python pipeline.py --output custom_results/")
        return

    # Quick mode: only first year
    if args.quick:
        args.years = [args.years[0]] if args.years else [ANALYSIS_YEARS[0]]
        logger.info("Quick mode: analyzing only first year")

    # Build config
    config = {
        'years': args.years,
        'categories': args.categories if args.categories else CATEGORIES,
        'threshold': args.threshold,
        'output_dir': args.output,
        'save_plots': True
    }

    # Run pipeline
    try:
        results = run_full_analysis(config)

        if results:
            logger.info("\nAnalysis completed successfully!")
            logger.info(f"Results saved to: {args.output}")
        else:
            logger.warning("\nAnalysis completed with warnings (check logs)")

    except KeyboardInterrupt:
        logger.warning("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nAnalysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
