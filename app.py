#!/usr/bin/env python3
"""
Streamlit Dashboard for Pakistan CPI Network Analysis.

Interactive web application for exploring network analysis results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ANALYSIS_YEARS, CATEGORIES, TAU, CITIES
from src.networks import build_network, get_network_statistics
from src.centrality import calculate_single_network_centrality
from src.weighting import apply_interactive_weighting
from src.visualization import (
    plot_network, plot_similarity_heatmap, plot_centrality_comparison,
    plot_rankings_comparison
)

# Page configuration
st.set_page_config(
    page_title="Pakistan CPI Network Analysis",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def setup_sidebar():
    """
    Configure sidebar controls.

    Returns:
        Dictionary with user-selected parameters
    """
    st.sidebar.header("🎛️ Analysis Parameters")

    # Year selection
    selected_years = st.sidebar.multiselect(
        "Select Years",
        options=ANALYSIS_YEARS,
        default=ANALYSIS_YEARS[:1] if ANALYSIS_YEARS else [],
        help="Choose one or more years to analyze"
    )

    # Category selection
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=CATEGORIES,
        default=CATEGORIES[:2] if len(CATEGORIES) >= 2 else CATEGORIES,
        help="Choose product categories"
    )

    # Threshold slider
    threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=TAU,
        step=0.05,
        help="Minimum similarity for edge creation"
    )

    # Weighting scheme
    weighting_scheme = st.sidebar.radio(
        "Weighting Scheme",
        options=["Equal", "Correlation", "Category", "Entropy", "Custom"],
        index=0,
        help="Method for combining centrality metrics"
    )

    st.sidebar.markdown("---")

    return {
        'years': selected_years,
        'categories': selected_categories,
        'threshold': threshold,
        'weighting_scheme': weighting_scheme
    }


def show_overview():
    """Display overview page with key statistics."""
    st.markdown('<div class="main-header">📊 Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Cities",
            value=len(CITIES),
            help="Number of cities analyzed"
        )

    with col2:
        st.metric(
            label="Categories",
            value=len(CATEGORIES),
            help="Number of product categories"
        )

    with col3:
        st.metric(
            label="Analysis Years",
            value=len(ANALYSIS_YEARS),
            help="Number of years in dataset"
        )

    with col4:
        st.metric(
            label="Threshold",
            value=f"{TAU:.2f}",
            help="Default similarity threshold"
        )

    st.markdown("---")

    # Project description
    st.subheader("About This Analysis")
    st.write("""
    This dashboard provides an interactive exploration of Pakistan's Consumer Price Index (CPI) data
    through network analysis. The analysis reveals how price patterns across different cities and
    product categories are interconnected.

    **Key Features:**
    - **Network Analysis**: Visualize how cities are connected based on price similarity
    - **Centrality Metrics**: Identify the most influential cities in price networks
    - **Temporal Evolution**: Track how networks change over time
    - **Multiple Weighting Schemes**: Compare different approaches to city ranking

    **Navigation:**
    Use the sidebar to select different pages and adjust analysis parameters.
    """)

    # Quick start guide
    with st.expander("📘 Quick Start Guide"):
        st.markdown("""
        1. **Select Parameters**: Use the sidebar to choose years, categories, and threshold
        2. **Explore Networks**: Navigate to the Networks page to visualize city connections
        3. **View Rankings**: Check the Rankings page for top cities under different schemes
        4. **Analyze Trends**: Visit the Temporal Evolution page to see changes over time
        5. **Custom Analysis**: Create your own weighting scheme on the Custom Analysis page
        """)


def show_network_demo():
    """Display network visualization demo."""
    st.markdown('<div class="main-header">🌐 Network Visualization</div>', unsafe_allow_html=True)

    st.info("📌 **Note**: This is a demonstration with synthetic data. "
           "To analyze real CPI data, implement the preprocessing module and load actual datasets.")

    # Demo: Create a simple test network
    st.subheader("Sample Network")

    # Create a demo graph
    G = nx.karate_club_graph()
    # Relabel nodes with city names (use first N cities)
    n_nodes = min(len(CITIES), G.number_of_nodes())
    mapping = {i: CITIES[i] for i in range(n_nodes)}
    G = nx.relabel_nodes(G, mapping)
    G = nx.Graph(G.subgraph(list(mapping.values())))

    # Network statistics
    col1, col2, col3 = st.columns(3)

    stats = get_network_statistics(G)

    with col1:
        st.metric("Nodes", stats['num_nodes'])

    with col2:
        st.metric("Edges", stats['num_edges'])

    with col3:
        st.metric("Density", f"{stats['density']:.3f}")

    # Visualization
    st.subheader("Network Graph")

    layout = st.selectbox(
        "Layout Algorithm",
        options=['spring', 'circular', 'kamada_kawai'],
        index=0
    )

    fig = plot_network(G, "Sample City Network", layout=layout)
    st.pyplot(fig)

    # Network metrics table
    st.subheader("Network Metrics")

    # Calculate centralities
    df_centrality = calculate_single_network_centrality(G)
    st.dataframe(
        df_centrality.sort_values('D_i', ascending=False),
        use_container_width=True
    )


def show_custom_analysis():
    """Display custom weighting analysis page."""
    st.markdown('<div class="main-header">⚙️ Custom Analysis</div>', unsafe_allow_html=True)

    st.write("""
    Create your own custom weighting scheme by adjusting the importance of each centrality metric.
    """)

    st.subheader("Adjust Weights")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Weight sliders
        st.write("**Centrality Metric Weights**")

        w_degree = st.slider(
            "Degree Centrality (D_i)",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="How many direct connections"
        )

        w_closeness = st.slider(
            "Closeness Centrality (C_i)",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Average distance to other cities"
        )

        w_betweenness = st.slider(
            "Betweenness Centrality (B_i)",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="How often city lies on shortest paths"
        )

        w_eigenvector = st.slider(
            "Eigenvector Centrality (E_i)",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Influence based on important connections"
        )

    with col2:
        # Weight summary
        total_weight = w_degree + w_closeness + w_betweenness + w_eigenvector

        st.write("**Weight Summary**")
        st.metric("Total Weight", f"{total_weight:.2f}")

        if abs(total_weight - 1.0) > 0.01:
            st.warning("⚠️ Weights should sum to 1.0")
        else:
            st.success("✅ Weights sum to 1.0")

        # Normalize button
        if st.button("Normalize Weights"):
            st.info("Weights will be normalized automatically when applied")

    st.markdown("---")

    st.info("💡 **Tip**: This demo shows how the custom weighting interface works. "
           "With real data, you would see updated rankings based on your weight choices.")


def show_about():
    """Display about page."""
    st.markdown('<div class="main-header">ℹ️ About</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Pakistan CPI Network Analysis

    This application analyzes Consumer Price Index (CPI) data from Pakistan using network science
    techniques to understand price similarity patterns across cities and product categories.

    ### Methodology

    1. **Similarity Calculation**: Compute cosine similarity between city price vectors
    2. **Network Construction**: Create networks based on similarity thresholds
    3. **Centrality Analysis**: Calculate four centrality metrics for each city
    4. **Weighting Schemes**: Combine metrics using five different approaches
    5. **Temporal Analysis**: Track network evolution over time

    ### Centrality Metrics

    - **Degree (D_i)**: Number of direct connections
    - **Closeness (C_i)**: Average distance to all other cities
    - **Betweenness (B_i)**: Frequency of appearing on shortest paths
    - **Eigenvector (E_i)**: Influence based on connections to important cities

    ### Weighting Schemes

    1. **Equal**: All metrics weighted equally (0.25 each)
    2. **Correlation-Based**: Down-weight correlated metrics
    3. **Category Importance**: Weight by economic significance
    4. **Entropy-Based**: Weight by information content
    5. **Custom**: User-defined weights

    ### Technical Stack

    - **Backend**: Python, NetworkX, NumPy, Pandas
    - **Frontend**: Streamlit
    - **Visualization**: Matplotlib, Seaborn, Plotly

    ### Project Structure

    ```
    Pakistan-cpi-network-analysis/
    ├── src/
    │   ├── similarity.py      # Similarity calculations
    │   ├── networks.py        # Network construction
    │   ├── centrality.py      # Centrality metrics
    │   ├── weighting.py       # Weighting schemes
    │   ├── temporal.py        # Temporal analysis
    │   └── visualization.py   # Plotting functions
    ├── pipeline.py            # Main analysis pipeline
    └── app.py                 # This Streamlit app
    ```

    ### Citation

    If you use this analysis in your research, please cite:
    ```
    Pakistan CPI Network Analysis (2024)
    GitHub: github.com/your-repo/Pakistan-cpi-network-analysis
    ```
    """)


def main():
    """Main application entry point."""

    # Sidebar controls
    params = setup_sidebar()

    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.header("📑 Navigation")

    page = st.sidebar.radio(
        "Select Page",
        options=[
            "Overview",
            "Network Demo",
            "Custom Analysis",
            "About"
        ],
        label_visibility="collapsed"
    )

    # Route to appropriate page
    if page == "Overview":
        show_overview()
    elif page == "Network Demo":
        show_network_demo()
    elif page == "Custom Analysis":
        show_custom_analysis()
    elif page == "About":
        show_about()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p>Pakistan CPI Network Analysis</p>
        <p>© 2024 | Version 1.0</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
