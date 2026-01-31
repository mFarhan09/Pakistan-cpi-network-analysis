# 📊 Automated Inflation Analysis Pipeline

A fully automated data-to-insight pipeline that continuously analyzes inflation dynamics across major Pakistani cities using network science, mathematical modeling, and interactive visualization.

---

## 📌 Project Summary

This project implements a fully automated data-to-insight pipeline that continuously analyzes inflation dynamics across major Pakistani cities using network science, mathematical modeling, and interactive visualization.

**Unlike static analysis projects, this system is designed as a living pipeline:**

- It scrapes fresh CPI data from the Pakistan Bureau of Statistics (PBS)
- Preprocesses and normalizes the data
- Converts inflation patterns into city-to-city networks
- Computes graph-based importance metrics
- Presents results through an interactive Streamlit dashboard
- Remains relevant and up-to-date for years, not months

---

##  Why This Matters 

✔ Automatically updates with new government data  
✔ No manual Excel handling after setup  
✔ Suitable for policy analysis, research, dashboards, and decision support  
✔ Demonstrates data engineering + math + visualization in one system  

**This is not a report — it is a repeatable analytical product.**

---

##  System Architecture: The Inflation Analysis Pipeline

```
Pakistan Bureau of Statistics Website
            ↓
Web Scraping (Monthly CPI Data)
            ↓
Data Cleaning & Reshaping
            ↓
Statistical Normalization
            ↓
Similarity Computation
            ↓
Network Construction
            ↓
Centrality & Scoring Models
            ↓
Aggregation & Ranking
            ↓
Streamlit Interactive Dashboard
            ↓
Up-to-Date Visual Insights
```

---

## 1️ Data Acquisition (Web Scraping Layer)

### Source

- Pakistan Bureau of Statistics (PBS)
- Monthly Consumer Price Index (CPI) data
- Multiple categories (Food, Utilities, Transport, Housing, etc.)
- Major Pakistani cities

### Method

- Automated web scraping of the latest published CPI tables
- Designed to run monthly (or on-demand)
- Eliminates dependence on static datasets

### 📌 Key Advantage

Because the data is scraped directly from PBS:

- The system remains relevant even after 1–2 years
- Visualizations automatically update as new data is released
- No re-engineering required for future inflation cycles

---

## 2️ Data Preprocessing & Reshaping

### Steps

- Parsing scraped tables into structured format
- Converting wide-format CPI data into relational (long) format
- Extracting temporal features (Year, Month)
- Tagging CPI categories

**Output:** Clean, structured dataset ready for mathematical analysis.

---

## 3️ Statistical Normalization

### Problem

CPI items exist on vastly different scales (e.g., wheat vs electricity).

### Solution: Z-Score Normalization

```
Z = (X − μ) / σ
```

**Where:**
- `X` = observed CPI value
- `μ` = mean
- `σ` = standard deviation

### Result

- Scale-independent comparison
- Each city represented as a normalized inflation behavior vector
- Enables valid similarity measurement

---

## 4️ Similarity Computation

### Technique: Cosine Similarity

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

### Interpretation

- `+1` → identical inflation behavior
- `0` → unrelated trends
- `−1` → opposing patterns

**Output:** City-to-city similarity matrices for each category and time period.

---

## 5️ Network Construction

### Graph Model

- **Nodes:** Cities
- **Edges:** Strong inflation similarity

### Edge Condition

```
similarity ≥ τ
```

where `τ = 0.75`

### Properties

- Sparse but meaningful graphs
- Captures real economic relationships
- Supports weighted or unweighted analysis

---

## 6️ Network Centrality Analysis (Core Intelligence Layer)

Each city's role in the inflation system is measured using four complementary metrics:

### 🔹 Degree Centrality

Inflation hubs with widespread similarity.

```
D(i) = degree(i) / (n − 1)
```

### 🔹 Closeness Centrality

Speed at which inflation signals spread.

```
C(i) = (n − 1) / Σ d(i, j)
```

### 🔹 Betweenness Centrality

Cities acting as bridges between regions.

```
B(i) = Σ σ(s, t | i) / σ(s, t)
```

### 🔹 Eigenvector Centrality

Influence via connection to other influential cities.

```
Ax = λx
```

---

## 7️ Scoring & Weighting Models

Multiple aggregation strategies are supported:

### ✔ Equal Weighting

Neutral baseline.

```
Score = 0.25(D + C + B + E)
```

### ✔ Correlation-Based Weighting

Reduces redundancy between metrics.

### ✔ Category-Importance Weighting

Prioritizes economically sensitive categories (e.g., Food, Utilities).

### ✔ Entropy-Based Weighting

Higher weight to metrics with higher information content.

---

## 8️ Streamlit Visualization Layer

### Technology

- Python + Streamlit
- Acts as the presentation and interaction layer of the pipeline

### Visual Outputs

- Interactive city networks
- Similarity heatmaps
- City importance rankings
- Temporal evolution of inflation influence

### 📊 User Experience

- Non-technical users can explore results
- Filters by category, city, and time
- Instant updates when new data is scraped

---

## 🔄 Long-Term Relevance & Automation

✔ Monthly web scraping keeps data fresh  
✔ Pipeline reruns automatically on new data  
✔ Visualizations update without manual effort  
✔ System remains valid years into the future  

### This makes the project suitable for:

- Research institutions
- Policy think tanks
- NGOs
- Data-driven consulting
- Freelance analytics dashboards

---

##  Key Insight

By combining web scraping, mathematical modeling, network science, and interactive visualization into a single pipeline, this system transforms raw CPI releases into continuously updated economic intelligence.

---

## 👤 Author

**Muhammad Farhan**  
Bachelor's in Computer Science  
Focus: Data Pipelines, Network Science, Applied Analytics

---
