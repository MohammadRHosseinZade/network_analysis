# Complex Networks Analysis — Bitcoin Alpha Trust Network

Analysis of the **Bitcoin Alpha Signed Trust Network** dataset. This repository contains graph loading, preprocessing, visualization, structural metrics, centrality analysis, and correlation studies for a large signed directed social network.

---

## Dataset

* **Source:** SNAP network datasets
* **Dataset Name:** Bitcoin Alpha signed trust network
* **File:** `soc-sign-bitcoinalpha.csv`
* **Type:** Directed, weighted, signed network
* **Format:** CSV (no header row)

### Dataset Columns

* `source` — source node ID
* `target` — target node ID
* `rating` — signed trust score between users
* `time` — timestamp of rating

### Dataset Size

* **Total Nodes:** 3,783
* **Total Edges:** 24,186

The network represents trust relationships between users of the Bitcoin Alpha platform, where edges encode both direction and signed trust ratings.

---

## Installation

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

Install required packages:

```bash
pip install -r requirements.txt
```

If you don’t have a requirements file yet, install the core libraries manually:

```bash
pip install networkx pandas numpy matplotlib seaborn scipy jupyter
```

---

## Jupyter Notebook

The full analysis is also provided as a notebook:

```
bitcoin_alpha_analysis.ipynb
```

Run it with:

```bash
jupyter notebook
```

Then open the notebook and run all cells to reproduce:

* Graph construction
* Metrics computation
* Centrality analysis
* Plots and visualizations
* Correlation matrices

---

## Project Features

This project performs a complete complex network analysis pipeline:

* Graph construction from CSV edge list
* Extraction of largest weakly connected component
* Network visualization with centrality-based styling
* Structural metrics computation
* Centrality analysis (5 measures)
* Degree distribution analysis (linear + log-log)
* Centrality distribution plots
* Correlation analysis between centrality metrics
* Heatmap visualization

---

## Graph Visualization

The generated network visualization includes:

* Node size proportional to **PageRank**
* Node color based on **degree centrality**
* Labels shown for top PageRank nodes
* Layout optimized for readability of the giant component

Output file:

```
graph_visualization.png
```

---

## Structural Metrics Computed

The analysis computes key global network properties:

* Number of nodes and edges
* Network density
* Diameter and radius
* Average degree and degree variance
* Largest WCC and SCC sizes
* Average shortest path length
* Global clustering coefficient
* Freeman degree centralization
* Maximum degree

Saved as:

```
metrics_table.csv
```

---

## Centrality Measures

The following node importance metrics are computed:

* Degree centrality
* Closeness centrality
* Betweenness centrality
* Eigenvector centrality
* PageRank

Outputs include:

* Full centrality table for all nodes
* Top-k node rankings
* Distribution plots
* Comparison bar charts

Saved as:

```
centrality_table.csv
top_nodes_bar_charts.png
centrality_distributions.png
```

---

## Distribution Analysis

The project evaluates whether the network shows scale-free behavior using:

* Degree histogram
* Log–log degree distribution plot

Outputs:

```
degree_distribution_histogram.png
degree_distribution_loglog.png
```

---

## Correlation Analysis

Correlation between centrality measures is computed using:

* Pearson correlation matrix
* Spearman rank correlation matrix
* Heatmap visualization

Outputs:

```
centrality_correlation_heatmap.png
```

---

## Repository Outputs

The analysis generates the following artifacts:

```
metrics_table.csv
centrality_table.csv
graph_visualization.png
degree_distribution_histogram.png
degree_distribution_loglog.png
centrality_distributions.png
top_nodes_bar_charts.png
centrality_correlation_heatmap.png
report.md
bitcoin_alpha_analysis.ipynb
```

---

## Typical Workflow

1. Load Bitcoin Alpha CSV dataset
2. Build directed signed graph
3. Extract giant component
4. Compute structural metrics
5. Compute centralities
6. Generate plots and visualizations
7. Run correlation analysis
8. Export tables and figures

---

## Use Cases

This project is suitable for:

* Complex network coursework
* Social trust network analysis
* Centrality comparison studies
* Scale-free network investigation
* Graph analytics pipelines

---

## License / Citation

Please cite the SNAP dataset source when using this data in academic work.
