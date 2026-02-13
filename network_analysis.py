"""
Complex Network Analysis - Bitcoin Alpha Signed Trust Network
Complete analysis with metrics, visualizations, and report generation
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("COMPLEX NETWORK ANALYSIS - BITCOIN ALPHA SIGNED TRUST NETWORK")
print("=" * 80)

# ============================================================================
# SECTION 2 - DATA LOADING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2 - DATA LOADING")
print("=" * 80)

# Load CSV file (no header)
df = pd.read_csv('soc-sign-bitcoinalpha.csv', header=None)

# Assign column names manually
df.columns = ["source", "target", "rating", "time"]

print("\nFirst 5 rows:")
print(df.head())

print(f"\nDataset shape: {df.shape}")
print(f"Number of rows: {len(df)}")

# Verify node ids are integers
print(f"\nSource column type: {df['source'].dtype}")
print(f"Target column type: {df['target'].dtype}")

# Build directed graph
G = nx.DiGraph()

# Add edges with rating as weight
for _, row in df.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['rating'])

print(f"\nGraph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ============================================================================
# SECTION 3 - BASIC GRAPH INFO
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3 - BASIC GRAPH INFO")
print("=" * 80)


def compute_basic_metrics(G):
    """Compute basic graph metrics"""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Directed density
    max_edges = n_nodes * (n_nodes - 1)
    density = n_edges / max_edges if max_edges > 0 else 0

    # Average degree (for directed: total edges / nodes)
    avg_degree = (2 * n_edges) / n_nodes if n_nodes > 0 else 0

    # Max degree
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 0

    # Degree variance
    degree_values = list(degrees.values())
    degree_variance = np.var(degree_values) if degree_values else 0

    return {
        'number_of_nodes': n_nodes,
        'number_of_edges': n_edges,
        'directed_density': density,
        'average_degree': avg_degree,
        'max_degree': max_degree,
        'degree_variance': degree_variance
    }


basic_metrics = compute_basic_metrics(G)

print("\nBasic Graph Metrics:")
for key, value in basic_metrics.items():
    print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

# Store in dataframe
basic_metrics_df = pd.DataFrame([basic_metrics])
print("\nBasic Metrics DataFrame:")
print(basic_metrics_df)

# ============================================================================
# SECTION 4 - COMPONENT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4 - COMPONENT ANALYSIS")
print("=" * 80)

# Weakly connected components
wcc = list(nx.weakly_connected_components(G))
wcc_sizes = [len(comp) for comp in wcc]
largest_wcc_size = max(wcc_sizes) if wcc_sizes else 0
max_wcc = max(wcc, key=len) if wcc else set()

print(f"\nNumber of weakly connected components: {len(wcc)}")
print(f"Largest WCC size: {largest_wcc_size}")

# Strongly connected components
scc = list(nx.strongly_connected_components(G))
scc_sizes = [len(comp) for comp in scc]
largest_scc_size = max(scc_sizes) if scc_sizes else 0

print(f"\nNumber of strongly connected components: {len(scc)}")
print(f"Largest SCC size: {largest_scc_size}")

# Create subgraph of largest WCC
G_lcc = G.subgraph(max_wcc).copy()
print(f"\nLargest WCC subgraph: {G_lcc.number_of_nodes()} nodes, {G_lcc.number_of_edges()} edges")

# ============================================================================
# SECTION 5 - DISTANCE METRICS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5 - DISTANCE METRICS (on largest WCC)")
print("=" * 80)


def compute_distance_metrics(G_lcc):
    """Compute distance metrics on largest WCC"""
    print("Computing distance metrics (this may take a while)...")

    # Check if graph is connected
    if not nx.is_weakly_connected(G_lcc):
        print("Warning: Graph is not weakly connected, using largest component")
        wcc_lcc = list(nx.weakly_connected_components(G_lcc))
        G_lcc = G_lcc.subgraph(max(wcc_lcc, key=len)).copy()

    # Convert to undirected for distance calculations
    G_undirected = G_lcc.to_undirected()

    # For large graphs, use approximate algorithms
    n_nodes = G_undirected.number_of_nodes()

    if n_nodes > 1000:
        print(f"Large graph ({n_nodes} nodes), using approximate algorithms...")

        # Approximate diameter using sampling
        try:
            diameter = nx.approximation.diameter(G_undirected)
        except:
            # Fallback: sample shortest paths
            sample_nodes = list(G_undirected.nodes())[:min(100, n_nodes)]
            max_path = 0
            for node in sample_nodes:
                paths = nx.single_source_shortest_path_length(G_undirected, node)
                if paths:
                    max_path = max(max_path, max(paths.values()))
            diameter = max_path

        # Approximate radius
        try:
            radius = nx.approximation.radius(G_undirected)
        except:
            # Fallback: sample eccentricities
            sample_nodes = list(G_undirected.nodes())[:min(100, n_nodes)]
            min_ecc = float('inf')
            for node in sample_nodes:
                ecc = nx.eccentricity(G_undirected, node)
                min_ecc = min(min_ecc, ecc)
            radius = min_ecc if min_ecc != float('inf') else 0

        # Approximate average shortest path length
        print("Computing average shortest path length (sampling)...")
        sample_size = min(100, n_nodes)
        sample_nodes = np.random.choice(list(G_undirected.nodes()),
                                        size=min(sample_size, n_nodes),
                                        replace=False)
        path_lengths = []
        for node in sample_nodes:
            paths = nx.single_source_shortest_path_length(G_undirected, node)
            path_lengths.extend([v for v in paths.values() if v > 0])

        avg_shortest_path = np.mean(path_lengths) if path_lengths else 0
    else:
        # Exact computation for smaller graphs
        print("Computing exact distance metrics...")
        diameter = nx.diameter(G_undirected)
        radius = nx.radius(G_undirected)
        avg_shortest_path = nx.average_shortest_path_length(G_undirected)

    return {
        'diameter': diameter,
        'radius': radius,
        'average_shortest_path_length': avg_shortest_path
    }


distance_metrics = compute_distance_metrics(G_lcc)

print("\nDistance Metrics:")
for key, value in distance_metrics.items():
    print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

# ============================================================================
# SECTION 6 - CLUSTERING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6 - CLUSTERING")
print("=" * 80)

# Convert to undirected for clustering
G_u = G.to_undirected()

# Global clustering coefficient (transitivity)
clustering_coeff = nx.transitivity(G_u)

print(f"\nGlobal Clustering Coefficient (Transitivity): {clustering_coeff:.6f}")

# ============================================================================
# SECTION 7 - FREEMAN DEGREE CENTRALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7 - FREEMAN DEGREE CENTRALIZATION")
print("=" * 80)


def compute_freeman_centralization(G):
    """Compute Freeman degree centralization"""
    degrees = dict(G.degree())
    if not degrees:
        return 0

    max_degree = max(degrees.values())
    degree_values = list(degrees.values())
    n = len(degree_values)

    # Theoretical maximum: (n-1) * (n-2) for directed graph
    # For undirected: (n-1) * (n-2)
    theoretical_max = (n - 1) * (n - 2) if n > 2 else 1

    # Sum of (max_degree - degree_i)
    sum_diff = sum(max_degree - d for d in degree_values)

    # Freeman centralization
    freeman_centralization = sum_diff / theoretical_max if theoretical_max > 0 else 0

    return freeman_centralization


freeman_centralization = compute_freeman_centralization(G)

print(f"\nFreeman Degree Centralization: {freeman_centralization:.6f}")

# ============================================================================
# SECTION 8 - CENTRALITY METRICS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 8 - CENTRALITY METRICS")
print("=" * 80)

print("Computing centrality metrics for all nodes (this may take a while)...")

# Degree centrality
print("  Computing degree centrality...")
degree_centrality = nx.degree_centrality(G)

# Closeness centrality (on largest WCC for connected graph)
print("  Computing closeness centrality...")
try:
    # Compute for all nodes in largest WCC at once
    closeness_centrality_lcc = nx.closeness_centrality(G_lcc)
    # Extend to all nodes
    closeness_centrality = {n: closeness_centrality_lcc.get(n, 0) for n in G.nodes()}
except:
    print("    Closeness centrality computation failed, using approximate method...")
    closeness_centrality = {n: 0 for n in G.nodes()}
    # Compute only for sample of nodes
    sample_nodes = list(G_lcc.nodes())[:min(500, G_lcc.number_of_nodes())]
    for node in sample_nodes:
        try:
            closeness_centrality[node] = nx.closeness_centrality(G_lcc, node)
        except:
            closeness_centrality[node] = 0

# Betweenness centrality (approximate for large graphs)
print("  Computing betweenness centrality...")
if G.number_of_nodes() > 1000:
    print("    Using approximate betweenness (sampling)...")
    sample_size = min(100, G.number_of_nodes())
    sample_nodes = np.random.choice(list(G.nodes()),
                                    size=min(sample_size, G.number_of_nodes()),
                                    replace=False)
    betweenness_centrality = nx.betweenness_centrality(G, k=sample_size)
else:
    betweenness_centrality = nx.betweenness_centrality(G)

# Eigenvector centrality
print("  Computing eigenvector centrality...")
try:
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
except:
    print("    Eigenvector centrality failed, using PageRank as fallback")
    eigenvector_centrality = nx.pagerank(G)

# PageRank - compute on largest WCC for efficiency
print("  Computing PageRank...")
try:
    # Compute on largest WCC (more efficient and usually sufficient)
    # Use a more lenient tolerance for convergence
    pagerank_lcc = nx.pagerank(G_lcc, max_iter=500, tol=1e-02, damping=0.85)
    # Extend to all nodes
    pagerank = {n: pagerank_lcc.get(n, 0) for n in G.nodes()}
    print("    PageRank computed successfully on largest WCC")
except Exception as e:
    try:
        print(f"    PageRank on WCC failed ({str(e)}), trying with even more relaxed parameters...")
        pagerank_lcc = nx.pagerank(G_lcc, max_iter=1000, tol=1e-01, damping=0.85)
        pagerank = {n: pagerank_lcc.get(n, 0) for n in G.nodes()}
        print("    PageRank computed with relaxed parameters")
    except:
        try:
            print("    Trying full graph with very relaxed parameters...")
            pagerank = nx.pagerank(G, max_iter=500, tol=1e-01, damping=0.85)
            print("    PageRank computed on full graph")
        except:
            print("    PageRank failed completely, using normalized degree centrality as fallback...")
            # Use normalized degree as fallback
            max_deg = max(degree_centrality.values()) if degree_centrality.values() else 1
            pagerank = {n: degree_centrality.get(n, 0) / max_deg if max_deg > 0 else 0
                        for n in G.nodes()}

# Create centrality dataframe
centrality_df = pd.DataFrame({
    'node': list(G.nodes()),
    'degree': [degree_centrality.get(n, 0) for n in G.nodes()],
    'closeness': [closeness_centrality.get(n, 0) for n in G.nodes()],
    'betweenness': [betweenness_centrality.get(n, 0) for n in G.nodes()],
    'eigenvector': [eigenvector_centrality.get(n, 0) for n in G.nodes()],
    'pagerank': [pagerank.get(n, 0) for n in G.nodes()]
})

print(f"\nCentrality DataFrame shape: {centrality_df.shape}")
print("\nFirst 5 rows of centrality DataFrame:")
print(centrality_df.head())

# ============================================================================
# SECTION 9 - TOP NODES
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 9 - TOP NODES")
print("=" * 80)


def get_top_nodes(centrality_df, metric, n=3):
    """Get top N nodes for a given centrality metric"""
    top = centrality_df.nlargest(n, metric)[['node', metric]]
    return top


top_nodes = {}
for metric in ['degree', 'closeness', 'betweenness', 'eigenvector', 'pagerank']:
    top = get_top_nodes(centrality_df, metric, n=3)
    top_nodes[metric] = top
    print(f"\nTop 3 nodes by {metric} centrality:")
    print(top)

# ============================================================================
# SECTION 10 - GRAPH VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 10 - GRAPH VISUALIZATION")
print("=" * 80)

print("Creating graph visualization (this may take a while)...")

# Use largest WCC for visualization
G_viz = G_lcc.copy()

# Get node attributes
nodes = list(G_viz.nodes())
pagerank_values = [pagerank.get(n, 0) for n in nodes]
degree_cent_values = [degree_centrality.get(n, 0) for n in nodes]

# Normalize for visualization
pagerank_norm = np.array(pagerank_values)
pagerank_norm = (pagerank_norm - pagerank_norm.min()) / (pagerank_norm.max() - pagerank_norm.min() + 1e-10)
node_sizes = 100 + 5000 * pagerank_norm

degree_cent_norm = np.array(degree_cent_values)
degree_cent_norm = (degree_cent_norm - degree_cent_norm.min()) / (
            degree_cent_norm.max() - degree_cent_norm.min() + 1e-10)

# Get top 20 nodes by PageRank for labels
top_20_nodes = centrality_df.nlargest(20, 'pagerank')['node'].tolist()
node_labels = {n: str(n) if n in top_20_nodes else '' for n in nodes}

# Layout
print("  Computing layout...")
pos = nx.spring_layout(G_viz, seed=42, k=0.1, iterations=50)

# Create figure
plt.figure(figsize=(20, 20))
nx.draw_networkx_nodes(G_viz, pos,
                       node_size=node_sizes,
                       node_color=degree_cent_norm,
                       cmap=plt.cm.viridis,
                       alpha=0.7)
nx.draw_networkx_edges(G_viz, pos,
                       alpha=0.1,
                       width=0.5,
                       arrows=True,
                       arrowsize=10)
nx.draw_networkx_labels(G_viz, pos,
                        labels=node_labels,
                        font_size=8,
                        font_weight='bold')

plt.title('Bitcoin Alpha Network Visualization\n(Node size: PageRank, Color: Degree Centrality)',
          fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('graph_visualization.png', dpi=300, bbox_inches='tight')
print("  Saved: graph_visualization.png")
plt.close()

# ============================================================================
# SECTION 11 - DISTRIBUTION PLOTS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 11 - DISTRIBUTION PLOTS")
print("=" * 80)

# Degree distribution
degrees = [G.degree(n) for n in G.nodes()]
degree_counts = pd.Series(degrees).value_counts().sort_index()

# Plot 1: Degree distribution histogram
plt.figure(figsize=(12, 6))
plt.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Degree', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Degree Distribution Histogram', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('degree_distribution_histogram.png', dpi=300, bbox_inches='tight')
print("  Saved: degree_distribution_histogram.png")
plt.close()

# Plot 2: Log-log degree distribution
plt.figure(figsize=(12, 6))
nonzero_degrees = [d for d in degree_counts.index if d > 0]
nonzero_counts = [degree_counts[d] for d in nonzero_degrees]
plt.loglog(nonzero_degrees, nonzero_counts, 'o', markersize=6, alpha=0.7)
plt.xlabel('Degree (log scale)', fontsize=12)
plt.ylabel('Frequency (log scale)', fontsize=12)
plt.title('Log-Log Degree Distribution', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('degree_distribution_loglog.png', dpi=300, bbox_inches='tight')
print("  Saved: degree_distribution_loglog.png")
plt.close()

# Plot 3: Centrality distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

centrality_metrics = ['degree', 'closeness', 'betweenness', 'eigenvector', 'pagerank']
for idx, metric in enumerate(centrality_metrics):
    ax = axes[idx]
    values = centrality_df[metric].values
    values = values[values > 0]  # Remove zeros for better visualization
    ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel(metric.capitalize(), fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{metric.capitalize()} Centrality Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Remove empty subplot
axes[5].axis('off')

plt.tight_layout()
plt.savefig('centrality_distributions.png', dpi=300, bbox_inches='tight')
print("  Saved: centrality_distributions.png")
plt.close()

# Plot 4: Bar chart of top nodes per metric
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, metric in enumerate(centrality_metrics):
    ax = axes[idx]
    top = get_top_nodes(centrality_df, metric, n=10)
    ax.barh(range(len(top)), top[metric].values, alpha=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['node'].values, fontsize=8)
    ax.set_xlabel(metric.capitalize(), fontsize=10)
    ax.set_title(f'Top 10 Nodes by {metric.capitalize()}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

axes[5].axis('off')

plt.tight_layout()
plt.savefig('top_nodes_bar_charts.png', dpi=300, bbox_inches='tight')
print("  Saved: top_nodes_bar_charts.png")
plt.close()

# ============================================================================
# SECTION 12 - CENTRALITY CORRELATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 12 - CENTRALITY CORRELATION")
print("=" * 80)

# Compute correlations
correlation_metrics = ['degree', 'closeness', 'betweenness', 'eigenvector', 'pagerank']
corr_data = centrality_df[correlation_metrics]

# Pearson correlation
pearson_corr = corr_data.corr(method='pearson')
print("\nPearson Correlation Matrix:")
print(pearson_corr)

# Spearman correlation
spearman_corr = corr_data.corr(method='spearman')
print("\nSpearman Correlation Matrix:")
print(spearman_corr)

# Create heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pearson heatmap
sns.heatmap(pearson_corr, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, ax=axes[0], cbar_kws={'label': 'Correlation'})
axes[0].set_title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')

# Spearman heatmap
sns.heatmap(spearman_corr, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, ax=axes[1], cbar_kws={'label': 'Correlation'})
axes[1].set_title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('centrality_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n  Saved: centrality_correlation_heatmap.png")
plt.close()

# ============================================================================
# SECTION 13 - RESULTS TABLE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 13 - RESULTS TABLE")
print("=" * 80)

# Compile all metrics
final_metrics = {
    'nodes': basic_metrics['number_of_nodes'],
    'edges': basic_metrics['number_of_edges'],
    'density': basic_metrics['directed_density'],
    'diameter': distance_metrics['diameter'],
    'radius': distance_metrics['radius'],
    'avg_degree': basic_metrics['average_degree'],
    'degree_variance': basic_metrics['degree_variance'],
    'freeman_centralization': freeman_centralization,
    'max_degree': basic_metrics['max_degree'],
    'largest_WCC': largest_wcc_size,
    'largest_SCC': largest_scc_size,
    'clustering_coefficient': clustering_coeff
}

metrics_table = pd.DataFrame([final_metrics])
print("\nFinal Metrics Table:")
print(metrics_table)

# Export to CSV
metrics_table.to_csv('metrics_table.csv', index=False)
print("\n  Saved: metrics_table.csv")

# Export centrality table
centrality_df.to_csv('centrality_table.csv', index=False)
print("  Saved: centrality_table.csv")


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nAll output files have been generated successfully.")
