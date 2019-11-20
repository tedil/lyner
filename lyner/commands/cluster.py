import logging
import sys
from itertools import product

import click
import numpy as np
import pandas as pd
from pandas import MultiIndex
from plotly.offline import plot as oplot
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist

from lyner._main import rnax
from lyner.click_extras import MutexOption, pass_pipe, arggist, Pipe, CombinatorialChoice, DICT
from lyner.custom_matrices import cluster_targets
from lyner.plotly_extras import _mk_networkx_figure

LOGGER = logging.getLogger("lyner")


@rnax.command()
@click.option('--method', '-m',
              type=click.Choice(['dbscan', 'k_means', 'mean_shift']),
              default='k_means')
@click.option('--num-clusters', '-n', type=click.INT, default=4,
              help="The exact number of clusters to build.")
@click.option("--mode-config", "-c", type=DICT, default={})
@pass_pipe
@arggist
def cluster(pipe: Pipe, num_clusters: int, method: str, mode_config: dict):
    """Clustering via k_mean / dbscan / mean_shift."""
    import sklearn
    clustering = getattr(sklearn.cluster, method)
    if method == 'k_means':
        mode_config['n_clusters'] = num_clusters
    centroids, labels, *_ = clustering(pipe.matrix.values, **mode_config)
    rows = pipe.matrix.index.values
    pipe.matrix.index = pd.MultiIndex.from_tuples(list(zip(labels, rows)))
    pipe.matrix.sort_index(level=0, axis=0, inplace=True)
    cluster_indices(pipe)


@rnax.command()
@click.option('--by', '-b', type=CombinatorialChoice(['trend', 'mean', 'median', 'mad', 'var']),
              default='trend',
              help="Any comma separated combination of: "
                   "'trend', 'mean', 'median', 'mad', 'var', 'ontology'. "
                   "Order is relevant.")
@click.option('--min-nclusters', '-l', type=click.INT, default=3, cls=MutexOption, mutually_exclusive=['nclusters'],
              help="The minimum number of clusters to build.")
@click.option('--max-nclusters', '-u', type=click.INT, default=20, cls=MutexOption, mutually_exclusive=['nclusters'],
              help="The maximum number of clusters to build.")
@click.option('--nclusters', '-n', type=click.INT, default=0, cls=MutexOption,
              mutually_exclusive=['min_nclusters', 'max_nclusters'],
              help="The exact number of clusters to build.")
@pass_pipe
@arggist
def cluster_agglomerative(pipe: Pipe, by: list, min_nclusters: int, max_nclusters: int, nclusters: int):
    """Agglomerative clustering."""
    matrix = pipe.matrix
    if not (0 < min_nclusters <= max_nclusters):
        print(f"[WARNING] 0 >= min_clusters ({min_nclusters}) > max_nclusters ({max_nclusters}). Swapping.",
              file=sys.stderr)
        min_nclusters, max_nclusters = min(min_nclusters, max_nclusters), max(min_nclusters, max_nclusters)
    if nclusters > 0:
        min_nclusters = max_nclusters = nclusters

    cluster_indices_list, cluster_labels = cluster_targets(matrix,
                                                           by=by,
                                                           min_nclusters=min_nclusters,
                                                           max_nclusters=max_nclusters)
    matrix['cluster'] = cluster_labels
    matrix['median'] = np.median(matrix, axis=1)
    matrix = matrix.sort_values(by=['cluster', 'median'], kind='mergesort')
    cluster_labels = matrix[['cluster']]

    labels = np.unique(cluster_labels)
    clusters = {label: matrix[(matrix[['cluster']] == label)['cluster']].dropna(how='all').index for label in labels}
    matrix = matrix.drop(['median', 'cluster'], axis=1)
    pipe.matrix = matrix

    # tuples need to be sorted lexicographically to enable multiindex slicing
    tuples = [(cluster_label, gene_id) for cluster_label, clust_idx in clusters.items() for gene_id in clust_idx]
    row_index = MultiIndex.from_tuples(tuples, names=['Cluster', 'Gene'])
    pipe.matrix.index = row_index
    pipe.matrix = pipe.matrix.sort_index(kind='mergesort')
    pipe.is_clustered = True
    cluster_indices(pipe)


@rnax.command()
@click.option('--method', '-m',
              type=click.Choice(['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']),
              default='ward')
@click.option('--distance-metric', '-d',
              type=click.Choice(
                  ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean',
                   'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
                   'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']),
              default='euclidean')
@click.option('--criterion', '-c',
              type=click.Choice(['inconsistent', 'distance', 'maxclust', 'monocrit', 'maxclust_monocrit']),
              default='inconsistent')
@click.option('--threshold', '-t', type=click.FLOAT, default=0.8)
@pass_pipe
@arggist
def cluster_hierarchical(pipe: Pipe, method: str, distance_metric: str, threshold: float, criterion: str):
    """Hierarchical clustering"""
    LOGGER.info(f"Calculating {method} linkage using {distance_metric} metric.")
    l = linkage(pipe.matrix.values, method=method, metric=distance_metric)
    clusters = fcluster(l, threshold, criterion=criterion)
    LOGGER.info(f"Found {len(np.unique(clusters))} clusters using criterion {criterion} "
                f"with a parameter/threshold of {threshold}.")
    pipe.matrix.index = pd.MultiIndex.from_tuples(list(zip(clusters, pipe.matrix.index.values)),
                                                  names=["Cluster", "Feature"])
    pipe.matrix = pipe.matrix.sort_index()
    pipe.is_clustered = True
    cluster_indices(pipe)


def cluster_indices(pipe: Pipe):
    attr = 'cluster_indices'
    if pipe.matrix.index.nlevels > 1:
        labels = pipe.matrix.index.get_level_values(0).values
        names = pipe.matrix.index.get_level_values(1).values
        attr += '_samples'
    elif pipe.matrix.columns.nlevels > 1:
        labels = pipe.matrix.columns.get_level_values(0).values
        names = pipe.matrix.columns.get_level_values(1).values
        attr += '_features'
    else:
        raise ValueError("No clusters defined. Use cluster or agglomerate to do so.")
    df = pd.Series(data=labels, index=names)
    df.sort_index(inplace=True)
    df = pd.concat([df], axis=1)
    df.columns = ['Cluster']
    if not hasattr(pipe, attr):
        pipe[attr] = df.T
    else:
        pipe[attr] = pd.concat([pipe[attr], df.T])


@rnax.command()
@click.argument("file", type=click.Path())
@pass_pipe
@arggist
def cluster_from(pipe: Pipe, file: str):
    """Use cluster indices from file."""
    index = pd.read_csv(file, sep='\t')
    columns = index.iloc[:, 0]
    values = index.iloc[:, 1]
    tuples = [(f"cluster_{v}", u) for (u, v) in list(zip(columns, values))
              if not np.isnan(v) or u in pipe.matrix.columns]
    pipe.matrix.columns = pd.MultiIndex.from_tuples(tuples)
    pipe.matrix.sort_index(level=1, axis=1, inplace=True)
    pipe.is_clustered = True


@rnax.command()
@click.option("--min-support", "-l", type=click.FLOAT, default=0.5)
@pass_pipe
@arggist
def frequent_sets(pipe: Pipe, min_support: float):
    """Calculate frequent sets using the apriori algorithm. Assumes one-hot encoded matrix."""
    from mlxtend.frequent_patterns import apriori
    df = pipe.matrix.astype(np.bool)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    print(frequent_itemsets)


@rnax.command()
@click.option("--metric", "-m",
              type=click.Choice(['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                                 'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                                 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
                                 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                                 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']),
              default="euclidean")
@pass_pipe
@arggist
def pairwise_distances(pipe: Pipe, metric):
    """Calculate pairwise distances between rows of the data matrix."""
    d = squareform(pdist(pipe.matrix.values, metric=metric))
    pipe.matrix = pd.DataFrame(data=d, columns=pipe.matrix.index, index=pipe.matrix.index)


@rnax.command()
@click.option("--threshold", "-t", type=click.FLOAT, default=-1)
@click.option("--layout", "-l", type=click.Choice(["fruchterman_reingold", "kamada_kawai"]),
              default="fruchterman_reingold")
@click.option("--cliques", "-c", is_flag=True, default=False)
@pass_pipe
@arggist
def dist_graph(pipe: Pipe, threshold: float, layout: str, cliques: bool):
    """Build a threshold graph, presumes pairwise_distances. """
    import networkx as nx
    assert pipe.matrix.index.values.shape == pipe.matrix.columns.values.shape, "call pdist first"
    samples = pipe.matrix.index.values
    weights = pipe.matrix.values
    n_samples = samples.shape[0]
    max_w = np.max(weights)
    min_w = np.min(weights + np.eye(n_samples) * max_w)
    G = nx.Graph()
    weight_values = []
    for (i, sa), (j, sb) in product(enumerate(samples), enumerate(samples)):
        if i != j:
            w = 1 - (weights[i, j] - min_w) / (max_w - min_w)
            G.add_edge(sa, sb, weight=w)
            weight_values.append(w)

    weight_values = np.array(weight_values)
    if threshold == -1:
        threshold = np.median(weight_values) - np.nextafter(0., 1)
    print(np.min(weight_values), np.median(weight_values), np.max(weight_values))
    under_threshold_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < threshold]

    G.remove_edges_from(under_threshold_edges)
    if cliques:
        G = nx.make_max_clique_graph(G)
        # G = nx.empty_graph()
        # cliques = list(nx.find_cliques(G))
        # for i, clique in enumerate(cliques):
        #     G.add_node(i, nodes=list(clique))
        use_weights = False
    else:
        use_weights = True
    layout_fn = getattr(nx, layout + "_layout", "fruchterman_reingold_layout")
    pos = layout_fn(G, weight="weight")
    fig = _mk_networkx_figure(G, pos, use_weights=use_weights)
    oplot(fig)


@rnax.command()
@pass_pipe
@arggist
def uncluster(pipe: Pipe):
    """Remove grouping of samples/features into clusters."""
    if pipe.is_clustered:
        # TODO: respect level again
        pipe.matrix.index = pipe.matrix.index.map(' | '.join)
        pipe.is_clustered = False
