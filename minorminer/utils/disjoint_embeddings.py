# Copyright 2024 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Methods provided for the generation of one or more disjoint embeddings. 
These methods sequentially generate disjoint embeddings of a source graph 
onto a target graph or provide supporting functionality.
"""
import warnings

import dwave_networkx as dnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Union
from collections import Counter

from minorminer.subgraph import find_subgraph


def visualize_embeddings(
    G: nx.Graph,
    embeddings: list,
    S: nx.Graph = None,
    seed: Union[int, np.random.RandomState, np.random.Generator] = None,
    one_to_iterable: bool = False,
    **kwargs,
) -> None:
    """Visualizes the embeddings using dwave_networkx's layout functions.

    This function visualizes embeddings of a source graph onto a target graph
    using specialized layouts for structured graphs (chimera, pegasus, or zephyr)
    or general layouts for unstructured graphs. Node and edge colors are used
    to differentiate embeddings.

    Args:
        G: The target graph to be visualized. If the graph
            represents a specialized topology, it must be constructed using
            dwave_networkx (e.g., chimera, pegasus, or zephyr graphs).
        embeddings: A list of embeddings where each embedding is a dictionary
            mapping nodes of the source graph to nodes in the target graph.
        S: The source graph to visualize (optional). If provided, only edges
            corresponding to the source graph embeddings are visualized.
        seed: A seed for the pseudo-random number generator. When provided,
            it randomizes the colormap assignment for embeddings.
        one_to_iterable: Specifies how embeddings are interpreted. Set to `True`
            to allow multiple target nodes to map to a single source node.
            Defaults to `False` for one-to-one embeddings.
        **kwargs: Additional keyword arguments passed to the drawing functions
            (e.g., `node_size`, `font_size`, `width`).

    Draws:
        - Specialized layouts: Uses dwave_networkx's `draw_chimera`,
          `draw_pegasus`, or `draw_zephyr` functions if the graph family is identified.
        - General layouts: Falls back to networkx's `draw_networkx` for
          graphs with unknown topology.
    """
    fig = plt.gcf()
    ax = plt.gca()
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("lightgrey")

    # Create node color mapping
    node_color_dict = {q: float("nan") for q in G.nodes()}

    if seed is not None:
        _embeddings = embeddings.copy()
        prng = np.random.default_rng(seed)
        prng.shuffle(_embeddings)
    else:
        _embeddings = embeddings

    if one_to_iterable:
        node_color_dict.update(
            {
                q: idx
                for idx, emb in enumerate(_embeddings)
                for c in emb.values()
                for q in c
            }
        )
    else:
        node_color_dict.update(
            {q: idx for idx, emb in enumerate(_embeddings) for q in emb.values()}
        )

    # Create edge color mapping
    edge_color_dict = {}
    highlight_edges = []
    if S is not None:
        for idx, emb in enumerate(_embeddings):
            for u, v in S.edges():
                if u in emb and v in emb:
                    if one_to_iterable:
                        targets_u = emb[u]
                        targets_v = emb[v]
                    else:
                        targets_u = [emb[u]]
                        targets_v = [emb[v]]
                    for tu in targets_u:
                        for tv in targets_v:
                            if G.has_edge(tu, tv):
                                edge_color_dict[(tu, tv)] = idx
                                highlight_edges.append((tu, tv))
    else:
        for v1, v2 in G.edges():
            if node_color_dict[v1] == node_color_dict[v2]:
                edge_color_dict[(v1, v2)] = node_color_dict[v1]
            else:
                edge_color_dict[(v1, v2)] = float("nan")

    # Default drawing arguments
    draw_kwargs = {
        "G": G,
        "node_color": [node_color_dict[q] for q in G.nodes()],
        "edge_color": "lightgrey",
        "node_shape": "o",
        "ax": ax,
        "with_labels": False,
        "width": 1,
        "cmap": cmap,
        "edge_cmap": cmap,
        "node_size": 300 / np.sqrt(G.number_of_nodes()),
    }
    draw_kwargs.update(kwargs)

    topology = G.graph.get("family")
    # Draw the combined graph with color mappings
    if topology == "chimera":
        pos = dnx.chimera_layout(G)
        dnx.draw_chimera(**draw_kwargs)
    elif topology == "pegasus":
        pos = dnx.pegasus_layout(G)
        dnx.draw_pegasus(**draw_kwargs)
    elif topology == "zephyr":
        pos = dnx.zephyr_layout(G)
        dnx.draw_zephyr(**draw_kwargs)
    else:
        pos = nx.spring_layout(G)
        nx.draw_networkx(**draw_kwargs)

    # Recolor specific edges on top of the original graph
    if S is None:
        highlight_edges = [e for e in G.edges() if not np.isnan(edge_color_dict[e])]
    highlight_colors = [edge_color_dict[e] for e in highlight_edges]

    nx.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=highlight_edges,
        edge_color=highlight_colors,
        width=1,
        edge_cmap=cmap,
        ax=ax,
    )


def shuffle_graph(
    G: nx.Graph, seed: Union[int, np.random.RandomState, np.random.Generator] = None
) -> nx.Graph:
    """Shuffle the node and edge ordering of a networkx graph.

    For embedding methods that operate as a function of the node or edge
    ordering this can force diversification in the returned embeddings. Note
    that special orderings that encode graph structure or geometry may enhance
    embedder performance (shuffling may lead to longer search times).

    Args:
        G: A networkx graph
        seed: When provided, is used to shuffle the order of nodes and edges in
        the source and target graph. This can allow sampling from otherwise deterministic routines.
    Returns:
        nx.Graph: The same graph with modified node and edge ordering.
    """
    prng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    prng.shuffle(nodes)
    edges = list(G.edges())
    prng.shuffle(edges)
    _G = nx.Graph()
    _G.add_nodes_from(nodes)
    _G.add_edges_from(edges)
    return _G


def find_multiple_embeddings(
    S: nx.Graph,
    T: nx.Graph,
    *,
    timeout: int = 10,
    max_num_emb: int = 1,
    skip_filter: bool = True,
    seed: Union[int, np.random.RandomState, np.random.Generator] = None,
    embedder: callable = None,
    embedder_kwargs: dict = None,
    one_to_iterable: bool = False,
) -> list:
    """Finds multiple disjoint embeddings of a source graph onto a target graph

    Uses a greedy strategy to deterministically find multiple disjoint
    1:1 embeddings of a source graph within a target graph. Randomizing the
    node order in `S` and/or `T` can enable non-deterministic behavior.

    Args:
        S: The source graph to embed.
        T: The target graph in which to embed.
        timeout (int, optional): Timeout per subgraph search in seconds.
            Defaults to 10. Note that `timeout=0` implies unbounded time for the
            default embedder.
        max_num_emb (int, optional): Maximum number of embeddings to find.
            Defaults to infinity (unbounded).
        skip_filter (bool, optional): Specifies whether to skip the subgraph
            lower bound filter. Defaults to `True`, meaning the filter is skipped.
            The filter is specific to subgraph embedders and should always be
            `True` when the embedder is not a subgraph search method.
        seed (Union[int, np.random.RandomState, np.random.Generator], optional):
            A random seed used to shuffle the order of nodes and edges in the
            source and target graphs. Allows non-deterministic sampling.
        embedder (Callable, optional): Specifies the embedding search method,
            a callable taking `S`, `T`, and `timeout` as parameters. Defaults to
            `minorminer.subgraph.find_subgraph`.
        embedder_kwargs (dict, optional): Additional arguments for the embedder
            beyond `S`, `T`, and `timeout`.
        one_to_iterable (bool, optional): Determines how embedding mappings are
            interpreted. Set to `True` to allow multiple target nodes to map to
            a single source node. Defaults to `False` for one-to-one embeddings.

    Returns:
        list: A list of disjoint embeddings. Each embedding follows the format
            dictated by the embedder. By default, each embedding defines a 1:1
            map from the source to the target graph as a dictionary without
            reusing target variables.
    """
    embs = []
    if embedder is None:
        embedder = find_subgraph
        embedder_kwargs = {"triggered_restarts": True}
    elif embedder_kwargs is None:
        embedder_kwargs = {}

    if max_num_emb == 1 and seed is not None:
        _T = T
    else:
        max_num_emb = min(int(T.number_of_nodes() / S.number_of_nodes()), max_num_emb)
        if seed is None:
            _T = T.copy()
        else:
            _T = shuffle_graph(T, seed=seed)
    if seed is None:
        _S = S
    else:
        _S = shuffle_graph(S, seed=seed)

    for _ in range(max_num_emb):
        # A potential feature enhancement would be to allow different embedding
        # heuristics here, including those that are not 1:1

        if skip_filter or embedding_feasibility_filter(_S, _T, not one_to_iterable):
            emb = embedder(_S, _T, timeout=timeout, **embedder_kwargs)
        else:
            emb = []
        if len(emb) == 0:
            break
        elif max_num_emb > 1:
            if one_to_iterable:
                _T.remove_nodes_from(n for c in emb.values() for n in c)
            else:
                _T.remove_nodes_from(emb.values())
        embs.append(emb)
    return embs


def embedding_feasibility_filter(
    S: nx.Graph, T: nx.Graph, one_to_one: bool = False
) -> bool:
    """Feasibility filter for embedding.

    Determines if the source graph `S` can be subgraph-embedded onto the target
    graph `T` based on their degree distributions. Returns `False` if embedding
    is definitely infeasible; otherwise, returns `True`. False positives are
    permissible because deciding the graph isomorphism problem is NP-complete,
    and this filter is designed to be efficient.

    The degree distribution test is a heuristic; stronger filters are possible
    by exploiting additional graph structure.

    Args:
        S: The source graph to embed.
        T: The target graph in which to embed.
        one_to_one: If True, only 1-to-1 (subgraph) embeddings are allowed.
            Defaults to False, permitting minor embeddings.

    Returns:
        bool: `False` if subgraph embedding is definitely infeasible, `True`
            otherwise.
    """
    # Comment, similar bounds are possible allowing for minor embeddings.
    # This could be a possible feature expansion.
    if (
        T.number_of_nodes() < S.number_of_nodes()
        or T.number_of_edges() < S.number_of_edges()
    ):
        return False
    else:
        S_degree = np.sort([S.degree[n] for n in S.nodes()])
        T_degree = np.sort([T.degree[n] for n in T.nodes()])

        if np.any(T_degree[-len(S_degree) :] < S_degree):
            if one_to_one or T_degree[-1] <= 2:
                # Too many high degree nodes in S
                return False
            else:  # Attempt minor embed (enhance T degrees)
                # Minor embedding feasibility reduces to bin packing when
                # considering a best target case knowing only the degree
                # distribution. In general feasibility is NP-complete, a cheap
                # marginal degree distribution filter is used.

                # We can eliminate nodes of equal degree assuming best case:
                ResidualCounts = Counter(T_degree)
                ResidualCounts.subtract(Counter(S_degree))

                # Target nodes of degree x <=2 are only of use for minor
                # embedding source nodes of degree <=x:
                for kS in range(3):
                    if ResidualCounts[kS] < 0:
                        ResidualCounts[kS + 1] += ResidualCounts[kS]
                    ResidualCounts[kS] = 0

                if all(v > 0 for v in ResidualCounts.values()):
                    return True
                nT_auxiliary = sum(ResidualCounts.values())
                if nT_auxiliary < 0:  # extra available to form chains
                    return False

                # In best case all target nodes have degree kTmax, and chains
                # are trees. To cover degree k in S requires n auxiliary target
                # nodes such that kTmax + n(kTmax-2) >= k
                kTmax = np.max([k for k, v in ResidualCounts.items() if v > 0])
                min_auxiliary_necessary = sum(
                    [
                        -v * np.ceil((k - kTmax) / (kTmax - 2))
                        for k, v in ResidualCounts.items()
                        if v < 0
                    ]
                )
                return min_auxiliary_necessary <= nT_auxiliary
        else:
            return True


def graph_rows_upper_bound(T: nx.Graph = None) -> int:
    """Determines a graph rows upper bound for subgraph embedding.

    Args:
        T: The target graph in which to embed. The graph must be of type
            zephyr, pegasus or chimera and constructed by dwave_networkx.
    Returns:
        int: The maximum possible size of a tile
    """
    return max(T.graph.get("rows"), T.graph.get("columns"))


def graph_rows_lower_bound(
    S: nx.Graph,
    T: nx.Graph = None,
    topology: str = None,
    t: int = None,
    one_to_one: bool = False,
) -> float:
    """Returns a lower bound on the graph size required for embedding.

    Using efficiently established graph properties such as the number of nodes,
    number of edges, node-degree distribution, and two-colorability, this
    function establishes a lower bound on the required scale (`graph_rows`) of
    dwave_networkx graphs. There may be no scale at which embedding is feasible,
    in which case None is returned. Either `T` or `topology` must be specified.

    Args:
        S: The source graph to embed.
        T: The target graph in which to embed. The graph must be of type
            'zephyr', 'pegasus', or 'chimera' and constructed by dwave_networkx.
        topology: The topology ('chimera', 'pegasus', or 'zephyr'). This is
            inferred from `T` by default. Any set value must be consistent with
            `T` (if `T` is not None).
        t: The tile parameter, relevant for 'zephyr' and 'chimera' topologies.
            Inferred from `T` by default. Any set value must be consistent with
            `T` (if `T` is not None).
        one_to_one: True if a subgraph embedding is assumed, False for general
            embeddings.

    Raises:
        ValueError: If `T` and `topology` are inconsistent, or if `T` and `t` are inconsistent.
        ValueError: If neither `T` nor `topology` is specified.
        ValueError: If `S` cannot be embedded in `T` or the specified topology.

    Returns:
        float: Minimum `graph_rows` for embedding to be feasible. Returns
            None if embedding for any number of graph rows is infeasible.
    """
    # Comment, similar bounds are possible allowing for minor embeddings,
    # the degree distribution is bounded as chain length increases, whilst
    # the number of nodes is depleted.
    # This could be a possible feature expansion.
    if T is not None:
        if embedding_feasibility_filter(S, T, one_to_one) is False:
            return None
        if topology is None:
            topology = T.graph.get("family")
        elif topology != T.graph.get("family"):
            raise ValueError("Arguments T and topology are inconsistent")
        if t is None:
            t = T.graph["tile"]
        elif topology != T.graph.get("family"):
            raise ValueError("Arguments T and t are inconsistent")
    else:
        if topology is None:
            raise ValueError("T or topology must be specified")
        if t is None:
            t = 4
        max_degrees = {"chimera": 2 + 2 * t, "pegasus": 15, "zephyr": 4 + 4 * t}
        max_source_degree = max(S.degree[n] for n in S.nodes())
        if max_source_degree > max_degrees[topology]:
            return None

    N = S.number_of_nodes()
    if topology == "chimera":
        # Two colorability is necessary, and cheap to check
        if any(
            c > 2
            for c in nx.algorithms.coloring.greedy_coloring.greedy_color(
                S, strategy="connected_sequential_bfs"
            ).values()
        ):
            return None

        def generator(graph_rows):
            return dnx.chimera_graph(m=graph_rows, n=graph_rows, t=t)

        # A lower bound based on number of variables N = m*n*2*t
        graph_rows = np.ceil(np.sqrt(N / 4 / t))
    elif topology == "pegasus":

        def generator(graph_rows):
            return dnx.pegasus_graph(m=graph_rows)

        # A lower bound based on number of variables N = (m*24-8)*(m-1)
        graph_rows = np.ceil(1 / 12 * (8 + np.sqrt(6 * N + 16)))
    elif topology == "zephyr":

        def generator(graph_rows):
            return dnx.zephyr_graph(m=graph_rows, t=t)

        # A lower bound based on number of variables N = (2m+1)*m*4*t
        graph_rows = np.ceil((np.sqrt(2 * N / t + 1) - 1) / 4)
    else:
        raise ValueError(
            "source graphs must be a graph constructed by "
            "dwave_networkx as chimera, pegasus or zephyr type"
        )
    # Evaluate tile feasibility (defect free subgraphs)
    graph_rows = round(graph_rows)
    tile = generator(graph_rows=graph_rows)
    while embedding_feasibility_filter(S, tile, one_to_one) is False:
        graph_rows += 1
        tile = generator(graph_rows=graph_rows)
    return graph_rows


def find_sublattice_embeddings(
    S: nx.Graph,
    T: nx.Graph,
    *,
    graph_rows: int = None,
    timeout: int = 10,
    max_num_emb: int = 1,
    tile: nx.Graph = None,
    skip_filter: bool = True,
    seed: Union[int, np.random.RandomState, np.random.Generator] = None,
    embedder: callable = None,
    embedder_kwargs: dict = None,
    one_to_iterable: bool = False,
) -> list:
    """Searches for embeddings on sublattices of the target graph.

    See https://doi.org/10.3389/fcomp.2023.1238988 for examples of usage.

    Args:
        S: The source graph to embed.
        T: The target graph in which to embed. If
            raster_embedding is not None, the graph must be of type zephyr,
            pegasus, or chimera and constructed by dwave_networkx.
        graph_rows: The number of (cell) rows (m) defining the square
           sublattice of T. If not specified,
           the full graph is searched. Using a smaller breadth can enable
           much faster searches but might also prevent any embeddings from
           being found. :code:`graph_rows_lower_bound()` provides a lower
           bound based on a fast feasibility filter.
        timeout: Timeout per subgraph search in seconds.
            Defaults to 10. Note that `timeout=0` implies unbounded time for the
            default embedder.
        max_num_emb: Maximum number of embeddings to find.
            Defaults to inf (unbounded).
        tile: A subgraph representing a fundamental
            unit (tile) of the target graph `T` used for embedding. If not
            provided, the tile is automatically generated based on the
            `graph_rows` and the family of `T` (chimera, pegasus, or
            zephyr). If `tile==S`, the embedder is bypassed since it is sufficient
            to check for lost edges.
        skip_filter: Specifies whether to skip the subgraph
            lower bound filter. Defaults to True, meaning the filter is skipped.
        seed: If provided, shuffles the ordering of
            mappings, nodes, and edges of source and target graphs. This can
            allow sampling from otherwise deterministic routines.
        embedder: Specifies the embedding search method,
            a callable taking S, T as the first two arguments and timeout as a
            parameter. Defaults to minorminer.subgraph.find_subgraph.
        embedder_kwargs: Dictionary specifying arguments for the embedder
            other than S, T, and timeout.
        one_to_iterable: Specifies whether the embedder returns a dict with
            iterable values. Defaults to False to match find_subgraph.

    Raises:
        ValueError: If the source graph `S` is too large for the specified tile
            or graph rows, or if the target graph `T` is not of type zephyr,
            pegasus, or chimera.

    Returns:
        list: A list of disjoint embeddings.
    """
    if graph_rows is None:
        return find_multiple_embeddings(
            S=S,
            T=T,
            timeout=timeout,
            max_num_emb=max_num_emb,
            seed=seed,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )

    if not skip_filter:
        feasibility_bound = graph_rows_lower_bound(
            S=S, T=T, one_to_one=not one_to_iterable
        )
        if feasibility_bound is None or graph_rows < feasibility_bound:
            warnings.warn("graph_rows < lower bound: embeddings will be empty.")
            return []
    # A possible feature enhancement might allow for graph_rows (m) to be
    # replaced by shape: (m,t) [zephyr] or (m,n,t) [Chimera]
    family = T.graph.get("family")
    if family == "chimera":
        sublattice_mappings = dnx.chimera_sublattice_mappings
        t = T.graph["tile"]
        if tile is None:
            tile = dnx.chimera_graph(m=graph_rows, n=graph_rows, t=t)
        elif (
            not skip_filter
            and embedding_feasibility_filter(S, tile, one_to_one) is False
        ):
            raise ValueError("S is too large for given tile")
    elif family == "pegasus":
        sublattice_mappings = dnx.pegasus_sublattice_mappings
        if tile is None:
            tile = dnx.pegasus_graph(m=graph_rows)
        elif (
            not skip_filter
            and embedding_feasibility_filter(S, tile, one_to_one) is False
        ):
            raise ValueError("S is too large for given tile")
    elif family == "zephyr":
        sublattice_mappings = dnx.zephyr_sublattice_mappings
        t = T.graph["tile"]
        if tile is None:
            tile = dnx.zephyr_graph(m=graph_rows, t=t)
        elif (
            not skip_filter
            and embedding_feasibility_filter(S, tile, one_to_one) is False
        ):
            raise ValueError("S is too large for given tile")
    else:
        raise ValueError(
            "source graphs must a graph constructed by "
            "dwave_networkx as chimera, pegasus or zephyr type"
        )
    tiling = tile == S
    embs = []
    if max_num_emb == 1 and seed is None:
        _T = T
    else:
        _T = T.copy()

    if seed is not None:
        sublattice_iter = list(sublattice_mappings(tile, _T))
        prng = np.random.default_rng(seed)
        prng.shuffle(sublattice_iter)
    else:
        sublattice_iter = sublattice_mappings(tile, _T)

    for i, f in enumerate(sublattice_iter):
        Tr = _T.subgraph([f(n) for n in tile])
        if tiling:
            if Tr.number_of_edges() == S.number_of_edges():
                sub_embs = [{k: v for k, v in zip(S.nodes, Tr.nodes)}]
            else:
                sub_embs = []
        else:
            sub_embs = find_multiple_embeddings(
                S,
                Tr,
                max_num_emb=max_num_emb,
                timeout=timeout,
                skip_filter=skip_filter,
                seed=seed,
                embedder=embedder,
                embedder_kwargs=embedder_kwargs,
            )
        embs += sub_embs
        if len(embs) >= max_num_emb:
            break

        for emb in sub_embs:
            # A potential feature extension would be to generate many
            # overlapping embeddings and solve an independent set problem. This
            # may allow additional parallel embeddings.
            if one_to_iterable:
                _T.remove_nodes_from([v for c in emb.values() for v in c])
            else:
                _T.remove_nodes_from(emb.values())

    return embs


def embeddings_to_ndarray(embs: list, node_order=None):
    """Convert list of embeddings into an ndarray

    Note this assumes the target graph is labeled by integers and the embedding
    is 1 to 1(numeric) in all cases. This is the format returned by
    minorminor.subgraph for the standard presentation of QPU graphs.

    Args:
        embs: A list of embeddings, each list entry in the
            form of a dictionary with integer values.
        node_order: An iterable giving the ordering of
            variables in each row. When not provided variables are ordered to
            match the first embedding :code:`embs[0].keys()`

    Raises:
        ValueError: If `embs` is empty and `node_order` cannot be inferred.

    Returns:
        np.ndarray: An embedding matrix; each row defines an embedding ordered
            by node_order.
    """
    if node_order is None:
        if len(embs) == 0:
            raise ValueError("shape of ndarray cannot be inferred")
        else:
            node_order = embs[0].keys()

    return np.asarray([[ie[v] for ie in embs] for v in node_order]).T


if __name__ == "__main__":
    print(" min m (graph rows) examples ")

    # Define the Graph Topologies, Tiles, and Generators
    visualize = True
    topologies = ["chimera", "pegasus", "zephyr"]
    smallest_tile = {"chimera": 1, "pegasus": 2, "zephyr": 1}
    generators = {
        "chimera": dnx.chimera_graph,
        "pegasus": dnx.pegasus_graph,
        "zephyr": dnx.zephyr_graph,
    }

    # Iterate over Topologies for Raster Embedding Checks
    for stopology in topologies:
        graph_rows_S = smallest_tile[stopology] + 1
        S = generators[stopology](graph_rows_S)

        # For each target topology, checks whether embedding the graph S into
        # that topology is feasible
        for ttopology in topologies:
            graph_rows = graph_rows_lower_bound(S, topology=ttopology, one_to_one=True)
            if graph_rows is None:
                print(
                    f"Embedding {stopology}-{graph_rows_S} in "
                    f"{ttopology} is infeasible."
                )
            else:
                print(
                    f"Embedding {stopology}-{graph_rows_S} in "
                    f"{ttopology} may be feasible, requires graph_rows "
                    f">= {graph_rows}."
                )
            T = generators[ttopology](smallest_tile[ttopology])
            graph_rows = graph_rows_lower_bound(S, T=T)
            if graph_rows is None:
                print(
                    f"Embedding {stopology}-{graph_rows_S} in "
                    f"{ttopology}-{smallest_tile[ttopology]} is infeasible."
                )
            else:
                print(
                    f"Embedding {stopology}-{graph_rows_S} in "
                    f"{ttopology}-{smallest_tile[ttopology]} may be feasible"
                    f", requires graph_rows >= {graph_rows}."
                )
    print()
    print(" raster embedding examples ")
    # A check at minimal scale:
    for topology in topologies:
        min_raster_scale = smallest_tile[topology]
        S = generators[topology](min_raster_scale)
        T = generators[topology](min_raster_scale + 1)  # Allows 4

        print()
        print(topology)
        # Perform Embedding Search and Validation
        embs = find_sublattice_embeddings(
            S, T, graph_rows=min_raster_scale, max_num_emb=float("inf")
        )
        print(f"{len(embs)} Independent embeddings by rastering")
        print(embs)
        assert all(set(emb.keys()) == set(S.nodes()) for emb in embs)
        assert all(set(emb.values()).issubset(set(T.nodes())) for emb in embs)
        value_list = [v for emb in embs for v in emb.values()]
        assert len(set(value_list)) == len(value_list)
        if visualize:
            plt.figure(figsize=(12, 12))
            visualize_embeddings(T, embeddings=embs)
            Saux = nx.Graph()
            Saux.add_nodes_from(S)
            Saux.add_edges_from(list(S.edges)[:10])  # First 10 edges only ..
            plt.figure(figsize=(12, 12))
            visualize_embeddings(T, embeddings=embs, S=Saux)
            plt.show()
        embs = find_sublattice_embeddings(S, T)
        print(f"{len(embs)} Independent embeddings by direct search")
        assert all(set(emb.keys()) == set(S.nodes()) for emb in embs)
        assert all(set(emb.values()).issubset(set(T.nodes())) for emb in embs)
        value_list = [v for emb in embs for v in emb.values()]
        assert len(set(value_list)) == len(value_list)

        print("Defaults (full graph search): Presented as an ndarray")
        print(embeddings_to_ndarray(embs))

    # print("See additional usage examples in test_embeddings")
