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
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import networkx as nx
from typing import Union

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
