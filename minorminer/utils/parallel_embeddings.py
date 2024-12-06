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

from minorminer.subgraph import find_subgraph
from minorminer.utils.embeddings import (
    shuffle_graph,
    embeddings_to_ndarray,
    visualize_embeddings,
)
from minorminer.utils.feasibility import (
    embedding_feasibility_filter,
    lattice_size_lower_bound,
)


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


def find_sublattice_embeddings(
    S: nx.Graph,
    T: nx.Graph,
    *,
    sublattice_size: int = None,
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
        sublattice_size: The parameter m of the dwave_networkx graph defining
           the lattice offsets searched and tile. See documentation for
           zephyr, pegasus or chimera graph generators and sublattice mappings.
           When tile is provided as an argument this parameter defines only the
           sublattice mappings. :code:`lattice_size_lower_bound()` provides a
           lower bound based on a fast feasibility filter.
        timeout: Timeout per subgraph search in seconds.
            Defaults to 10. Note that `timeout=0` implies unbounded time for the
            default embedder.
        max_num_emb: Maximum number of embeddings to find.
            Defaults to inf (unbounded).
        tile: A subgraph representing a fundamental
            unit (tile) of the target graph `T` used for embedding. If not
            provided, the tile is automatically generated based on the
            `sublattice_size` and the family of `T` (chimera, pegasus, or
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
            or `sublattice_size`, or if the target graph `T` is not of type
            zephyr, pegasus, or chimera.

    Returns:
        list: A list of disjoint embeddings.
    """
    if sublattice_size is None:
        return find_multiple_embeddings(
            S=S,
            T=T,
            timeout=timeout,
            max_num_emb=max_num_emb,
            seed=seed,
            embedder=embedder,
            embedder_kwargs=embedder_kwargs,
        )

    if not skip_filter and tile is None:
        feasibility_bound = lattice_size_lower_bound(
            S=S, T=T, one_to_one=not one_to_iterable
        )
        if feasibility_bound is None or sublattice_size < feasibility_bound:
            warnings.warn("sublattice_size < lower bound: embeddings will be empty.")
            return []
    # A possible feature enhancement might allow for sublattice_size (m) to be
    # replaced by shape: (m,t) [zephyr] or (m,n,t) [Chimera]
    family = T.graph.get("family")
    if family == "chimera":
        sublattice_mappings = dnx.chimera_sublattice_mappings
        t = T.graph["tile"]
        if tile is None:
            tile = dnx.chimera_graph(m=sublattice_size, n=sublattice_size, t=t)
        elif (
            not skip_filter
            and embedding_feasibility_filter(S, tile, not one_to_iterable) is False
        ):
            warnings.warn("tile is infeasible: embeddings will be empty.")
            return []
    elif family == "pegasus":
        sublattice_mappings = dnx.pegasus_sublattice_mappings
        if tile is None:
            tile = dnx.pegasus_graph(m=sublattice_size)
        elif (
            not skip_filter
            and embedding_feasibility_filter(S, tile, not one_to_iterable) is False
        ):
            warnings.warn("tile is infeasible: embeddings will be empty.")
            return []
    elif family == "zephyr":
        sublattice_mappings = dnx.zephyr_sublattice_mappings
        t = T.graph["tile"]
        if tile is None:
            tile = dnx.zephyr_graph(m=sublattice_size, t=t)
        elif (
            not skip_filter
            and embedding_feasibility_filter(S, tile, not one_to_iterable) is False
        ):
            warnings.warn("tile is infeasible: embeddings will be empty.")
            return []
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
        sublattice_size_S = smallest_tile[stopology] + 1
        S = generators[stopology](sublattice_size_S)

        # For each target topology, checks whether embedding the graph S into
        # that topology is feasible
        for ttopology in topologies:
            sublattice_size = lattice_size_lower_bound(
                S, topology=ttopology, one_to_one=True
            )
            if sublattice_size is None:
                print(
                    f"Embedding {stopology}-{sublattice_size_S} in "
                    f"{ttopology} is infeasible."
                )
            else:
                print(
                    f"Embedding {stopology}-{sublattice_size_S} in "
                    f"{ttopology} may be feasible, requires sublattice_size "
                    f">= {sublattice_size}."
                )
            T = generators[ttopology](smallest_tile[ttopology])
            sublattice_size = lattice_size_lower_bound(S, T=T)
            if sublattice_size is None:
                print(
                    f"Embedding {stopology}-{sublattice_size_S} in "
                    f"{ttopology}-{smallest_tile[ttopology]} is infeasible."
                )
            else:
                print(
                    f"Embedding {stopology}-{sublattice_size_S} in "
                    f"{ttopology}-{smallest_tile[ttopology]} may be feasible"
                    f", requires sublattice_size >= {sublattice_size}."
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
            S, T, sublattice_size=min_raster_scale, max_num_emb=float("inf")
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
