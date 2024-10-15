# Copyright 2023 D-Wave
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

from minorminer import subgraph as glasgow
from .independent_embeddings import get_independent_embeddings
import dwave_networkx as dnx
import numpy as np
from dwave import embedding


def search_for_subgraphs_in_subgrid(B, subgraph, timeout=10, max_number_of_embeddings=np.inf, verbose=True, **kwargs):
    """
    Searches for subgraphs within a given subgrid.

    Args:
        B (networkx.Graph): The hardware graph to search within.
        subgraph (networkx.Graph): The subgraph to find.
        timeout (int, optional): Timeout for the subgraph search. Defaults to 10.
        max_number_of_embeddings (int, optional): Maximum number of embeddings to find. Defaults to np.inf.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of embeddings found.
    """
    embs = []
    while True and len(embs) < max_number_of_embeddings:
        temp = glasgow.find_subgraph(subgraph, B, timeout=timeout, triggered_restarts=True)
        if len(temp) == 0:
            break
        else:
            B.remove_nodes_from(temp.values())
            embs.append(temp)
            if verbose:
                print(f'{len(B)} vertices remain...')

    if verbose:
        print(f'Found {len(embs)} embeddings.')
    return embs


def raster_embedding_search(
        _A, subgraph, gridsize=0, raster_breadth=5, delete_used=True,
        verbose=True, topology='pegasus',
        greed_depth=0,
        verify_embeddings=True,
        max_number_of_embeddings=np.inf,
        **kwargs):
    """
    Searches for embeddings within a rastered subgraph.

    Args:
        _A (networkx.Graph): The hardware graph.
        subgraph (networkx.Graph): The subgraph to embed.
        gridsize (int, optional): Grid size. Defaults to 0.
        raster_breadth (int, optional): Raster breadth. Defaults to 5.
        delete_used (bool, optional): Whether to delete used nodes after embedding. Defaults to True.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
        topology (str, optional): The topology type ('chimera', 'pegasus', 'zephyr'). Defaults to 'pegasus'.
        greed_depth (int, optional): Depth of greedy improvement. Defaults to 0.
        verify_embeddings (bool, optional): Whether to verify embeddings. Defaults to True.
        max_number_of_embeddings (int, optional): Maximum number of embeddings to find. Defaults to np.inf.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: An embedding matrix.
    """
    A = _A.copy()

    assert list(subgraph.nodes) == list(range(len(subgraph))), "Subgraph must have consecutive nonnegative integer nodes."

    embs = []

    if topology == 'chimera':
        sublattice_mappings = dnx.chimera_sublattice_mappings
        tile = dnx.chimera_graph(raster_breadth)
    elif topology == 'pegasus':
        sublattice_mappings = dnx.pegasus_sublattice_mappings
        tile = dnx.pegasus_graph(raster_breadth)
    elif topology == 'zephyr':
        sublattice_mappings = dnx.zephyr_sublattice_mappings
        tile = dnx.zephyr_graph(raster_breadth)
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    for i, f in enumerate(sublattice_mappings(tile, A)):
        B = A.subgraph([f(_) for _ in tile]).copy()

        if verbose:
            print(f'tile {i:3d}: offset={f.offset} starting with {len(B)} vertices')

        sub_embs = search_for_subgraphs_in_subgrid(B, subgraph, verbose=verbose,
                                                   max_number_of_embeddings=max_number_of_embeddings,
                                                   **kwargs)
        if delete_used:
            for sub_emb in sub_embs:
                A.remove_nodes_from(sub_emb.values())

        if verify_embeddings:
            for emb in sub_embs:
                X = list(embedding.diagnose_embedding(
                    {p: [emb[p]] for p in sorted(emb.keys())}, subgraph, _A
                ))
                if X:
                    raise Exception("Embedding verification failed.")

        embs += sub_embs
        if len(embs) >= max_number_of_embeddings:
            break

    # Get independent set of embeddings
    independent_embs = get_independent_embeddings(embs, greed_depth=greed_depth)

    embmat = np.asarray([[ie[v] for ie in independent_embs] for v in sorted(subgraph.nodes)]).T

    if verify_embeddings:
        for emb in embmat:
            X = list(embedding.diagnose_embedding({p: [emb[p]] for p in range(len(emb))}, subgraph, _A))
            if X:
                raise Exception("Embedding verification failed.")

    assert len(np.unique(embmat)) == len(embmat.ravel()), "Embeddings are not unique."

    return embmat



def whole_graph_embedding_search(
        _A, subgraph,
        verbose=True,
        greed_depth=0,
        verify_embeddings=True,
        max_number_of_embeddings=np.inf,
        **kwargs):
    """
    Searches for embeddings across the whole graph.

    Args:
        _A (networkx.Graph): The hardware graph.
        subgraph (networkx.Graph): The subgraph to embed.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
        greed_depth (int, optional): Depth of greedy improvement. Defaults to 0.
        verify_embeddings (bool, optional): Whether to verify embeddings. Defaults to True.
        max_number_of_embeddings (int, optional): Maximum number of embeddings to find. Defaults to np.inf.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: An embedding matrix.
    """
    A = _A.copy()

    assert list(subgraph.nodes) == list(range(len(subgraph))), "Subgraph must have consecutive nonnegative integer nodes."

    if verbose:
        print(f'Whole graph, starting with {len(A)} vertices')

    embs = search_for_subgraphs_in_subgrid(
        A, subgraph, verbose=verbose,
        max_number_of_embeddings=max_number_of_embeddings,
        **kwargs)

    if verify_embeddings:
        for emb in embs:
            X = list(embedding.diagnose_embedding(
                {p: [emb[p]] for p in sorted(emb.keys())}, subgraph, _A
            ))
            if X:
               raise Exception("Embedding verification failed.")

    # Get independent set of embeddings
    independent_embs = get_independent_embeddings(embs, greed_depth=greed_depth)

    embmat = np.asarray([[ie[v] for ie in independent_embs] for v in sorted(subgraph.nodes)]).T

    if verify_embeddings:
        for emb in embmat:
            X = list(embedding.diagnose_embedding(
                {p: [emb[p]] for p in range(len(emb))}, subgraph, _A
            ))
            if X:
                raise Exception("Embedding verification failed.")

    assert len(np.unique(embmat)) == len(embmat.ravel()), "Embeddings are not unique."

    return embmat