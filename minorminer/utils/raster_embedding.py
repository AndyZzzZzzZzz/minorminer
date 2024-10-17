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
import numpy as np
import warnings
from minorminer.subgraph import find_subgraph


def find_multiple_embeddings(S, T, timeout=10, max_num_emb=float('inf')):
    """
    Finds multiple disjoint embeddings of a source graph onto a target graph

    Uses a greedy strategy to deterministically find multiple disjoint
    1:1 embeddings of a source graph within a target graph. Randomizing the
    node order in S and/or T can be used for a non-deterministic pattern.

    Args:
        S (networkx.Graph): The source graph to embed.
        T (networkx.Graph): The target graph in which to embed.
        timeout (int, optional): Timeout per subgraph search in seconds.
            Defaults to 10.
        max_num_emb (int, optional): Maximum number of embeddings to find.
            Defaults to inf (unbounded).
    Returns:
        list: A list of disjoint embeddings. Each embedding defines a 1:1 map
            from the source to the target in the form of a dictionary with no
            reusue of target variables.
    """
    _T = T.copy()
    embs = []
    while True and len(embs) < max_num_emb:
        # A potential feature enhancement would be to allow different embedding
        # heuristics here, including those that are not 1:1
        emb = find_subgraph(S, _T, timeout=timeout, triggered_restarts=True)
        if len(emb) == 0:
            break
        else:
            _T.remove_nodes_from(emb.values())
            embs.append(emb)

    return embs


def raster_embedding_search(
        A, subgraph, raster_breadth=5,
        topology='pegasus',
        max_number_of_embeddings=np.inf):
    """
    Searches for multiple embeddings within a rastered target graph.

    Args:
        S (networkx.Graph): The source graph to embed.
        T (networkx.Graph): The target graph in which to embed. If
            raster_embedding is not None the graph must be of type zephyr,
            pegasus or chimera and constructed by dwave_networkx.
        raster_breadth (int, optional): Raster breadth. If not specified
           the full graph is searched.
        timeout (int, optional): Timeout per subgraph search in seconds.
            Defaults to 10.
        max_num_emb (int, optional): Maximum number of embeddings to find.
            Defaults to inf (unbounded).
    Returns:
        list: A list of disjoint embeddings.
    """
    if raster_breadth is None:
        return find_multiple_embeddings(
            S, T, timeout=timeout, max_num_emb=max_num_emb)

    # A possible feature enhancement might allow for raster_breadth to be
    # replaced by raster shape.
    if T.graph.get('family') == 'chimera':
        sublattice_mappings = dnx.chimera_sublattice_mappings
        t = T.graph['tile']
        tile = dnx.chimera_graph(m=raster_breadth, n=raster_breadth, t=t)
    elif T.graph.get('family') == 'pegasus':
        sublattice_mappings = dnx.pegasus_sublattice_mappings
        tile = dnx.pegasus_graph(m=raster_breadth)
    elif T.graph.get('family') == 'zephyr':
        sublattice_mappings = dnx.zephyr_sublattice_mappings
        t = T.graph['tile']
        tile = dnx.zephyr_graph(m=raster_breadth, t=t)
    else:
        raise ValueError(f"Unsupported topology: {topology}")
    
    for i, f in enumerate(sublattice_mappings(tile, _A)):
        B = _A.subgraph([f(_) for _ in tile]).copy()

        sub_embs = search_for_subgraphs_in_subgrid(B, subgraph,
                                                   max_number_of_embeddings=max_number_of_embeddings)
        
        # Move verification to testing script 
        # for sub_emb in sub_embs:
        #    _A.remove_nodes_from(sub_emb.values())

        # if verify_embeddings:
        #    for emb in sub_embs:
        #        X = list(embedding.diagnose_embedding(
        #            {p: [emb[p]] for p in sorted(emb.keys())}, subgraph, A
        #        ))
        #        if X:
        #            raise Exception("Embedding verification failed.")

        embs += sub_embs
        if len(embs) >= max_num_emb:
            break

        for emb in sub_embs:
            # A potential feature extension would be to generate many
            # overlapping embeddings and solve an independent set problem. This
            # may allow additional parallel embeddings.
            _T.remove_nodes_from(emb.values())

    return embs


def embeddings_to_ndarray(embs, node_order=None):
    """ Convert list of embeddings into an ndarray

    Note this assumes the target graph is labeled by integers and the embedding
    is 1 to 1(numeric) in all cases. This is the format returned by
    minorminor.subgraph for the standard presentation of QPU graphs.

    Args:
        embs (networkx.Graph): A list of embeddings, each list entry in the
            form of a dictionary with integer values.
        node_order (iterable, optional): An iterable giving the ordering of
            variables in each row.
    Returns:
        np.ndarray: An embedding matrix; each row defines an embedding ordered
            by node_order.
    """
    if node_order is None:
        if len(embs) is None:
            raise ValueError('shape of ndarray cannot be inferred')
        else:
            node_order = sorted(embs[0].keys())

    return np.asarray([[ie[v] for ie in embs] for v in node_order]).T


if __name__ == "__main__":
    # A check at minimal scale:
    for topology in ['chimera', 'pegasus', 'zephyr']:
        if topology == 'chimera':
            min_raster_scale = 1
            S = dnx.chimera_graph(min_raster_scale)
            T = dnx.chimera_graph(min_raster_scale + 1)  # Allows 4
            num_anticipated = 4
        elif topology == 'pegasus':
            min_raster_scale = 2
            S = dnx.pegasus_graph(min_raster_scale)
            T = dnx.pegasus_graph(min_raster_scale + 1)  # Allows 2
            num_anticipated = 6
        elif topology == 'zephyr':
            min_raster_scale = 1
            S = dnx.zephyr_graph(min_raster_scale)
            T = dnx.zephyr_graph(min_raster_scale + 1)  # Allows 2
            num_anticipated = 1
        print()
        print(topology)
        embs = raster_embedding_search(S, T, raster_breadth=min_raster_scale)
        print(f'{len(embs)} Independent embeddings by rastering')
        print(embs)
        assert all(set(emb.keys()) == set(S.nodes()) for emb in embs)
        assert all(set(emb.values()).issubset(set(T.nodes())) for emb in embs)
        value_list = [v for emb in embs for v in emb.values()]
        assert len(set(value_list)) == len(value_list)

        embs = raster_embedding_search(S, T)
        print(f'{len(embs)} Independent embeddings by direct search')
        assert all(set(emb.keys()) == set(S.nodes()) for emb in embs)
        assert all(set(emb.values()).issubset(set(T.nodes())) for emb in embs)
        value_list = [v for emb in embs for v in emb.values()]
        assert len(set(value_list)) == len(value_list)

        print('Defaults (full graph search): Presented as an ndarray')
        print(embeddings_to_ndarray(embs))

    print('See additional usage examples in test_raster_embedding')
