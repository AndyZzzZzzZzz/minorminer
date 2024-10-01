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
#
from minorminer import subgraph as glasgow
from .independent_embeddings import get_independent_embeddings
from tqdm import tqdm
import dwave_networkx as dnx
import networkx as nx
from dwave.system.samplers import DWaveSampler
import dimod
import time
import numpy as np
import matplotlib.pyplot as plt
from dwave import embedding


def write_lad_graph(_f, _graph):
    _f.write(f'{len(_graph)}\n')
    _graph = nx.convert_node_labels_to_integers(_graph.copy())
    for v in _graph:
        s = f'{_graph.degree(v)} '
        for u in _graph.neighbors(v):
            s += f'{u} '
        print(s)
        _f.write(s + '\n')
        pass


def get_chimera_subgrid(A, rows, cols, gridsize=16):
    """Make a subgraph of a Chimera-16 (DW2000Q) graph on a set of rows and columns of unit cells.

    :param A: Qubit connectivity graph
    :param rows: Iterable of rows of unit cells to include
    :param cols: Iterable of columns of unit cells to include
    :return: The subgraph of A induced on the nodes in "rows" and "cols"
    """
    raise Exception("Not implemented")
    pass


def get_pegasus_subgrid(A, rows, cols, gridsize=16):
    """Make a subgraph of a Pegasus-16 (Advantage) graph on a set of rows and columns of unit cells.

    :param A: Qubit connectivity graph
    :param rows: Iterable of rows of unit cells to include
    :param cols: Iterable of columns of unit cells to include
    :return: The subgraph of A induced on the nodes in "rows" and "cols"
    """

    coords = [dnx.pegasus_coordinates(gridsize).linear_to_nice(v) for v in A.nodes]
    used_coords = [c for c in coords if c[1] in cols and c[2] in rows]

    return A.subgraph([dnx.pegasus_coordinates(gridsize).nice_to_linear(c) for c in used_coords]).copy()


def get_zephyr_subgrid(A, rows, cols, gridsize):
    """Make a subgraph of a Zephyr (Advantage2) graph on a set of rows and columns of unit cells.

    :param A: Qubit connectivity graph
    :param rows: Iterable of rows of unit cells to include
    :param cols: Iterable of columns of unit cells to include
    :return: The subgraph of A induced on the nodes in "rows" and "cols"
    """

    tile = dnx.zephyr_graph(len(rows))
    sublattice_mappings = dnx.zephyr_sublattice_mappings
    f = sublattice_mappings(
        tile, A, offset_list=[(rows[0], cols[0])]
    )
    ff = list(f)[0]
    subgraph = A.subgraph([ff(_) for _ in tile]).copy()

    return subgraph


def search_for_subgraphs_in_subgrid(B, subgraph, timeout=10, max_number_of_embeddings=np.inf, verbose=True, **kwargs):
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
    A = _A.copy()


    # Do we assert that A has correctly sorted nodes?
    #assert list(_A.nodes) == sorted(list(_A.nodes)), "Input hardware graph must have nodes in sorted order."

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
        raise ValueError

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
                X = list(embedding.diagnose_embedding({p: [emb[p]] for p in sorted(emb.keys())}, subgraph, _A))
                if len(X):
                    print(X[0])
                    raise Exception

        embs += sub_embs
        if len(embs) >= max_number_of_embeddings:
            break

    # Get independent set of embeddings
    independent_embs = get_independent_embeddings(embs, greed_depth=greed_depth)

    embmat = np.asarray([[ie[v] for ie in independent_embs] for v in sorted(subgraph.nodes)]).T

    if verify_embeddings:
        for emb in embmat:
            X = list(embedding.diagnose_embedding({p: [emb[p]] for p in range(len(emb))}, subgraph, _A))
            if len(X):
                print(X[0])
                raise Exception

    assert len(np.unique(embmat)) == len(embmat.ravel())

    return embmat



def whole_graph_embedding_search(
        _A, subgraph,
        verbose=True,
        greed_depth=0,
        verify_embeddings=True,
        max_number_of_embeddings=np.inf,
        **kwargs):
    A = _A.copy()


    # Do we assert that A has correctly sorted nodes?
    #assert list(_A.nodes) == sorted(list(_A.nodes)), "Input hardware graph must have nodes in sorted order."

    assert list(subgraph.nodes) == list(range(len(subgraph))), "Subgraph must have consecutive nonnegative integer nodes."


    if verbose:
        print(f'Whole graph, starting with {len(A)} vertices')

    embs = search_for_subgraphs_in_subgrid(
        A, subgraph, verbose=verbose,
        max_number_of_embeddings=max_number_of_embeddings,
        **kwargs)

    if verify_embeddings:
        for emb in embs:
            X = list(embedding.diagnose_embedding({p: [emb[p]] for p in sorted(emb.keys())}, subgraph, _A))
            if len(X):
                print(X[0])
                raise Exception

    # Get independent set of embeddings
    independent_embs = get_independent_embeddings(embs, greed_depth=greed_depth)

    embmat = np.asarray([[ie[v] for ie in independent_embs] for v in sorted(subgraph.nodes)]).T

    if verify_embeddings:
        for emb in embmat:
            X = list(embedding.diagnose_embedding({p: [emb[p]] for p in range(len(emb))}, subgraph, _A))
            if len(X):
                print(X[0])
                raise Exception

    assert len(np.unique(embmat)) == len(embmat.ravel())

    return embmat