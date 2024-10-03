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
import dwave_networkx as dnx
import networkx as nx
import numpy as np
from dwave import embedding


def write_lad_graph(file_handle, graph):
    """
    Writes the given graph to a file in LAD (Leighton Algorithm Description) format.

    :param file_handle: File handle to write the graph data.
    :param graph: NetworkX graph to be written.
    """
    file_handle.write(f'{len(graph)}\n')
    graph = nx.convert_node_labels_to_integers(graph.copy())
    for node in graph:
        line = f'{graph.degree(node)} '
        for neighbor in graph.neighbors(node):
            line += f'{neighbor} '
        file_handle.write(line + '\n')


def get_chimera_subgrid(qubit_graph, rows, cols, grid_size=16):
    """
    Create a subgraph of a Chimera-16 (DW2000Q) graph based on specified rows and columns of unit cells.

    :param qubit_graph: Qubit connectivity graph (NetworkX graph).
    :param rows: Iterable of row indices of unit cells to include.
    :param cols: Iterable of column indices of unit cells to include.
    :param grid_size: Size of the Chimera grid (default is 16).
    :return: Subgraph of qubit_graph induced on the nodes in 'rows' and 'cols'.
    """
    raise Exception("Not implemented")


def get_pegasus_subgrid(qubit_graph, rows, cols, grid_size=16):
    """
    Create a subgraph of a Pegasus-16 (Advantage) graph based on specified rows and columns of unit cells.

    :param qubit_graph: Qubit connectivity graph (NetworkX graph).
    :param rows: Iterable of row indices of unit cells to include.
    :param cols: Iterable of column indices of unit cells to include.
    :param grid_size: Size of the Pegasus grid (default is 16).
    :return: Subgraph of qubit_graph induced on the nodes in 'rows' and 'cols'.
    """
    coordinates = [dnx.pegasus_coordinates(grid_size).linear_to_nice(node) for node in qubit_graph.nodes]
    used_coords = [coord for coord in coordinates if coord[1] in cols and coord[2] in rows]
    nodes = [dnx.pegasus_coordinates(grid_size).nice_to_linear(coord) for coord in used_coords]
    return qubit_graph.subgraph(nodes).copy()


def get_zephyr_subgrid(qubit_graph, rows, cols, grid_size):
    """
    Create a subgraph of a Zephyr (Advantage2) graph based on specified rows and columns of unit cells.

    :param qubit_graph: Qubit connectivity graph (NetworkX graph).
    :param rows: Iterable of row indices of unit cells to include.
    :param cols: Iterable of column indices of unit cells to include.
    :param grid_size: Size of the Zephyr grid.
    :return: Subgraph of qubit_graph induced on the nodes in 'rows' and 'cols'.
    """
    tile_graph = dnx.zephyr_graph(len(rows))
    sublattice_mappings = dnx.zephyr_sublattice_mappings
    mapping_iterator = sublattice_mappings(
        tile_graph, qubit_graph, offset_list=[(rows[0], cols[0])]
    )
    mapping_function = list(mapping_iterator)[0]
    subgraph_nodes = [mapping_function(node) for node in tile_graph]
    return qubit_graph.subgraph(subgraph_nodes).copy()


def search_for_subgraphs_in_subgrid(base_graph, subgraph, timeout=10, max_number_of_embeddings=np.inf, verbose=True, **kwargs):
    """
    Find embeddings of subgraph within base_graph using the Glasgow subgraph finder.

    :param base_graph: NetworkX graph to search within.
    :param subgraph: NetworkX graph representing the subgraph to find.
    :param timeout: Timeout for each subgraph search.
    :param max_number_of_embeddings: Maximum number of embeddings to find.
    :param verbose: If True, prints progress messages.
    :param **kwargs: Additional keyword arguments for the subgraph finder.
    :return: List of embeddings found.
    """
    embeddings = []
    while len(embeddings) < max_number_of_embeddings:
        embedding_result = glasgow.find_subgraph(subgraph, base_graph, timeout=timeout, triggered_restarts=True)
        if len(embedding_result) == 0:
            break
        else:
            base_graph.remove_nodes_from(embedding_result.values())
            embeddings.append(embedding_result)
            if verbose:
                print(f'{len(base_graph)} vertices remain...')

    if verbose:
        print(f'Found {len(embeddings)} embeddings.')
    return embeddings


def raster_embedding_search(
        hardware_graph, subgraph, grid_size=0, raster_breadth=None, delete_used=True,
        verbose=True, topology='pegasus',
        greed_depth=0,
        verify_embeddings=True,
        max_number_of_embeddings=np.inf,
        **kwargs):
    """
    Perform raster scan embedding search on the hardware graph.

    :param hardware_graph: NetworkX graph representing the hardware connectivity.
    :param subgraph: NetworkX graph representing the subgraph to embed.
    :param grid_size: Size of the grid (default is 0).
    :param raster_breadth: Breadth of the raster scan (default is 5 if None).
    :param delete_used: If True, remove used nodes after each embedding is found.
    :param verbose: If True, prints progress messages.
    :param topology: Topology type ('chimera', 'pegasus', or 'zephyr').
    :param greed_depth: Depth of greediness for improving independent embeddings.
    :param verify_embeddings: If True, verify each embedding found.
    :param max_number_of_embeddings: Maximum number of embeddings to find.
    :param **kwargs: Additional keyword arguments.
    :return: Numpy array of embeddings.
    """
    working_graph = hardware_graph.copy()

    if raster_breadth is None:
        raster_breadth = 5
        
    # Do we assert that A has correctly sorted nodes?
    #assert list(_A.nodes) == sorted(list(_A.nodes)), "Input hardware graph must have nodes in sorted order."

    assert list(subgraph.nodes) == list(range(len(subgraph))), "Subgraph must have consecutive nonnegative integer nodes."

    embeddings = []

    if topology == 'chimera':
        sublattice_mappings = dnx.chimera_sublattice_mappings
        tile_graph = dnx.chimera_graph(raster_breadth)
    elif topology == 'pegasus':
        sublattice_mappings = dnx.pegasus_sublattice_mappings
        tile_graph = dnx.pegasus_graph(raster_breadth)
    elif topology == 'zephyr':
        sublattice_mappings = dnx.zephyr_sublattice_mappings
        tile_graph = dnx.zephyr_graph(raster_breadth)
    else:
        raise ValueError("Unsupported topology type.")

    for tile_index, mapping_function in enumerate(sublattice_mappings(tile_graph, working_graph)):
        tile_subgraph = working_graph.subgraph([mapping_function(node) for node in tile_graph]).copy()

        if verbose:
            print(f'tile {tile_index:3d}: offset={mapping_function.offset} starting with {len(tile_subgraph)} vertices')

        tile_embeddings = search_for_subgraphs_in_subgrid(tile_subgraph, subgraph, verbose=verbose,
                                                          max_number_of_embeddings=max_number_of_embeddings,
                                                          **kwargs)
        if delete_used:
            for embedding in tile_embeddings:
                working_graph.remove_nodes_from(embedding.values())

        if verify_embeddings:
            for embedding_result in tile_embeddings:
                issues = list(embedding.diagnose_embedding({p: [embedding_result[p]] for p in sorted(embedding_result.keys())}, subgraph, hardware_graph))
                if issues:
                    print(issues[0])
                    raise Exception("Embedding verification failed.")

        embeddings += tile_embeddings
        if len(embeddings) >= max_number_of_embeddings:
            break

    independent_embeddings = get_independent_embeddings(embeddings, greed_depth=greed_depth)
    embedding_matrix = np.asarray([[embedding[node] for embedding in independent_embeddings] for node in sorted(subgraph.nodes)]).T

    if verify_embeddings:
        for emb in embedding_matrix:
            issues = list(embedding.diagnose_embedding({p: [emb[p]] for p in range(len(emb))}, subgraph, hardware_graph))
            if issues:
                print(issues[0])
                raise Exception("Embedding verification failed.")

    assert len(np.unique(embedding_matrix)) == len(embedding_matrix.ravel())

    return embedding_matrix

def whole_graph_embedding_search(
        hardware_graph, subgraph,
        verbose=True,
        greed_depth=0,
        verify_embeddings=True,
        max_number_of_embeddings=np.inf,
        **kwargs):
    """
    Perform an embedding search over the entire hardware graph.

    :param hardware_graph: NetworkX graph representing the hardware connectivity.
    :param subgraph: NetworkX graph representing the subgraph to embed.
    :param verbose: If True, prints progress messages.
    :param greed_depth: Depth of greediness for improving independent embeddings.
    :param verify_embeddings: If True, verify each embedding found.
    :param max_number_of_embeddings: Maximum number of embeddings to find.
    :param **kwargs: Additional keyword arguments.
    :return: Numpy array of embeddings.
    """
    working_graph = hardware_graph.copy()


    # Do we assert that A has correctly sorted nodes?
    #assert list(_A.nodes) == sorted(list(_A.nodes)), "Input hardware graph must have nodes in sorted order."

    assert list(subgraph.nodes) == list(range(len(subgraph))), "Subgraph must have consecutive nonnegative integer nodes."

    if verbose:
        print(f'Whole graph, starting with {len(working_graph)} vertices')

    embeddings = search_for_subgraphs_in_subgrid(
        working_graph, subgraph, verbose=verbose,
        max_number_of_embeddings=max_number_of_embeddings,
        **kwargs)

    if verify_embeddings:
        for embedding_result in embeddings:
            issues = list(embedding.diagnose_embedding({p: [embedding_result[p]] for p in sorted(embedding_result.keys())}, subgraph, hardware_graph))
            if issues:
                print(issues[0])
                raise Exception("Embedding verification failed.")

    independent_embeddings = get_independent_embeddings(embeddings, greed_depth=greed_depth)
    embedding_matrix = np.asarray([[embedding[node] for embedding in independent_embeddings] for node in sorted(subgraph.nodes)]).T

    if verify_embeddings:
        for emb in embedding_matrix:
            issues = list(embedding.diagnose_embedding({p: [emb[p]] for p in range(len(emb))}, subgraph, hardware_graph))
            if issues:
                print(issues[0])
                raise Exception("Embedding verification failed.")

    assert len(np.unique(embedding_matrix)) == len(embedding_matrix.ravel())

    return embedding_matrix