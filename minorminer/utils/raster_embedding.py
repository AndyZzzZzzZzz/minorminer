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
import warnings

import dwave_networkx as dnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from minorminer.subgraph import find_subgraph

def visualize_embeddings(H, embeddings=None, **kwargs):
    """Visualizes the embeddings using dwave_networkx's layout functions.

    Args:
        H (networkx.Graph): The input graph to be visualized. If the graph 
            represents a specialized topology, it must be constructed using 
            dwave_networkx (e.g., chimera, pegasus, or zephyr graphs).
        embeddings (list of dict, optional): A list of embeddings where each 
            embedding is a dictionary mapping nodes of the source graph to 
            nodes in the target graph. If not provided, only the graph `H` 
            will be visualized without specific embeddings.
        **kwargs: Additional keyword arguments passed to the drawing functions 
            (e.g., node size, font size).
    Draws:
        - Specialized layouts: Uses dwave_networkx's `draw_chimera`, 
          `draw_pegasus`, or `draw_zephyr` if the graph family is identified.
        - General layouts: Falls back to networkx's `draw_networkx` for 
          graphs with unknown topology.
    """
    fig = plt.gcf() 
    ax = plt.gca()
    cmap = plt.get_cmap("turbo")
    cmap.set_bad("lightgrey")

    # Create node color mapping
    node_color_dict = {q: float("nan") for q in H.nodes()}
    if embeddings is not None:
        node_color_dict.update(
            {q: idx for idx, emb in enumerate(embeddings, 1) for q in emb.values()}
        )

    # Create edge color mapping
    edge_color_dict = {}
    for v1, v2 in H.edges():
        if node_color_dict[v1] == node_color_dict[v2]:
            edge_color_dict[(v1, v2)] = node_color_dict[v1]
        else:
            edge_color_dict[(v1, v2)] = float("nan")

    # Default drawing arguments
    draw_kwargs = {
        'G': H,
        'node_color': [node_color_dict[q] for q in H.nodes()],
        'edge_color': [edge_color_dict[e] for e in H.edges()],
        'node_shape': 'o',
        'ax': ax,
        'with_labels': False,
        'width': 1,
        'cmap': cmap,
        'edge_cmap': cmap,
    }
    draw_kwargs.update(kwargs)

    topology = H.graph.get('family') 
    # Draw the combined graph with color mappings
    if topology == 'chimera':
        dnx.draw_chimera(**draw_kwargs)
    elif topology == 'pegasus':
        dnx.draw_pegasus(**draw_kwargs)
    elif topology == 'zephyr':
        dnx.draw_zephyr(**draw_kwargs)
    else:
        nx.draw_networkx(**draw_kwargs)
      

def find_multiple_embeddings(S, T, timeout=10, max_num_emb=float('inf')):
    """Finds multiple disjoint embeddings of a source graph onto a target graph

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
        
        if subgraph_embedding_feasibility_filter(S, T):
            emb = find_subgraph(S, _T, timeout=timeout, triggered_restarts=True)
        else:
            emb = []
        if len(emb) == 0:
            break
        else:
            _T.remove_nodes_from(emb.values())
            embs.append(emb)
    return embs

def subgraph_embedding_feasibility_filter(S, T):
    """ Feasibility filter for subgraph embedding.

    If S cannot subgraph embed on T based on number of nodes and degree
    distribution return False. Otherwise returns True. Note that
    False positives are possible, the graph isomorphisms problem
    is NP-hard and this is a cheap filter.

    Args:
        S (networkx.Graph): The source graph to embed.
        T (networkx.Graph, optional): The target graph in which to embed.
    Returns:
        bool: False is subgraph embedding is definitely infeasible, True
           otherwise.
    """
    # Comment, similar bounds are possible allowing for minor embeddings.
    # This could be a possible feature expansion.
    if (T.number_of_nodes() < S.number_of_nodes() or
        T.number_of_edges() < S.number_of_edges()):
        return False
    else:
        S_degree = sorted(S.degree[n] for n in S.nodes())
        T_degree = sorted(T.degree[n] for n in T.nodes())
        if any(T_degree[-i-1] < d for i,d in enumerate(S_degree[::-1])):
            return False
        else:
            return True

def raster_breadth_subgraph_lower_bound(S, T=None, topology=None, t=None):
    """Determines a raster breadth lower bound for subgraph embedding.

    Using efficiently established graph properties such as number of nodes,
    number of edges, node-degree distribution and two-colorability establish
    a lower bound on the required raster_breadth for a 1:1 (subgraph)
    embedding.
    If embedding is infeasible by any raster breadth is infeasible based on
    this rudimentary filter then None is returned.
    Either T or topology must be specified.

    Args:
        S (networkx.Graph): The source graph to embed.
        T (networkx.Graph, optional): The target graph in which to embed. The
            graph must be of type zephyr, pegasus or chimera and constructed by
            dwave_networkx.
        topology (str, optional): The topology 'chimera', 'pegasus' or
            'zephyr'. This is inferred from T by default. Any set value must
            be consistent with T (if T is not None).
        t (int, optional): the tile parameter, relevant for zephyr and chimera
            cases. Inferred from T by defaut. Any set value must be consistent
            with T (if T is not None).
    Returns
        float: minimum raster_breadth for embedding to be feasible. Returns
            None if embedding for any raster breadth is infeasible.
    """
    # Comment, similar bounds are possible allowing for minor embeddings,
    # the degree distribution is bounded as chain length increases, whilst
    # the number of nodes is depleted.
    # This could be a possible feature expansion.
    if T is not None:
        if subgraph_embedding_feasibility_filter(S, T) is False:
            return None
        if topology is None:
            topology = T.graph.get('family')
        elif topology != T.graph.get('family'):
            raise ValueError('Arguments T and topology are inconsistent')
        if t is None:
            t = T.graph['tile']
        elif topology != T.graph.get('family'):
            raise ValueError('Arguments T and t are inconsistent')
    else:
        if topology is None:
            raise ValueError('T or topology must be specified')
        if t is None:
            t = 4
        max_degrees = {'chimera': 2 + 2*t,
                       'pegasus': 15,
                       'zephyr': 4 + 4*t}
        max_source_degree = max(S.degree[n] for n in S.nodes())
        if max_source_degree > max_degrees[topology]:
            return None

    N = S.number_of_nodes()
    if  topology == 'chimera':
        # Two colorability is necessary, and cheap to check
        if any(c > 2 for c in
               nx.algorithms.coloring.greedy_coloring.greedy_color(
                   S, strategy='connected_sequential_bfs').values()):
            return None
        def generator(raster_breadth):
            return dnx.chimera_graph(m=raster_breadth, n=raster_breadth, t=t)
        # A lower bound based on number of variables N = m*n*2*t
        raster_breadth = np.ceil(np.sqrt(N/4/t))
    elif topology == 'pegasus':
        def generator(raster_breadth):
            return dnx.pegasus_graph(m=raster_breadth)
        # A lower bound based on number of variables N = (m*24-8)*(m-1)
        raster_breadth = np.ceil(1/12*(8+np.sqrt(6*N + 16)))
    elif topology == 'zephyr':
        def generator(raster_breadth):
            return dnx.zephyr_graph(m=raster_breadth, t=t)
        # A lower bound based on number of variables N = (2m+1)*m*4*t
        raster_breadth = np.ceil((np.sqrt(2*N/t + 1)-1)/4)
    else:
        raise ValueError("source graphs must be a graph constructed by "
                         "dwave_networkx as chimera, pegasus or zephyr type")
    # Evaluate tile feasibility (defect free subgraphs)
    raster_breadth = round(raster_breadth)
    tile = generator(raster_breadth=raster_breadth)
    while subgraph_embedding_feasibility_filter(S, tile) is False:
        raster_breadth += 1
        tile = generator(raster_breadth=raster_breadth)
    return raster_breadth

def raster_embedding_search(S, T, timeout=10, raster_breadth=None,
                            max_num_emb=float('Inf'), tile=None):
    """Searches for multiple embeddings within a rastered target graph.

    Args:
        S (networkx.Graph): The source graph to embed.
        T (networkx.Graph): The target graph in which to embed. If
            raster_embedding is not None the graph must be of type zephyr,
            pegasus or chimera and constructed by dwave_networkx.
        raster_breadth (int, optional): Raster breadth. If not specified
           the full graph is searched. Using a smaller breadth can enable
           much faster search but might also prevent any embeddings being
           found, :code:`raster_breadth_subgraph_lower_bound()`
           provides a lower bound based on a fast feasibility filter.
        timeout (int, optional): Timeout per subgraph search in seconds.
            Defaults to 10.
        max_num_emb (int, optional): Maximum number of embeddings to find.
            Defaults to inf (unbounded).
        tile (networkx.Graph, optional): 
            A subgraph representing a fundamental unit (tile) of the target graph `T` used for embedding. 
            If none provided, the tile is automatically generated based on the `raster_breadth` and the 
            family of `T` (chimera, pegasus, or zephyr). 
    Returns:
        list: A list of disjoint embeddings.
    """
    if raster_breadth is None:
        return find_multiple_embeddings(
            S, T, timeout=timeout, max_num_emb=max_num_emb)
    else:
        feasibility_bound = raster_breadth_subgraph_lower_bound(S, T=T)
        if feasibility_bound is None or raster_breadth < feasibility_bound:
            warnings.warn('raster_breadth < lower bound')
            return []
    # A possible feature enhancement might allow for raster_breadth to be
    # replaced by raster shape.
    family = T.graph.get('family') 
    if family == 'chimera':
        sublattice_mappings = dnx.chimera_sublattice_mappings
        t = T.graph['tile']
        if tile is None:
            tile = dnx.chimera_graph(m=raster_breadth, n=raster_breadth, t=t)
    elif family == 'pegasus':
        sublattice_mappings = dnx.pegasus_sublattice_mappings
        if tile is None:
            tile = dnx.pegasus_graph(m=raster_breadth)
    elif family == 'zephyr':
        sublattice_mappings = dnx.zephyr_sublattice_mappings
        t = T.graph['tile']
        if tile is None:
            tile = dnx.zephyr_graph(m=raster_breadth, t=t)
    else:
        raise ValueError("source graphs must a graph constructed by "
                         "dwave_networkx as chimera, pegasus or zephyr type")

    _T = T.copy()
    embs = []
    for i, f in enumerate(sublattice_mappings(tile, _T)):
        Tr = _T.subgraph([f(n) for n in tile])

        sub_embs = find_multiple_embeddings(
            S, Tr,
            max_num_emb=max_num_emb,
            timeout=timeout)
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
    """Convert list of embeddings into an ndarray

    Note this assumes the target graph is labeled by integers and the embedding
    is 1 to 1(numeric) in all cases. This is the format returned by
    minorminor.subgraph for the standard presentation of QPU graphs.

    Args:
        embs (networkx.Graph): A list of embeddings, each list entry in the
            form of a dictionary with integer values.
        node_order (iterable, optional): An iterable giving the ordering of
            variables in each row. When not provided variables are ordered to
            match the first embedding :code:`embs[0].keys()`
    Returns:
        np.ndarray: An embedding matrix; each row defines an embedding ordered
            by node_order.
    """
    if node_order is None:
        if len(embs) is None:
            raise ValueError('shape of ndarray cannot be inferred')
        else:
            node_order = embs[0].keys()

    return np.asarray([[ie[v] for ie in embs] for v in node_order]).T


if __name__ == "__main__":
    print(' min raster scale examples ')

    # Define the Graph Topologies, Tiles, and Generators
    topologies = ['chimera', 'pegasus', 'zephyr']
    smallest_tile = {'chimera': 1,
                     'pegasus': 2,
                     'zephyr': 1}
    generators = {'chimera': dnx.chimera_graph,
                  'pegasus': dnx.pegasus_graph,
                  'zephyr': dnx.zephyr_graph}
    
    # Iterate over Topologies for Raster Embedding Checks
    for stopology in topologies:
        raster_breadth_S = smallest_tile[stopology] + 1
        S = generators[stopology](raster_breadth_S)

       
        # For each target topology, checks whether embedding the graph S into that topology is feasible
        for ttopology in topologies:
            raster_breadth = raster_breadth_subgraph_lower_bound(
                S, topology=ttopology)
            if raster_breadth is None:
                print(f'Embedding {stopology}-{raster_breadth_S} in '
                      f'{ttopology} is infeasible.')
            else:
                print(f'Embedding {stopology}-{raster_breadth_S} in '
                      f'{ttopology} may be feasible, requires raster_breadth '
                      f'>= {raster_breadth}.')
            T = generators[ttopology](smallest_tile[ttopology])
            raster_breadth = raster_breadth_subgraph_lower_bound(
                S, T=T)
            if raster_breadth is None:
                print(f'Embedding {stopology}-{raster_breadth_S} in '
                      f'{ttopology}-{smallest_tile[ttopology]} is infeasible.')
            else:
                print(f'Embedding {stopology}-{raster_breadth_S} in '
                      f'{ttopology}-{smallest_tile[ttopology]} may be feasible'
                      f', requires raster_breadth >= {raster_breadth}.')
    print()
    print(' raster embedding examples ')
    # A check at minimal scale:
    for topology in topologies:
        min_raster_scale = smallest_tile[topology]
        S = generators[topology](min_raster_scale)
        T = generators[topology](min_raster_scale + 1)  # Allows 4

        print()
        print(topology)

        # Perform Embedding Search and Validation
        embs = raster_embedding_search(S, T, raster_breadth=min_raster_scale)
        print(f'{len(embs)} Independent embeddings by rastering')
        print(embs)
        assert all(set(emb.keys()) == set(S.nodes()) for emb in embs)
        assert all(set(emb.values()).issubset(set(T.nodes())) for emb in embs)
        value_list = [v for emb in embs for v in emb.values()]
        assert len(set(value_list)) == len(value_list)

        visualize_embeddings(T, embeddings=embs)
        embs = raster_embedding_search(S, T)
        print(f'{len(embs)} Independent embeddings by direct search')
        assert all(set(emb.keys()) == set(S.nodes()) for emb in embs)
        assert all(set(emb.values()).issubset(set(T.nodes())) for emb in embs)
        value_list = [v for emb in embs for v in emb.values()]
        assert len(set(value_list)) == len(value_list)

        print('Defaults (full graph search): Presented as an ndarray')
        print(embeddings_to_ndarray(embs))

    print('See additional usage examples in test_raster_embedding')
