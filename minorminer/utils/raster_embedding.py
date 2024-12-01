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
from collections import Counter

from minorminer.subgraph import find_subgraph

def visualize_embeddings(H: nx.Graph, embeddings: list, prng: np.random.Generator=None, one_to_iterable: bool=False, **kwargs) -> None:
    """Visualizes the embeddings using dwave_networkx's layout functions.

    Args:
        H (networkx.Graph): The input graph to be visualized. If the graph
            represents a specialized topology, it must be constructed using
            dwave_networkx (e.g., chimera, pegasus, or zephyr graphs).
        embeddings (list of dict): A list of embeddings where each
            embedding is a dictionary mapping nodes of the source graph to
            nodes in the target graph. If not provided, only the graph H
            will be visualized without specific embeddings.
        prng (np.random.Generator): A pseudo-random number generator with
            an associated shuffle() operation. This is used to randomize
            the colormap assignment.
        one_to_iterable (bool, optional): Determines how embedding mappings are interpreted.
            Set to True to allow multiple target nodes to be associated with a single source node.
            Use this option when embeddings map to multiple nodes per source. Defaults to `False` for 
            one-to-one embeddings where each source node maps to exactly one target node. 
        **kwargs: Additional keyword arguments passed to the drawing functions
            (e.g., node size, font size).
    Draws:
        - Specialized layouts: Uses dwave_networkx's draw_chimera,
          draw_pegasus, or draw_zephyr if the graph family is identified.
        - General layouts: Falls back to networkx's draw_networkx for
          graphs with unknown topology.
    """
    fig = plt.gcf()
    ax = plt.gca()
    cmap = plt.get_cmap("turbo")
    cmap.set_bad("lightgrey")

    # Create node color mapping
    node_color_dict = {q: float("nan") for q in H.nodes()}

    _embeddings = embeddings
    if prng is not None:
        prng.shuffle(_embeddings)
    if one_to_iterable:
        node_color_dict.update(
            {q: idx for idx, emb in enumerate(_embeddings) for c in emb.values() for q in c}
        )
    else:
        node_color_dict.update(
            {q: idx for idx, emb in enumerate(_embeddings) for q in emb.values()}
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
        'edge_color': "lightgrey",
        'node_shape': 'o',
        'ax': ax,
        'with_labels': False,
        'width': 1,
        'cmap': cmap,
        'edge_cmap': cmap,
        'node_size': 300/np.sqrt(H.number_of_nodes())
    }
    draw_kwargs.update(kwargs)

    topology = H.graph.get('family')
    # Draw the combined graph with color mappings
    if topology == 'chimera':
        pos = dnx.chimera_layout(H)
        dnx.draw_chimera(**draw_kwargs)
    elif topology == 'pegasus':
        pos = dnx.pegasus_layout(H)
        dnx.draw_pegasus(**draw_kwargs)
    elif topology == 'zephyr':
        pos = dnx.zephyr_layout(H)
        dnx.draw_zephyr(**draw_kwargs)
    else:
        pos = nx.spring_layout(H)
        nx.draw_networkx(**draw_kwargs)

    # Recolor specific edges on top of the original graph
    highlight_edges = [e for e in H.edges() if not np.isnan(edge_color_dict[e])]
    highlight_colors = [edge_color_dict[e] for e in highlight_edges]

    nx.draw_networkx_edges(
        H,
        pos=pos,
        edgelist=highlight_edges,
        edge_color=highlight_colors,
        width=1,
        edge_cmap=cmap,
        ax=ax)

def shuffle_graph(T, prng=None):
    """Shuffle the node and edge ordering of a networkx graph. """
    if prng is None:
        prng = np.random.default_rng()
    nodes = list(T.nodes())
    prng.shuffle(nodes)
    edges = list(T.edges())
    prng.shuffle(edges)
    _T = nx.Graph()
    _T.add_nodes_from(nodes)
    _T.add_edges_from(edges)
    return _T

def find_multiple_embeddings(S: nx.Graph, T: nx.Graph, *, timeout: int=10, max_num_emb: int=1, skip_filter: bool=True,
                             prng: np.random.Generator=None, embedder: callable=None, embedder_kwargs: dict=None,
                             one_to_iterable: bool=False) -> list:
    """Finds multiple disjoint embeddings of a source graph onto a target graph

    Uses a greedy strategy to deterministically find multiple disjoint
    1:1 embeddings of a source graph within a target graph. Randomizing the
    node order in S and/or T can be used for a non-deterministic pattern.

    Args:
        S (networkx.Graph): The source graph to embed.
        T (networkx.Graph): The target graph in which to embed.
        timeout (int, optional): Timeout per subgraph search in seconds.
            Defaults to 10. Note that timeout=0 implies unbounded time for the
            default ::code::`embedder=find_subgraph method`
        max_num_emb (int, optional): Maximum number of embeddings to find.
            Defaults to inf (unbounded).
        skip_filter (bool, optional): Specifies whether to skip the subgraph
            lower bound filter. Defaults to `True`, meaning the filter is skipped.
            The filter is specific to subgraph embedders, and skip_filter should
            always be `True` when embedder is not a subgraph search method.
        prng (np.random.Generator, optional): When provided, is used to shuffle
            the order of nodes and edges in the source and target graph. This
            can allow sampling from otherwise deterministic routines.
        embedder (Callable, optional): Specifies the embedding search method,
            a callable taking S, T as first two arguments and timeout as a
            parameter. Defaults to minorminer.subgraph.find_subgraph.
        embedder_kwargs (dict, optional): Specifies arguments for embedder
            other than S, T and timeout.
        one_to_iterable (bool, optional): Determines how embedding mappings are interpreted.
            Set to True to allow multiple target nodes to be associated with a single source node.
            Use this option when embeddings map to multiple nodes per source. Defaults to `False` for 
            one-to-one embeddings where each source node maps to exactly one target node. 
    Returns:
        list: A list of disjoint embeddings. Each embedding follows the format
            dictated by embedder. By default each embedding defines a 1:1 map
            from the source to the target in the form of a dictionary with no
            reusue of target variables.
    """
    embs = []
    if embedder is None:
        embedder = find_subgraph
        embedder_kwargs = {'triggered_restarts': True}
    elif embedder_kwargs is None:
        embedder_kwargs = {}

    if max_num_emb == 1 and prng is not None:
        _T = T
    else:
        max_num_emb = min(int(T.number_of_nodes()/S.number_of_nodes()), max_num_emb)
        if prng is None:
            _T = T.copy()
        else:
            _T = shuffle_graph(T, prng)
    if prng is None:
        _S = S
    else:
        _S = shuffle_graph(S, prng)

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

def embedding_feasibility_filter(S: nx.Graph, T: nx.Graph, one_to_one: bool=False) -> bool:
    """ Feasibility filter for embedding.

    If S cannot subgraph embed on T based on the degree distribution of
    the source and target graphs returns False. Otherwise returns True.
    False positives are permissable, deciding the graph isomorphisms problem
    is NP-complete and this is an efficient filter.
    The degree distribution test is one of many possible heuristics, exploiting
    additional graph structure strictly stronger filters are possible

    Args:
        S: The source graph to embed.
        T: The target graph in which to embed.
        one_to_one: Permit only 1 to 1 embeddings (subgraph embeddings),
            are permitted.
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
        S_degree = np.sort([S.degree[n] for n in S.nodes()])
        T_degree = np.sort([T.degree[n] for n in T.nodes()])

        if np.any(T_degree[-len(S_degree):] < S_degree):
            if one_to_one or T_degree[-1] <=2:
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
                        ResidualCounts[kS+1] += ResidualCounts[kS]
                    ResidualCounts[kS] = 0

                if all(v>0 for v in ResidualCounts.values()):
                    return True
                nT_auxiliary = sum(ResidualCounts.values())
                if nT_auxiliary < 0:  # extra available to form chains
                    return False

                # In best case all target nodes have degree kTmax, and chains
                # are trees. To cover degree k in S requires n auxiliary target
                # nodes such that kTmax + n(kTmax-2) >= k
                kTmax = np.max([k for k,v in ResidualCounts.items() if v>0])
                min_auxiliary_necessary = sum(
                    [-v*np.ceil((k-kTmax)/(kTmax-2))
                     for k, v in ResidualCounts.items() if v<0])
                return min_auxiliary_necessary <= nT_auxiliary
        else:
            return True


def raster_breadth_subgraph_upper_bound(T: nx.Graph=None) -> int:
    """Determines a raster breadth upper bound for subgraph embedding.

    Args:
        T (networkx.Graph, optional): The target graph in which to embed. The
            graph must be of type zephyr, pegasus or chimera and constructed by
            dwave_networkx.
    Returns:
        int: The maximum possible size of a tile
    """
    return max(T.graph.get('rows'), T.graph.get('columns'))

def raster_breadth_subgraph_lower_bound(S: nx.Graph, T: nx.Graph=None, topology: str=None, t: int=None,
                                        one_to_one: bool=False) -> float:
    """Returns a lower bound on the graph size required for embedding.

    Using efficiently established graph properties such as number of nodes,
    number of edges, node-degree distribution, and two-colorability establish
    a lower bound on the required scale (m) of dwave_networkx graphs.
    There may be no scale at which embedding is feasible, in which case None is
    returned.
    Either T or topology must be specified.

    Args:
        S: The source graph to embed.
        T: The target graph in which to embed. The
            graph must be of type zephyr, pegasus or chimera and constructed by
            dwave_networkx.
        topology: The topology 'chimera', 'pegasus' or
            'zephyr'. This is inferred from T by default. Any set value must
            be consistent with T (if T is not None).
        t: the tile parameter, relevant for zephyr and chimera
            cases. Inferred from T by default. Any set value must be consistent
            with T (if T is not None).
        one_to_one: True if a subgraph embedding is assumed, False for general
            embeddings.
    Returns
        float: minimum raster_breadth for embedding to be feasible. Returns
            None if embedding for any raster breadth is infeasible.
    """
    # Comment, similar bounds are possible allowing for minor embeddings,
    # the degree distribution is bounded as chain length increases, whilst
    # the number of nodes is depleted.
    # This could be a possible feature expansion.
    if T is not None:
        if embedding_feasibility_filter(S, T, one_to_one) is False:
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
    while embedding_feasibility_filter(S, tile, one_to_one) is False:
        raster_breadth += 1
        tile = generator(raster_breadth=raster_breadth)
    return raster_breadth

def raster_embedding_search(S: nx.Graph, T: nx.Graph, *, raster_breadth: int=None, timeout: int=10,
                            max_num_emb: int=1, tile: nx.Graph=None, skip_filter: bool=True,
                            prng: np.random.Generator=None, embedder: callable=None, embedder_kwargs: dict=None,
                            one_to_iterable: bool=False) -> list:
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
            Defaults to 10. Note that timeout=0 implies unbounded time for the
            default ::code::`embedder=find_subgraph method`
        max_num_emb (int, optional): Maximum number of embeddings to find.
            Defaults to inf (unbounded).
        tile (networkx.Graph, optional): A subgraph representing a fundamental
            unit (tile) of the target graph `T` used for embedding. If none
            provided, the tile is automatically generated based on the `raster_breadth`
            and the family of `T` (chimera, pegasus, or zephyr).
        skip_filter (bool, optional): Specifies whether to skip the subgraph
            lower bound filter. Defaults to True, meaning the filter is skipped.
        prng (np.random.Generator, optional): If provided the ordering of
            mappings, nodes and edges of source and target graphs are all
            shuffled. This can allow sampling from otherwise deterministic
            routines.
        embedder (Callable, optional): Specifies the embedding search method,
            a callable taking S, T as first two arguments and timeout as a
            parameter. Defaults to minorminer.subgraph.find_subgraph.
        embedder_kwargs (dict, optional): Specifies arguments for embedder
            other than S, T and timeout.
        one_to_iterable (bool): If the embedder returns a dict with iterable
            values set to True, otherwise where the values of nodes of the
            target graph set to False. False by default to match find_subgraph.
    Returns:
        list: A list of disjoint embeddings.
    """
    if raster_breadth is None:
        return find_multiple_embeddings(
            S=S, T=T, timeout=timeout, max_num_emb=max_num_emb, prng=prng,
            embedder=embedder, embedder_kwargs=embedder_kwargs)

    if not skip_filter:
        feasibility_bound = raster_breadth_subgraph_lower_bound(
            S=S, T=T, one_to_one=not one_to_iterable)
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

    embs = []
    if max_num_emb == 1 and prng is None:
        _T = T
    else:
        _T = T.copy()

    if prng is not None:
        sublattice_iter = list(sublattice_mappings(tile, _T))
        prng.shuffle(sublattice_iter)
    else:
        sublattice_iter = sublattice_mappings(tile, _T)

    for i, f in enumerate(sublattice_iter):
        Tr = _T.subgraph([f(n) for n in tile])

        sub_embs = find_multiple_embeddings(
            S, Tr,
            max_num_emb=max_num_emb,
            timeout=timeout, skip_filter=skip_filter, prng=prng,
            embedder=embedder, embedder_kwargs=embedder_kwargs)
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
                S, topology=ttopology, one_to_one=True)
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
        embs = raster_embedding_search(S, T, raster_breadth=min_raster_scale, max_num_emb=float('inf'))
        print(f'{len(embs)} Independent embeddings by rastering')
        print(embs)
        assert all(set(emb.keys()) == set(S.nodes()) for emb in embs)
        assert all(set(emb.values()).issubset(set(T.nodes())) for emb in embs)
        value_list = [v for emb in embs for v in emb.values()]
        assert len(set(value_list)) == len(value_list)

        plt.figure(figsize=(12, 12))
        #visualize_embeddings(T, embeddings=embs)
        #plt.show()
        embs = raster_embedding_search(S, T)
        print(f'{len(embs)} Independent embeddings by direct search')
        assert all(set(emb.keys()) == set(S.nodes()) for emb in embs)
        assert all(set(emb.values()).issubset(set(T.nodes())) for emb in embs)
        value_list = [v for emb in embs for v in emb.values()]
        assert len(set(value_list)) == len(value_list)

        print('Defaults (full graph search): Presented as an ndarray')
        print(embeddings_to_ndarray(embs))

    print('See additional usage examples in test_raster_embedding')
