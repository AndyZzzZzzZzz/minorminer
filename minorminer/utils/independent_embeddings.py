import itertools
import time
import networkx as nx
from numpy import random
from tqdm.auto import tqdm

def greedy_independent_set(graph, independent_set, greed=1):
    """
    Attempts to expand an independent set by removing subsets of size "greed" and replacing them with larger subsets.
    This function iteratively tries to improve the given independent set by exploring the neighborhoods of the removed nodes.

    Parameters:
    - graph (networkx.Graph): The input graph.
    - independent_set (list): A list of nodes forming an independent set in the graph.
    - greed (int): The size of subsets to remove from the independent set during each iteration.

    Returns:
    - list: An independent set that is at least as large as the original.
    """
    independent_set = independent_set.copy()
    improved_flag = True

    while improved_flag:
        improved_flag = False
        new_independent_set = independent_set.copy()

        # Iterate over all combinations of nodes in the independent set of size 'greed'
        for subset_to_remove in itertools.combinations(independent_set, greed):
            backup_independent_set = new_independent_set.copy()
            for node in subset_to_remove:
                 new_independent_set.remove(node)
            neighbors_of_removed = []

            # Collect neighbors of the removed nodes
            for node in subset_to_remove:
                neighbors_of_removed += list(graph.neighbors(node))
            neighbors_of_removed = sorted(list(set(neighbors_of_removed)))

            # Remove nodes from neighbors_of_removed that are adjacent to the current independent set
            for node in new_independent_set:
                for neighbor in neighbors_of_removed.copy():
                    if graph.has_edge(node, neighbor):
                        neighbors_of_removed.remove(neighbor)


            if len(neighbors_of_removed) == 0:
                # No new nodes to add, restore backup and continue
                new_independent_set = backup_independent_set.copy()
                continue

            # Try to find a larger independent set in the subgraph induced by neighbors_of_removed
            new_nodes_to_add = []
            for _ in range(10):
                temp_set = nx.maximal_independent_set(graph.subgraph(neighbors_of_removed))
                if len(temp_set) > len(new_nodes_to_add):
                    new_nodes_to_add = temp_set.copy()

            # Add the new nodes to the independent set
            new_independent_set += new_nodes_to_add

            if len(new_independent_set) > len(backup_independent_set):
                # Ensure the new set is indeed an independent set
                assert nx.number_of_edges(nx.subgraph(graph, new_independent_set)) == 0
                print(f"Improving by greed={greed}. Deleted {greed} and added {len(new_nodes_to_add)}, total = {len(new_independent_set)}")
                independent_set = new_independent_set.copy()
                improved_flag = True
                break  # Restart the process with the improved set

            # Restore the independent set and continue
            new_independent_set = backup_independent_set.copy()

    return independent_set


def make_embedding_graph(embeddings):
    """
    Constructs a graph where each node represents an embedding, and edges connect embeddings that share common elements.
    This graph helps identify embeddings that are mutually disjoint.

    Parameters:
    - embeddings (list): A list of embeddings, where each embedding is either a dict or a list.

    Returns:
    - networkx.Graph: The constructed graph representing conflicts between embeddings.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(embeddings)))

    if len(graph) == 0:
        return graph

    # Create a list of sets representing the embeddings
    vertex_sets = []
    if isinstance(embeddings[0], dict):
        for embedding in embeddings:
            vertex_sets.append(set(embedding.values()))
    elif isinstance(embeddings[0], list):
        for embedding in embeddings:
            vertex_sets.append(set(embedding))
    else:
        raise ValueError("Embeddings must be a list of dicts or a list of lists.")

    # Build the graph by adding edges between embeddings that are not disjoint
    num_embeddings = len(graph)
    total_pairs = int(num_embeddings * (num_embeddings - 1) / 2)
    for i, j in tqdm(itertools.combinations(range(num_embeddings), 2), total=total_pairs):
        if not vertex_sets[i].isdisjoint(vertex_sets[j]):
            graph.add_edge(i, j)

    return graph

def get_independent_embeddings(embeddings, greed_depth=1, num_stable_sets=10):
    """
    Generates a large subset of mutually disjoint embeddings from a set of possibly overlapping embeddings.
    It uses a greedy maximal independent set algorithm and attempts to improve it using the specified greed depth.

    Parameters:
    - embeddings (list): A list of embeddings (dicts or lists).
    - greed_depth (int): The maximum size of subsets to consider when improving the independent set.
    - num_stable_sets (int): The number of times to attempt finding a better independent set.

    Returns:
    - list: A subset of embeddings that are mutually disjoint.
    """
    start_time = time.process_time()

    if len(embeddings) > 20000:
        print(f'We have {len(embeddings)} embeddings, which is too many to analyze. Taking 20,000 at random.')
        embeddings = random.choice(embeddings, 20000)

    print(f'Building graph ({len(embeddings)} embeddings).')
    graph = make_embedding_graph(embeddings)
    print(f'Took {time.process_time() - start_time} seconds')
    start_time = time.process_time()

    best_independent_set = None
    max_size = 0

    for _ in tqdm(range(num_stable_sets)):
        if len(graph) > 0:
            # Get multiple starting points and improve the best one
            independent_set = []
            for _ in range(100):
                temp_set = nx.maximal_independent_set(graph)
                if len(temp_set) > len(independent_set):
                    independent_set = temp_set.copy()

            # Attempt to improve the independent set using the specified greed depth
            for greed in range(1, greed_depth + 1):
                independent_set = greedy_independent_set(graph, independent_set, greed=greed)
        else:
            return []
        
        if len(independent_set) > max_size:
            best_independent_set = independent_set.copy()
            max_size = len(best_independent_set)

    print(f'Built {num_stable_sets * 100} greedy MIS. Took {time.process_time() - start_time} seconds')
    print(f'Found {len(best_independent_set)} disjoint embeddings.')
    return [embeddings[i] for i in best_independent_set]