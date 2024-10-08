import itertools
import time
import networkx as nx
from numpy import random
from tqdm.auto import tqdm


def improve_greedy_independent_set(_G, _S, greed=1):
    """
    Attempts to expand an independent set by removing sets of size "greed" and replacing them with a set of size "greed+1".

    Args:
        _G (networkx.Graph): The input graph.
        _S (set): Current independent set.
        greed (int): The size of the subset to remove and attempt to replace.

    Returns:
        set: An improved independent set, at least as large as the original.
    """
    _S = _S.copy()
    improved_flag = True

    while improved_flag:
        improved_flag = False
        new_s = _S.copy()

        for C in itertools.combinations(_S, greed):
            backup_new_s = new_s.copy()
            for c in C:
                new_s.remove(c)

            NC = []
            for c in C:
                NC += (list(_G.neighbors(c)))
            NC = sorted(list(set(NC)))

            for v in new_s:
                for c in NC.copy():
                    if _G.has_edge(v, c):
                        NC.remove(c)

            if len(NC) == 0:
                new_s = backup_new_s.copy()
                continue

            S_NC = []
            for iSNC in range(10):
                temp = nx.maximal_independent_set(_G.subgraph(NC))
                if len(temp)>len(S_NC):
                    S_NC = temp.copy()

            new_s += S_NC

            if len(new_s) > len(backup_new_s):
                assert nx.number_of_edges(nx.subgraph(_G, new_s)) == 0
                print(f"Improving by greed={greed}.  Deleted {greed} and added {len(S_NC)}, total = {len(new_s)}")
                _S = new_s.copy()
                improved_flag = True
                break
            new_s = backup_new_s.copy()

    return _S


def make_embedding_graph(embs):
    """
    Constructs a graph where each node represents an embedding, and edges connect overlapping embeddings.

    Args:
        embs (list): A list of embeddings, each either a dict or a list.

    Returns:
        networkx.Graph: The embedding graph.
    """
    Gemb = nx.Graph()
    Gemb.add_nodes_from(range(len(embs)))

    if len(Gemb) == 0:
        return Gemb

    vertex_sets = []
    if type(embs[0]) is dict:
        for i, emb in enumerate(embs):
            vertex_sets.append(set(emb.values()))
    elif type(embs[0]) is list:
        for i, emb in enumerate(embs):
            vertex_sets.append(set(emb))
    else:
        raise ValueError("Embeddings must be either a list of dicts or a list of lists.")

    for i, j in tqdm(itertools.combinations(range(len(Gemb)), 2), total=int(len(Gemb)*(len(Gemb)-1)/2)):
        if not vertex_sets[i].isdisjoint(vertex_sets[j]):
            Gemb.add_edge(i, j)

    return Gemb

def get_independent_embeddings(embs, greed_depth=1, num_stable_sets=10):
    """
    Generates a large subset of mutually disjoint embeddings from a set of possibly overlapping embeddings.
    Uses a greedy maximal independent set algorithm. Fast and reasonably good.

    Args:
        embs (list): A list of embeddings, each either a dict or a list.
        greed_depth (int): The depth of greediness in improving the independent set.
        num_stable_sets (int): Number of stable sets to generate and evaluate.

    Returns:
        list: A sublist of embeddings that are mutually disjoint.
    """
    start = time.process_time()
    # len(emb) is too many to analyze. Taking 20,000 at random
    if len(embs) > 20000:
        embs = random.choice(embs, 20000)

    print(f'Building graph ({len(embs)} embeddings).  ',end='\n')
    Gemb = make_embedding_graph(embs)
    print(f'Took {time.process_time() - start} seconds')
    start = time.process_time()

    Sbest = None
    max_size = 0

    for _ in tqdm(range(num_stable_sets)):
        if len(Gemb) > 0:
            S = []
            for _ in range(100):
                Stemp = nx.maximal_independent_set(Gemb)
                if len(Stemp) > len(S):
                    S = Stemp.copy()

            for _greed_depth in range(1,greed_depth+1):
                S = improve_greedy_independent_set(Gemb, S, greed=_greed_depth)
        else:
            return []
        
        if len(S) > max_size:
            Sbest = S.copy()
            max_size = len(Sbest)

    print(f'Built 1,000 greedy MIS.  Took {time.process_time() - start} seconds')
    print(f'Found {len(Sbest)} disjoint embeddings.')
    return [embs[x] for x in Sbest]