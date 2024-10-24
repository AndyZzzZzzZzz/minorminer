import dwave_networkx as dnx
from raster_embedding import (raster_breadth_subgraph_lower_bound,
                                               raster_embedding_search)
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx 

# data_containers, find some good ways to visualize the data!
min_raster_breadth = {}     # Stores minimal raster breadth required for each (L, topology)
time_required_many = {}     # Stores time required to find multiple embeddings
time_required_1 = {}        # Stores time required to find one embedding
num_embeddings_many = {}     # Stores number of embeddings found


timeout = 1 # Can make bigger when you gain confidence, but don't want to blow our time budget on hard cases.
timeout_many = 60  # Maximum total time allowed for multiple embeddings

target_topologies = ['chimera', 'pegasus', 'zephyr']
generator = {'chimera': dnx.chimera_graph, 'pegasus': dnx.pegasus_graph, 'zephyr': dnx.zephyr_graph}
max_processor_scale = {'chimera': 16, 'pegasus': 16, 'zephyr': 12}

# Loops of length 4 to 2048 on exponential scale
Ls = 2**np.arange(4,12)
for L in Ls:
    S = nx.from_edgelist({(i, (i+1)%L) for i in range(L)}) # 1d ring
    for target_topology in target_topologies:
        rb = raster_breadth_subgraph_lower_bound(S, topology=target_topology)
        min_raster_breadth[(L,target_topology)] = rb  # maybe try rb+1 as well?
        for m_target in range(rb, 16): # increase towards full processor scale.
            T = generator[topology](m_target)
            # Time to first valid embeddings
            t0 = perf_counter()
            embs = raster_embedding_search(S, T, raster_breadth=rb,
                                           max_num_emb=1, timeout=timeout)
            if len(embs) < 1:
                time_required_1[(L, target_topology, m_target)] = float('Inf')
                break
            else:
                time_required_1[(L, target_topology, m_target)] = perf_counter()-t0
        for m_target in range(rb, max_processor_scale[topology]): 
            T = generator[topology](m_target)
 
            # Time to many valid embeddings
            t0 = perf_counter()
            embs = raster_embedding_search(S, T, raster_breadth=rb, timeout=timeout)
            if len(embs) < 1:
                time_required_many[(L, target_topology, m_target)] = float('Inf')
                break
            else:
                time_required_many[(L, target_topology, m_target)] = perf_counter()-t0
                if time_required_many[(L, target_topology, m_target)] > timeout_many:
                    break
            num_embeddings_many[(L, target_topology, m_target)] = len(embs)
            
# Do something interesting with the data, maybe plot L versus time to embed 1 to start with:
for target_topology in target_topologies:
    plt.plot(Ls, [time_required_1[(L, target_topology, min_raster_breadth[(L, target_topology)])] for L in Ls],
             label=f'{target_topology}')
plt.yscale('log')
plt.ylabel('Time, seconds')
plt.xlabel('Length of loop, L')
plt.title('Time to one embedding for smallest viable raster (defect free graph)')
plt.legend()
plt.show()