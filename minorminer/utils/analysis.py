import dwave_networkx as dnx
from raster_embedding import (raster_breadth_subgraph_lower_bound,
                                               raster_embedding_search)
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx 
import random
import pickle
import os

# data_containers, find some good ways to visualize the data!
min_raster_breadth = {}     # Stores minimal raster breadth required for each (L, topology)
time_required_many = {}     # Stores time required to find multiple embeddings
time_required_1 = {}        # Stores time required to find one embedding
num_embeddings_many = {}    # Stores number of embeddings found
successful_m_target = {}    # Stores minimal m_target for successful embedding

timeout = 1 # Can make bigger when you gain confidence, but don't want to blow our time budget on hard cases.
timeout_many = 60  # Maximum total time allowed for multiple embeddings

target_topologies = ['chimera', 'pegasus', 'zephyr']
generator = {'chimera': dnx.chimera_graph, 'pegasus': dnx.pegasus_graph, 'zephyr': dnx.zephyr_graph}
max_processor_scale = {'chimera': 16, 'pegasus': 16, 'zephyr': 12}

# Data filename for saving/loading
data_filename = 'embedding_data.pkl'

# Loops of length 4 to 2048 on exponential scale
Ls = 2**np.arange(4,12)  # Ls = [16, 32, 64, 128, 256, 512, 1024, 2048]

for L in Ls:
    # Create a 1D ring (cycle graph) of length L
    S = nx.from_edgelist({(i, (i+1)%L) for i in range(L)}) 
    for target_topology in target_topologies:

        # Compute the minimal raster breadth required for embedding S into the target topology
        rb = raster_breadth_subgraph_lower_bound(S, topology=target_topology)
        # Store the minimal raster breadth
        min_raster_breadth[(L,target_topology)] = rb  # maybe try rb+1 as well?

        # Try increasing m_target to find embeddings (up to max_processor_scale)
        for m_target in range(rb, max_processor_scale[target_topology]):
            # Generate target topology graph T
            T = generator[target_topology](m_target)

            # Time to first valid embeddings
            t0 = perf_counter()
            embs = raster_embedding_search(S, T, raster_breadth=rb,
                                           max_num_emb=1, timeout=timeout)
            
            if len(embs) < 1:
                # No embedding found, record infinite time and break
                time_required_1[(L, target_topology, m_target)] = float('Inf')
                break  # No need to try larger m_target if embedding not possible
            else:
                # Embedding found, record time taken
                time_required_1[(L, target_topology, m_target)] = perf_counter()-t0
                # Record successful m_target
                successful_m_target[(L, target_topology)] = m_target
                break # Found an embedding; proceed to next L
        pass
        # Now search for multiple embeddings
        for m_target in range(rb, max_processor_scale[target_topology]): 
            T = generator[target_topology](m_target)  # Generate target topology graph T

 
            # Time to many valid embeddings
            t0 = perf_counter()
            embs = raster_embedding_search(S, T, raster_breadth=rb, timeout=timeout)

            if len(embs) < 1:
                # No embeddings found, record infinite time and break
                time_required_many[(L, target_topology, m_target)] = float('Inf')
                break  # No embeddings found, record infinite time and break
            else:
                # Embeddings found, record time taken and number of embeddings
                time_required_many[(L, target_topology, m_target)] = perf_counter()-t0
                #if time_required_many[(L, target_topology, m_target)] > timeout_many:
                num_embeddings_many[(L, target_topology, m_target)] = len(embs)
                break
            
            
# Do something interesting with the data, maybe plot L versus time to embed 1 to start with:
# Time (for finding one embeddings) vs length of the loop 
for target_topology in target_topologies:
    times = []
    for L in Ls:
        m_target = min_raster_breadth[(L, target_topology)]  # Get minimal raster breadth
        key = (L, target_topology, m_target)
        time = time_required_1.get(key, float('inf'))  # Get time required or inf if not found
        times.append(time)
    plt.plot(Ls, times, label=f'{target_topology}')

plt.yscale('log')  # Use logarithmic scale for y-axis
plt.ylabel('Time (seconds)')
plt.xlabel('Length of loop, L')
plt.title('Time to find one embedding for smallest viable raster (defect-free graph)')
plt.legend()
plt.show()
plt.show()


# minimal m_target that allowed a successful embedding vs length of loop
for target_topology in target_topologies:
    m_targets = []
    for L in Ls:
        m_target = successful_m_target.get((L, target_topology), np.nan) + random.random()/8
        m_targets.append(m_target)
    plt.plot(Ls, m_targets, marker='o', label=f'{target_topology}')

plt.ylabel('Minimal m_target for Successful Embedding')
plt.xlabel('Length of Loop, L')
plt.title('Minimal m_target Required vs. Loop Length L')
plt.legend()
plt.xscale('log')  # Since L varies exponentially
plt.grid(True)
plt.tight_layout()
plt.show()

