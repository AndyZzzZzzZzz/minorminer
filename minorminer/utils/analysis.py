import dwave_networkx as dnx
from raster_embedding import (raster_breadth_subgraph_lower_bound,
                                               raster_embedding_search)
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx 
import seaborn as sns
import pandas as pd
import json
import os
from tqdm import tqdm
from line_profiler import LineProfiler

folder_path = "experiment_results"
file_name = f"ring(m_target=None).json"
file_path = os.path.join(folder_path, file_name)

# Load existing data from JSON if available
if os.path.exists(file_path):
    with open(file_path, 'r') as json_file:
        experiment_results = json.load(json_file)
else:
    experiment_results = {}
    os.makedirs(folder_path, exist_ok=True)



# data_containers, find some good ways to visualize the data!
# Initialize lists to store experiment results
min_raster_breadth_list = []       # To store minimal raster breadth for each (L, topology)
time_required_many_list = []       # To store time required to find multiple embeddings
time_required_1_list = []          # To store time required to find one embedding
num_embeddings_many_list = []      # To store number of embeddings found

timeout = 1 # Can make bigger when you gain confidence, but don't want to blow our time budget on hard cases.
timeout_many = 60  # Maximum total time allowed for multiple embeddings

target_topologies = ['chimera', 'pegasus', 'zephyr']
generator = {'chimera': dnx.chimera_graph, 'pegasus': dnx.pegasus_graph, 'zephyr': dnx.zephyr_graph}
max_processor_scale = {'chimera': 16, 'pegasus': 16, 'zephyr': 12}


# Loops of length 4 to 2048 on exponential scale
Ls = 2**np.arange(4,12)  # Ls = [16, 32, 64, 128, 256, 512, 1024, 2048]

for L in tqdm(range(16, 2048, 2)):
    L_str = str(L)
    # Create a 1D ring (cycle graph) of length L
    S = nx.from_edgelist({(i, (i+1)%L) for i in range(L)}) 
    for target_topology in target_topologies:

        # Compute the minimal raster breadth required for embedding S into the target topology
        rb = raster_breadth_subgraph_lower_bound(S, topology=target_topology) + 1
        # Append to min_raster_breadth_list
        min_raster_breadth_list.append({
            'L': L,
            'Topology': target_topology,
            'Raster_Breadth': rb
        })  # maybe try rb+1 as well?

        if L_str not in experiment_results:
            experiment_results[L_str] = {}
        if target_topology not in experiment_results[L_str]:
            experiment_results[L_str][target_topology] = {}

        # Try increasing m_target to find embeddings (up to max_processor_scale)
        for m_target in [None]:
            # Generate target topology graph T
            T = generator[target_topology](m_target)

            # Time to first valid embeddings, rb = None case
            t0 = perf_counter()
            val = raster_embedding_search(S, T, raster_breadth=rb,
                                            max_num_emb=1, timeout=timeout)
        
            if not val:
                # No embedding found, record infinite time and break
                time_required_1_list.append({
                    'L': L,
                    'Topology': target_topology,
                    'm_target': m_target,
                    'Time_1': float('inf')
                })
                experiment_results[L_str][target_topology]['Time_1'] = {
                    "m_target": m_target,
                    "Time_1": 'null'
                }
                break  # No need to try larger m_target if embedding not possible
            else:
                # Embedding found, record time taken
                time_required_1_list.append({
                    'L': L,
                    'Topology': target_topology,
                    'm_target': m_target,
                    'Time_1': perf_counter() - t0
                })
                experiment_results[L_str][target_topology]['Time_1'] = {
                    "m_target": m_target,
                    "Time_1": perf_counter() - t0
                }
                #successful_m_target_list({
                #    'L': L,
                #    'target_topology': target_topology,
                #    'm_target': m_target
                #})
                break # Found an embedding; proceed to next L
        
        """
        # Now search for multiple embeddings
        for m_target in range(rb, max_processor_scale[target_topology]): 
            T = generator[target_topology](m_target)  # Generate target topology graph T


            # Time to many valid embeddings
            t0 = perf_counter()
            embs = raster_embedding_search(S, T, raster_breadth=rb, timeout=timeout)
            
            if len(embs) < 1:
                # No embeddings found, record infinite time and break
                time_required_many_list.append({
                    'L': L,
                    'Topology': target_topology,
                    'm_target': m_target,
                    'Time_Many': float('inf')
                })
                break  # No embeddings found, record infinite time and break
            else:
                # Embeddings found, record time taken and number of embeddings
                time_required_many_list.append({
                    'L': L,
                    'Topology': target_topology,
                    'm_target': m_target,
                    'Time_Many': perf_counter() - t0
                })
                #if time_required_many[(L, target_topology, m_target)] > timeout_many:
                num_embeddings_many_list.append({
                    'L': L,
                    'Topology': target_topology,
                    'm_target': m_target,
                    'Num_Embeddings': len(embs)
                })
                break
        """
with open(file_path, 'w') as json_file:
    json.dump(experiment_results, json_file, indent=4)

print("Experiment results saved to JSON.")
pass
# Convert lists of dictionaries to Pandas DataFrames
df_min_raster_breadth = pd.DataFrame(min_raster_breadth_list)
df_time_required_1 = pd.DataFrame(time_required_1_list)
df_time_required_many = pd.DataFrame(time_required_many_list)
df_num_embeddings_many = pd.DataFrame(num_embeddings_many_list)


upper_threshold = timeout
df_filled = df_time_required_1.copy()
df_filled['Time_1'] = df_time_required_1['Time_1'].fillna(upper_threshold)
df_filled['Is_Threshold'] = df_time_required_1['Time_1'].isna()

# get the even Ls
df_filled_even = df_filled[df_filled['L'] % 2 == 0]
df_filled_even['Sqrt_L'] = np.sqrt(df_filled_even['L'])

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid")
# 1. Plot L versus Time to Find One Embedding (Same as before, but using DataFrame)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_filled_even,
    x='Sqrt_L',
   y='Time_1',
   hue='Topology',
   marker='o'
)

plt.axhline(upper_threshold, color='red', linestyle='--', label=f'Threshold = {upper_threshold}')
plt.yscale('log')  
plt.xscale('log') 
plt.ylabel('Time to Find One Embedding (seconds)')
plt.xlabel('Length of Loop, L')
plt.title('Time to Find One Embedding for Smallest Viable Raster (Defect-Free Graph)')
plt.legend(title='Topology')
plt.tight_layout()
plt.show()