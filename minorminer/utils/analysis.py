import dwave_networkx as dnx
from raster_embedding import (raster_breadth_subgraph_lower_bound,
                                               raster_embedding_search)
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx 
import random
import seaborn as sns
import pandas as pd

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

for L in Ls:
    # Create a 1D ring (cycle graph) of length L
    S = nx.from_edgelist({(i, (i+1)%L) for i in range(L)}) 
    for target_topology in target_topologies:

        # Compute the minimal raster breadth required for embedding S into the target topology
        rb = raster_breadth_subgraph_lower_bound(S, topology=target_topology)
        # Append to min_raster_breadth_list
        min_raster_breadth_list.append({
            'L': L,
            'Topology': target_topology,
            'Raster_Breadth': rb
        })  # maybe try rb+1 as well?

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
                time_required_1_list.append({
                    'L': L,
                    'Topology': target_topology,
                    'm_target': m_target,
                    'Time_1': float('inf')
                })
                break  # No need to try larger m_target if embedding not possible
            else:
                # Embedding found, record time taken
                time_required_1_list.append({
                    'L': L,
                    'Topology': target_topology,
                    'm_target': m_target,
                    'Time_1': perf_counter() - t0
                })
                #successful_m_target_list({
                #    'L': L,
                #    'target_topology': target_topology,
                #    'm_target': m_target
                #})
                break # Found an embedding; proceed to next L
        
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
        

# Convert lists of dictionaries to Pandas DataFrames
df_min_raster_breadth = pd.DataFrame(min_raster_breadth_list)
df_time_required_1 = pd.DataFrame(time_required_1_list)
df_time_required_many = pd.DataFrame(time_required_many_list)
df_num_embeddings_many = pd.DataFrame(num_embeddings_many_list)

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

# 1. Plot L versus Time to Find One Embedding (Same as before, but using DataFrame)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_time_required_1,
    x='L',
    y='Time_1',
    hue='Topology',
    marker='o'
)
plt.yscale('log')  # Logarithmic scale for y-axis
plt.ylabel('Time to Find One Embedding (seconds)')
plt.xlabel('Length of Loop, L')
plt.title('Time to Find One Embedding for Smallest Viable Raster (Defect-Free Graph)')
plt.legend(title='Topology')
plt.tight_layout()
plt.show()


# 2. Seaborn Visualization: Heatmap of Time Required to Find Multiple Embeddings
# Create a separate heatmap for each topology
for topology in target_topologies:
    # Filter data for the current topology
    df_topo = df_time_required_many[df_time_required_many['Topology'] == topology]
    
    if df_topo.empty:
        print(f"No data available for topology: {topology}")
        continue
    
    # Pivot the DataFrame to have m_target as rows and L as columns
    pivot_time_many = df_topo.pivot(index='m_target', columns='L', values='Time_Many')
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_time_many,
        annot=True,
        fmt=".1f",
        cmap='viridis',
        cbar_kws={'label': 'Time (seconds)'},
        linewidths=.5
    )
    plt.title(f'Time to Find Multiple Embeddings for Topology: {topology.capitalize()}')
    plt.xlabel('Length of Loop, L')
    plt.ylabel('m_target')
    plt.tight_layout()
    plt.show()

# 3. Seaborn Visualization: Scatter Plot of Time vs. m_target with L as Hue
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_time_required_many,
    x='m_target',
    y='Time_Many',
    hue='L',
    style='Topology',
    palette='viridis',
    alpha=0.7
)
plt.yscale('log')  # Log scale for better visibility
plt.xlabel('m_target')
plt.ylabel('Time to Find Multiple Embeddings (seconds)')
plt.title('Embedding Time vs. m_target for Various Loop Lengths and Topologies')
plt.legend(title='Loop Length (L)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 4. Seaborn Visualization: FacetGrid of Time vs. L for Each Topology and m_target
g = sns.FacetGrid(
    df_time_required_many,
    col="Topology",
    hue="m_target",
    col_wrap=3,
    height=4,
    aspect=1.5,
    palette='viridis'
)
g.map(sns.lineplot, "L", "Time_Many", marker='o')
g.add_legend(title="m_target")
g.set_titles(col_template="{col_name} Topology")
g.set_axis_labels("Length of Loop, L", "Time to Find Multiple Embeddings (seconds)")
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Time to Find Multiple Embeddings Across Loop Lengths and m_target', fontsize=16)
plt.show()