import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import dwave_networkx as dnx
import json
import os
from minorminer.utils.raster_embedding import (raster_embedding_search,
                                               raster_breadth_subgraph_lower_bound,
                                               visualize_embeddings)
from time import perf_counter

graph_dict = {'biclique': [(2, L, L) for L in range(5, 10)],
              'diamond': [(L, L, 8) for L in range(3, 6)],
              '3ddimer': [(3,2,2), (3,2,3), (3,3,3)],
              '2d': [(L, L) for L in range(4, 9)]}
max_processor_scale = {'chimera': 16, 'pegasus': 16, 'zephyr': 12}

folder_path = "experiment_results"
figures_folder = os.path.join(folder_path, "figures")  # Subdirectory for figures
os.makedirs(figures_folder, exist_ok=True)  # Create figures subdirectory if it doesn't exist
file_name = "rb_timing_results.json"
file_path = os.path.join(folder_path, file_name)
experiment_data = {}

def get_source_graph(lattice_name, shape, randomize=False):
    fname = f'instances/{lattice_name}_{shape}_precision256/seed00.npz'
    data = np.load(fname)
    edgelist = zip(data['i'],data['j'])
    nodelist = sorted(set(data['i']) | set(data['j']))
    if randomize:
        np.random.shuffle(nodelist)
    G = nx.Graph()
    G.add_nodes_from(nodelist)
    G.add_edges_from(edgelist)
    return G


T = dnx.pegasus_graph(m=16)

for lattice_name, shapes in tqdm(graph_dict.items()):
    experiment_data[lattice_name] = []
    for shape in shapes:
        S = get_source_graph(lattice_name, shape)
        NS = S.number_of_nodes()

        lb = raster_breadth_subgraph_lower_bound(
            S, topology='pegasus')
        
        for rb in range(lb, 16):
            NT = dnx.pegasus_graph(m=rb)
            ratio = NT.number_of_nodes() / NS
            print(f"lattice: {lattice_name}, shape: {shape}, rb: {rb}, NT/NS: {ratio:4f}")
            t0 = perf_counter()
            # Save data so you don't need to repeat! Especially large timeout or
            # max_num_emb>1
            # timing for raster_breadth equals None and raster breadth not none
            embs = raster_embedding_search(S, T, raster_breadth=rb,
                                        timeout=60)
            if embs:
                time_elapsed = perf_counter()-t0
            else:
                time_elapsed = 'null'
        #experiment_data[lattice_name]["results"].append({
        #    "raster_breadth": rb,
        #    "time_elapsed": time_elapsed
        # })
        

        #fig, ax = plt.subplots(figsize=(16, 16))
        #ax.set_title(f'Embedding Visualization for {lattice_name} {shape}')
        #visualize_embeddings(H=T, embeddings=embs, ax=ax)
        #figure_filename = f"{lattice_name}_{shape}.png"
        #figure_path = os.path.join(figures_folder, figure_filename)
        #fig.savefig(figure_path, format="png", bbox_inches='tight')
        #plt.close(fig)
       
        # S = get_source_graph(lattice_name, shape, randomize=True)  # Is this slower?
        # t0 = perf_counter()
        # Save data so you don't need to repeat! Especially large timeout or
        # max_num_emb>1
        # embs = raster_embedding_search(S, T, raster_breadth=lb+1,
        #                               timeout=60, max_num_emb=1)
        # if False:
        # Visualization:
        # embs = raster_embedding_search(S, T, raster_breadth=lb+1,
        #                              timeout=60)  # Swap T out for T created from properties
        #pdf_filename = f"{lattice_name}_{shape}.pdf"
        #pdf_path = os.path.join(folder_path, pdf_filename)
        
        # Save the plot as PDF
        #plt.savefig(pdf_path, format="pdf")
        
#with open(file_path, 'w') as json_file:
#    json.dump(experiment_data, json_file, indent=4)