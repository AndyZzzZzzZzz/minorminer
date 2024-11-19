# test_raster_embedding.py

import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product

import dwave_networkx as dnx
from minorminer.utils.raster_embedding import (visualize_embeddings,
                                               shuffle_graph,
                                               find_multiple_embeddings,
                                               raster_embedding_search)

_display = os.environ.get('DISPLAY', '') != ''


class TestRasterEmbedding(unittest.TestCase):

    @unittest.skipUnless(_display, " No display found")
    def testVisualizeEmbeddings(self):
        plt.figure(1)  # Temporary
        embeddings = [{}]
        T = dnx.chimera_graph(2)
        visualize_embeddings(T, embeddings)
        blocks_of = [1, 8]
        one_to_iterable = [True, False]
        for b, o in product(blocks_of, one_to_iterable):
            if o:
                embeddings = [{0: tuple(n + idx*b
                                        for n in range(b))}
                              for idx in range(T.number_of_nodes()//b)]  # Blocks of 8
            else:
                embeddings = [{n: n + idx*b
                               for n in range(b)}
                              for idx in range(T.number_of_nodes()//b)]  # Blocks of 8
                
            plt.figure(f'2 {b} {o}')  # Temporary
            visualize_embeddings(T, embeddings)
            prng = np.random.default_rng()
            plt.figure(f'3 {b} {o}')  # Temporary
            visualize_embeddings(T, embeddings, prng=prng)
        # plt.show()  # Temporary

    def testShuffleGraph(self):
        prng = np.random.default_rng()
        T = dnx.zephyr_graph(1)
        Ts = shuffle_graph(T, prng)
        self.assertEqual(list(T.nodes()), list(T.nodes()))
        self.assertEqual(list(T.edges()), list(T.edges()))
        self.assertNotEqual(list(T.nodes()), list(Ts.nodes()))
        self.assertNotEqual(list(T.edges()), list(Ts.edges()))

    def testFindMultipleEmbeddingsBasic(self):
        square = {((0, 0),(0, 1)),
                  ((0, 1),(1, 1)),
                  ((1, 1),(1, 0)),
                  ((1, 0),(0, 0))}
        L = 2
        squares = {tuple((v1 + i, v2 + j)
                         for v1, v2 in e)
                   for i in range(L)
                   for j in range(L)
                   for e in square}
        S = nx.from_edgelist(square)
        T = nx.from_edgelist(squares) # Room for 1!
        embs = raster_embedding_search(S, T)
        self.assertEqual(len(embs), 1,
                        'mismatched number of embeddings')
        L = 3 # Room for 4 embeddings, might find less (as few as 1)
        squares = {tuple((v1 + i, v2 + j)
                         for v1, v2 in e)
                   for i in range(L)
                   for j in range(L)
                   for e in square}
        T = nx.from_edgelist(squares) # Room for 4!
        embs = raster_embedding_search(S, T,
                                       max_num_emb=float('inf'))
        
        self.assertLess(len(embs), 5,
                        'Impossibly many')
        self.assertTrue(all(set(emb.keys()) == set(S.nodes())
                            for emb in embs), 'bad keys in embedding(s)')
        self.assertTrue(all(set(emb.values()).issubset(set(T.nodes()))
                            for emb in embs), 'bad values in embedding(s)')
        value_list = [v for emb in embs for v in emb.values()]
        self.assertEqual(len(set(value_list)), len(value_list),
                         'embeddings overlap')
        
    def testFindMultipleEmbeddingsAdvanced(self):
        square = {((0, 0),(0, 1)),
                  ((0, 1),(1, 1)),
                  ((1, 1),(1, 0)),
                  ((1, 0),(0, 0))}
        L = 3
        squares = {tuple((v1 + i, v2 + j)
                         for v1, v2 in e)
                   for i in range(L)
                   for j in range(L)
                   for e in square}
        S = nx.from_edgelist(square)
        T = nx.from_edgelist(squares)

        # max_num_embs
        embss = [find_multiple_embeddings(S, T, max_num_emb=mne) for mne in range(1, 4)]
        for mne in range(1,4):
            i = mne - 1
            self.assertLessEqual(len(embss[i]), mne)
            if i > 0:
                self.assertLessEqual(len(embss[i-1]), len(embss[i]))
                # Should proceed deterministically, sequentially:
                self.assertTrue(embs[i-1][idx] == emb[idx] for idx, emb in enumerate(embss[i]))
        
        # timeout
        m = 3  # Feasible, but takes significantly more than a second
        S = dnx.chimera_graph(2*m)
        T = dnx.zephyr_graph(m)
        timeout = 1 # An unfortunate behaviour
        embs = find_multiple_embeddings(S, T, timeout=timeout)
        self.assertEqual(len(embs), 0)

        
    def testMinimalRaster(self):
        for topology in ['chimera', 'pegasus', 'zephyr']:
            if topology == 'chimera':
                min_raster_scale = 1
                S = dnx.chimera_graph(min_raster_scale)
                T = dnx.chimera_graph(min_raster_scale+1)
                num_emb = 4
            elif topology == 'pegasus':
                min_raster_scale = 2
                S = dnx.pegasus_graph(min_raster_scale)
                T = dnx.pegasus_graph(min_raster_scale+1)
                num_emb = 2
            elif topology == 'zephyr':
                min_raster_scale = 1
                S = dnx.zephyr_graph(min_raster_scale)
                T = dnx.zephyr_graph(min_raster_scale+1)
                num_emb = 2
                
            embs = raster_embedding_search(S, T,
                                           raster_breadth=min_raster_scale)
            self.assertEqual(len(embs), 1,
                             'mismatched number of embeddings')
            
            embs = raster_embedding_search(S, T,
                                           raster_breadth=min_raster_scale,
                                           max_num_emb=float('Inf'))
            self.assertEqual(len(embs), num_emb,
                             'mismatched number of embeddings')
            self.assertTrue(all(set(emb.keys()) == set(S.nodes())
                                for emb in embs), 'bad keys in embedding(s)')
            self.assertTrue(all(set(emb.values()).issubset(set(T.nodes()))
                                for emb in embs), 'bad values in embedding(s)')
            value_list = [v for emb in embs for v in emb.values()]
            self.assertEqual(len(set(value_list)), len(value_list),
                             'embeddings overlap')

    
if __name__ == '__main__':
    unittest.main()
