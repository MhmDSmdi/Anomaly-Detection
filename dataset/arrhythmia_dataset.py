import random

import numpy as np
from gensim.models import Word2Vec
from scipy.io import loadmat

from deepwalk import graph
from deepwalk import walks as serialized_walks
from deepwalk.skipgram import Skipgram


class ArrhythmiaDataSet:

    def adj_matrix_to_list(self, address, node_numbers, output_name):
        adj_matrix = np.loadtxt(address, usecols=range(node_numbers))
        adj_list = []
        for i in range(node_numbers):
            if adj_matrix[i, i] == 1:
                adj_list.append([i])
            else:
                adj_list.append([])
            for j in range(node_numbers):
                if adj_matrix[i, j] == 1 and i != j:
                    adj_list[i].append(j)
        self.create_adj_list_file(adj_list, output_name)
        return adj_list

    def create_adj_list_file(self, adj_list, file_name):
        file = open(file_name, 'w')
        for i in range(len(adj_list)):
            for j in range(len(adj_list[i])):
                file.write(str((adj_list[i])[j]))
                file.write(" ")
            file.write("\n")
        file.close()

    def load_graph(self, input_address, output_name="g1_out.embeddings", number_walks=10, walk_length=40,
                   max_memory_data_size=1000000000, matfile_variable_name="network", format='adjlist', undirected=True,
                   representation_size=16, workers=1, window_size=5, vertex_freq_degree=False, seed=0):
        if format == "adjlist":
            G = graph.load_adjacencylist(input_address, undirected=undirected)
        elif format == "edgelist":
            G = graph.load_edgelist(input_address, undirected=undirected)
        elif format == "mat":
            G = graph.load_matfile(input_address, variable_name=matfile_variable_name, undirected=undirected)
        else:
            raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % format)

        print("Number of nodes: {}".format(len(G.nodes())))

        num_walks = len(G.nodes()) * number_walks

        print("Number of walks: {}".format(num_walks))

        data_size = num_walks * walk_length

        print("Data size (walks*length): {}".format(data_size))

        if data_size < max_memory_data_size:
            print("Walking...")
            walks = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                                path_length=walk_length, alpha=0, rand=random.Random(seed))
            print("Training...")
            model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1,
                             workers=workers)
        else:
            print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(
                data_size,
                max_memory_data_size))
            print("Walking...")

            walks_filebase = output_name + ".walks"
            walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=number_walks,
                                                              path_length=walk_length, alpha=0,
                                                              rand=random.Random(seed),
                                                              num_workers=workers)

            print("Counting vertex frequency...")
            if not vertex_freq_degree:
                vertex_counts = serialized_walks.count_textfiles(walk_files, workers)
            else:
                # use degree distribution for frequency in tree
                vertex_counts = G.degree(nodes=G.iterkeys())

            print("Training...")
            walks_corpus = serialized_walks.WalksCorpus(walk_files)
            model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                             size=representation_size,
                             window=window_size, min_count=0, trim_rule=None, workers=workers)

        model.wv.save_word2vec_format(output_name)

    def prepare_data_set_matrix(self, matrix_address, node_numbers, output_name):
        self.adj_matrix_to_list(matrix_address, node_numbers, "adj_list_{}.txt".format(output_name))
        self.load_graph(input_address="./adj_list_{}.txt".format(output_name),
                        output_name="embedding_{}.txt".format(output_name))
        output_file = open("./output_{}.txt".format(output_name), 'w')
        file = open("embedding_{}.txt".format(output_name), 'r')
        line = file.readline()
        while line:
            line = file.readline()
            line = line.split(" ")
            line.pop(0)
            for x in line:
                output_file.write(" " + x)
        file.close()
        output_file.close()


def load_dataSet(test_size, embedding_file_address="./dataset/output_DataSet.txt", feature_numbers=16):
    data = loadmat("./dataset/arrhythmia.mat")
    y = data['y']
    X = np.loadtxt(embedding_file_address, usecols=range(feature_numbers))
    X_train = X[: test_size, :]
    X_test = X[test_size:, :]
    y_train = y[: test_size]
    y_test = y[test_size:]
    print(X.shape, len(y))
    return (X_train, X_test), (y_train, y_test)


if __name__ == '__main__':
    load_dataSet(120)
