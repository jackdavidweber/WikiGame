from collections import defaultdict
import json
import os

import numpy as np
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class GreedySemanticSearch:

    def __init__(self, graph_file, method):
        with open(graph_file) as f:
            self.data = json.load(f)

        self.method = method
        self.setup_method()

    def setup_method(self):
        if self.method == 'universal-sentence-encoder':
            self.embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        else:
            raise ValueError(f'method {self.method} not recognised')

    def predict(self, target, links):
        if self.method == 'universal-sentence-encoder':
            embeddings = self.embed([target] + links).numpy()
            cosines = np.dot(embeddings[0], embeddings[1:].T)
            return links[cosines.argmax()]

    def search(self, start, end):
        path = [start]
        visited = defaultdict(bool)
        visited[start] = True
        current = start

        while current != end:
            links = self.data[current]
            links = [l for l in links if not visited[l]]
            current = self.predict(end, links)

            if current in path:
                print('Got stuck in a loop')
                return path + [current]

            path.append(current)
            visited[current] = True
            print(current)

        return path


if __name__ == '__main__':
    g = GreedySemanticSearch('wiki_graph.json', 'universal-sentence-encoder')
    print(g.search('New York State Route 373', 'Numbered highways in New York'))
