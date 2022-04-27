import argparse
from collections import defaultdict
import json
import os

from scraping import get_wiki_graph_one_step

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)


class WikipediaSearch:

    def __init__(self, method, graph_file='', verbose=False, seed=42):
        if graph_file:
            with open(graph_file) as f:
                self.data = json.load(f)

        self.seed=seed

        self.method = method
        self.setup_method()

        self.verbose=verbose

    def setup_method(self):
        if self.method == 'semantic':
            self.embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        elif self.method == 'random':
            self.rng = np.random.default_rng(seed=self.seed)
        else:
            raise ValueError(f'method {self.method} not recognised')

    def predict(self, target, links):
        if self.method == 'semantic':
            embeddings = self.embed([target] + links).numpy()
            cosines = np.dot(embeddings[0], embeddings[1:].T)
            return [links[i] for i in cosines.argsort()]
        elif self.method == 'random':
            return self.rng.permutation(links).tolist()

    def search(self, start, end, limit=100):
        to_visit = [start]
        visited = defaultdict(bool)
        previous = defaultdict(bool)
        steps = 0

        while len(to_visit) > 0 and steps < limit:
            current = to_visit.pop()
            
            if self.verbose:
                print(current)

            visited[current] = True
            if current == end:
                break

            # Get links on current page, replace all underscores with spaces, get rid of already-
            # visited pages, and sort by self.predict
            links = get_wiki_graph_one_step([current], verbose=False)[current]
            links = [l.replace('_', ' ') for l in links]
            links = [l for l in links if not visited[l]]
            links = self.predict(end, links)

            # Add all the links in order to the DFS stack
            for l in links:
                to_visit.append(l)
                previous[l] = current

            steps += 1

        # Reconstruct the discovered path using previous
        if visited[end]:
            path = [end]
            current = end
            while previous[current]:
                current = previous[current]
                path.append(current)
            return list(reversed(path))
        else:
            return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start', type=str)
    parser.add_argument('end', type=str)
    parser.add_argument('method', type=str)
    args = parser.parse_args()

    g1 = WikipediaSearch(args.method)
    print()
    print(f'{args.method}:')
    print(g1.search(args.start, args.end))
