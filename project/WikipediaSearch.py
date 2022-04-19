from collections import defaultdict
import json
import os

from scraping import get_wiki_graph_one_step

import numpy as np
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class WikipediaSearch:

    def __init__(self, method, graph_file=''):
        if graph_file:
            with open(graph_file) as f:
                self.data = json.load(f)

        self.method = method
        self.setup_method()

    def setup_method(self):
        if self.method == 'universal-sentence-encoder':
            self.embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        elif self.method == 'random':
            self.rng = np.random.default_rng()
        else:
            raise ValueError(f'method {self.method} not recognised')

    def predict(self, target, links):
        if self.method == 'universal-sentence-encoder':
            embeddings = self.embed([target] + links).numpy()
            cosines = np.dot(embeddings[0], embeddings[1:].T)
            return [links[i] for i in cosines.argsort()]
        elif self.method == 'random':
            return self.rng.permutation(links).tolist()

    def search(self, start, end):
        to_visit = [start]
        visited = defaultdict(bool)
        previous = defaultdict(bool)

        while len(to_visit) > 0:
            current = to_visit.pop()
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
    g = WikipediaSearch('random')
    print(g.search('New York State Route 373', 'Canada'))
