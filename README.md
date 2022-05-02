# CSCI 181Y Final Project Report

_Jack Weber, Adam Guo_

_[Presentation link](https://docs.google.com/presentation/d/1GxHRYCPbECe-QAT_uPcoYR31yHpl1f1XA6l_k-nwTvw)_

The [Wikipedia Game](https://en.wikipedia.org/wiki/Wikipedia:Wiki_Game) is a game where players start at the same page on the Wikipedia website and try to navigate to a target page using only the hyperlinks on each page. The goal is to reach the target page in the fewest clicks (or least time). For our project, we wanted to investigate methods for computationally playing the Wikipedia Game and compare them to human players.

"Solving" the Wikipedia game is pretty straightforward. We can model the website as a directed graph where pages are represented by nodes, and an edge exists from page A to page B if page A links to page B. Hence, all we need is a graph search algorithm like breadth first search to find the shortest path. [Six Degrees of Wikipedia](https://www.sixdegreesofwikipedia.com/) demonstrates this in action: given any two pages, it computes the shortest paths between them. For this project, we were more interested in creating an algorithm that models the way a human plays the Wikipedia game. A human player doesn't have access to the entire Wikipedia page graph, and if they did it would defeat the purpose of the game, which is to explore and exploit connections between concepts in our collective knowledge.

As players, we choose a link to click on based on how similar it is to the target page. As a proxy for conceptual similarity, we thought it would be interesting to use semantic similarity. Hence, our algorithm goes page-by-page and chooses a link to click on based on its semantic similarity with the target page. We compared our semantic player with some data from human players and a random player (chooses links randomly) as a baseline, and found that the semantic player underperforms a good human player but still manages to reach the target page most of the time.

## Progress

We began the project by figure out how to access Wikipedia pages and get the links on each page. The Wikimedia Foundation releases dumps of Wikipedia on their website, but these dumps are pretty challenging to download and use, so we decided to do it ourselves by using `BeautifulSoup` on the live site. Using the fact that Wikipedia URLs take the form `https://en.wikipedia.org/wiki/{article_name}`, we wrote some code to download the page corresponding to an article name and extract the articles that it links to.


```python
from scraping import get_wiki_graph_one_step

links = get_wiki_graph_one_step(['Star_Wars'])['Star_Wars']
# to get the article names, we need to separate each word with a space instead of an underscore
links = [l.replace('_', ' ') for l in links]
print(f'First 5 links: {links[:5]}')
print(f'Total number of links: {len(links)}')
```

    (0/1): Star_Wars
    First 5 links: ['Cad Bane', 'Cara Dune', 'Disneytoon Studios', 'Atari 2600', 'Copyright']
    Total number of links: 1264


Next, we tried to construct our own Wikipedia graph. The idea is that starting from some page, we can just add all the pages that it links to, then all the pages that those pages link to, and so on. This ended up being more difficult than we expected because we quickly ran into time and memory limitations -- 3 degrees away from the source page was enough to fail. After rethinking our approach, we realised that we don't need to store a local copy of the graph anyway. For our semantic algorithm to work, we only need the links from the current page, which we can fetch on an ad-hoc basis. Hence, we have all the scraping that we need for this to work.

Our next step is to find and use a model that computes the semantic similarities. Something like word2vec doesn't quite work, since many article names are made up of multiple words, many of which are proper nouns and can't be found in an English dictionary anyway. Instead, we decided to use the [Universal Sentence Encoder](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder), which generates sentence-level embeddings. So we are treating each article name as a "sentence" and computing similarities based on that. The `tensorflow_hub` package makes it easy to download and use a pre-trained version of the model.

Given our embeddings, all we need to do is compute a cosine similarity between each link on the current page and the target page, and pick the one that is most similar.


```python
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
target = 'George R. R. Martin'
embeddings = embed([target] + links).numpy()
cosines = np.dot(embeddings[0], embeddings[1:].T)
print(f'{links[cosines.argmax()]} is most similar to {target}')
```

    George Miller (director) is most similar to George R. R. Martin


To turn this into a coherent search algorithm, we implemented a depth first search that sorts the links at each step according to their semantic similarity with the target. This allows the algorithm to backtrack if it reaches a dead end (i.e. it reaches a page that only has links that have already been visited, though that seems unlikely).


```python
from WikipediaSearch import WikipediaSearch

wiki_search = WikipediaSearch('semantic')
print(wiki_search.search('Interlingue', 'Orange (colour)'))
```

    ['Interlingue', 'Solresol', 'Color spectrum', 'Orange (colour)']


To test our algorithm, we set up a competition between a human player (data pulled from [The Wiki Game](https://www.thewikigame.com/)), our semantic player, and a random player. We found that the semantic player is much better than the random player, though it still underperforms a strong human player. Note that the random player never manages to find the target page because we cut off the algorithm after 25 steps to save time.

| Player   | Average path length | Rate of finding target |
| -------- | ------------------- | ---------------------- |
| Human    | 6.10                | 1.00                   |
| Semantic | 8.26                | 0.76                   |
| Random   | N/A                 | 0.00                   |

## Next steps

There are a few ways we can continue this project. First, we can improve upon our semantic player by giving it more information about each page. As a human player, we know more about about an article than just its name. We have some idea of what the thing itself is. Since the semantic player only models the semantic relationships between article titles, it misses conceptual relationships. One way to improve this would be to give the model more information about each page, such as the first paragraph of each article, which would essentiall "teach" the model what each article is actually about.

Second, it would be interesting to replace the graph search framework with a reinforcement learning framework and see if the performance improves with some training. For instance, we train a reinforcement learning model by rewarding it for reaching the target in the fewest number of steps, and punish it for taking a long time or getting lost. It will be interesting to see how it learns longer-distance relationships between articles and how it compares to our greedy semantic search model.

If we were to redo this project, we would probably have spent less time on trying to scrape Wikipedia. We realised that we didn't need a local copy of the whole graph after we spent a bunch of time trying to construct it, which prevented us from exploring some of the avenues of our project. However, some of the code we wrote for graph scraping was still useful to us, so it wasn't all wasted effort. Other than that, we are pretty happy with how our project went.
