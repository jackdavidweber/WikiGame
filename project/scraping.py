# %%
import json
import argparse
from argparse import ArgumentParser
import time

import bs4
from bs4 import BeautifulSoup     
import requests

PARSER = "lxml"            # to use lxml (the most common), you'll need to install with .../pip install lxml


# %%
def request_wiki(wiki_url):
    try:
        response = requests.get(wiki_url)
        return response
    except:
        time.sleep(0.1)
        return False


        

# %%
def get_wiki_graph_one_step(l, verbose=True):
    """
    Takes a list of wikipedia articles with no repeats. 
    Returns A dictionary where keys are items in original list. 
    Values are lists of references to other wikipedia articles
    """
    d = {}
    # s = set()

    for i in range(len(l)):
        title = l[i]
        wiki_url = "https://en.wikipedia.org/wiki/" + title

        # Request Wikipedia and Parse
        response = request_wiki(wiki_url)
        if response == False:
            d[title] = [None]
        else:
            data_from_url = response.text
            soup = BeautifulSoup(data_from_url,PARSER)

            if verbose:
                print(f"({i}/{len(l)}): {title}")

            # Capture all referenced articles within article
            link_set = set()  # Use a set to ensure no repeats
            for link in soup.find_all('a'):
                s = link.get('href')
                if (s and s[:6] == "/wiki/"):
                    ref = s[6:]

                    # Make sure that title does not include ":" (means it is not normal wikipedia page)
                    if (ref.find(":") == -1):
                        link_set.add(ref)
            d[title] = list(link_set)

    return d

    

# %%
def list_of_lists_to_set(lol):
    s = set()
    for l in lol:
        for item in l:
            s.add(item)

    return s

# %%
def save_dict(d, filename):
    # create json object from dictionary
    json_file = json.dumps(d)

    # open file for writing, "w" 
    f = open(filename,"w")

    # write json object to file
    f.write(json_file)

    # close file
    f.close()

# %%
def get_wiki_graph(final_d = {}, starting_refs=[], num_steps=1, filename="wiki_graph"):
    if starting_refs:
        ref_list = starting_refs
    else:
        ref_list = ["New_York_State_Route_373"]

    for i in range(num_steps):
        step_d = get_wiki_graph_one_step(ref_list)
        final_d.update(step_d)

        # Get all of the references that haven't been added to graph
        step_refs = list_of_lists_to_set(list(step_d.values()))
        existing_refs = set(final_d.keys())
        unseen_refs = step_refs.difference(existing_refs)

        ref_list = list(unseen_refs)
        
        save_dict(final_d, f"{filename}_{i}.json")

    return (final_d, step_refs, ref_list)


# %%
if __name__ == "__main__":
    aparser = ArgumentParser("Wiki Scrape")
    aparser.add_argument("--starting_article", type=str, default="New_York_State_Route_373")
    aparser.add_argument("--filename", type=str, default="wiki_graph_NYSR")
    aparser.add_argument("--num_steps", type=int, default=1)

    args = aparser.parse_args()

    get_wiki_graph(starting_refs=[args.starting_article], num_steps=args.num_steps, filename=args.filename)

