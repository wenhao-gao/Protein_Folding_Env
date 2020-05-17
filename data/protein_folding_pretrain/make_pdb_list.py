import random
from tqdm import tqdm
import os, urllib
from collections import defaultdict


def download_cached(url, target_location):
    """ Download with caching """
    target_dir = os.path.dirname(target_location)
    if not os.path.isfile(target_location):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Use MMTF for speed
        response = urllib.request.urlopen(url)
        size = int(float(response.headers['Content-Length']) / 1e3)
        print('Downloading {}, {} KB'.format(target_location, size))
        with open(target_location, 'wb') as f:
            f.write(response.read())
    return target_location


if __name__ == "__main__":
    # CATH hierarchical classification
    cath_base_url = 'http://download.cathdb.info/cath/releases/latest-release/'
    cath_domain_fn = 'cath-domain-list.txt'
    cath_domain_url = cath_base_url + 'cath-classification-data/' + cath_domain_fn
    cath_domain_file = 'cath/cath-domain-list.txt'
    download_cached(cath_domain_url, cath_domain_file)

    # CATH topologies
    cath_nodes = defaultdict(list)
    with open(cath_domain_file, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#')]
        for line in lines:
            entries = line.split()
            cath_id, cath_node = entries[0], '.'.join(entries[1:4])
            chain_name = cath_id[:4] + '.' + cath_id[4]
            cath_nodes[chain_name].append(cath_node)
    cath_nodes = {key: list(set(val)) for key, val in cath_nodes.items()}

    cath_node = list(cath_nodes.values())
    cath_chains = list(cath_nodes.keys())

    cath_nr = []
    chains = {}
    index = 0
    for i in tqdm(range(len(cath_node))):
        if cath_node[i] in cath_nr:
            j = cath_nr.index(cath_node[i])
            temp_list = chains[j]
            temp_list.append(cath_chains[i])
            chains[j] = temp_list
        else:
            cath_nr.append(cath_node[i])
            chains[index] = [cath_chains[i]]
            index += 1

    dataset = []
    for i in range(len(chains)):
        a = chains[i]
        dataset.append(random.sample(a, 1)[0])

    # dataset = dataset[:2]

    with open('pdb_idx_full.txt', 'wt') as f:
        for data in dataset:
            f.write(data + '\n')

