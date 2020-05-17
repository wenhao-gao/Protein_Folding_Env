from tqdm import tqdm
import os
import argparse
import pyrosetta
pyrosetta.init()
from pyrosetta.toolbox import pose_from_rcsb


def fetch_pdb(pdb_id, chain, cwd):
    chain_idx = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    if not chain in chain_idx:
        try:
            chain = int(chain)
            chain = chain_idx[chain]
        except:
            return False

    try:
        pose = pose_from_rcsb(pdb_id)
    except:
        return False

    if len(pose.sequence()) > 2000:
        return False
    else:
        poses = pose.split_by_chain()
        c_id = chain_idx.index(chain)
        try:
            if len(poses) > c_id:
                while c_id != len(poses):
                    chain_pose = poses.pop()
                if len(chain_pose.sequence()) > 500:
                    return False
                else:
                    chain_pose.dump_pdb(cwd + "/" + pdb_id + "." + chain + ".pdb")
                    return True
            else:
                return False
        except:
            return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help='The file containing the pdb ids.')
    parser.add_argument("-n", "--n_start", default=0, type=int,
                        help='The starting index of pdb ids.')
    args = parser.parse_args()

    dataset = []
    with open(args.input, 'r') as f:
        line = f.readline()
        line = line.strip()
        while line:
            dataset.append(line)
            line = f.readline()
            line = line.strip()

    dataset = dataset[args.n_start:]

    with open('pdb_idx.txt', 'wt') as f:

        cwd = os.getcwd()
        cache_dir = os.path.join(cwd, "pdb_orig/")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        os.chdir(cache_dir)
        for data in tqdm(dataset):
            pdb, chain = data.split('.')
            if_use = fetch_pdb(pdb, chain, cwd)
            if if_use:
                f.write(data + '\n')
        os.chdir(cwd)
