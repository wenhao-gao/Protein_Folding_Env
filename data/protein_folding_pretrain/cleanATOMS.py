"""
clean all pdbs in current folder, delete original file and rename
"""

import pyrosetta
pyrosetta.init()

from pyrosetta.toolbox import cleanATOM

import argparse
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="./", 
            help="The directory we need to clean")
    parser.add_argument("-k", "--keep", default=False, action="store_true", 
            help="If keep the original file and rename")
    args = parser.parse_args()

    pdb_list = os.listdir(args.dir)
    pdb_list = [file for file in pdb_list if file.endswith(".pdb") and not file.startswith(".")]

    for pdb in tqdm(pdb_list):
        pdb_id = pdb[:-4]
        cleanATOM(pdb)
        if args.keep:
            pass
        else:
            os.remove(os.path.join(args.dir, pdb))
            os.rename(os.path.join(args.dir, pdb_id + ".clean.pdb"), os.path.join(args.dir, pdb_id + ".pdb"))


