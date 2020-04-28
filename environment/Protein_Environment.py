"""
Code define the basic protein modeling environment
"""

import pyrosetta
from pyrosetta import *
pyrosetta.init()

from pyrosetta.rosetta.protocols.moves import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.protocols.simple_moves import *
from pyrosetta.rosetta.protocols.minimization_packing import *

from torch_geometric.data import Data
from utilities.Protein import *
from gym import spaces


class Protein_Modeling_Environment(object):
    """
    Define the basic Markov decision process of protein modeling
    """

    def __init__(self):
        self.set_up_score_fxn()
        self.observation_space = spaces.Box(low=-999.0, high=999.0, shape=(9, 9), dtype=np.float32)
        self.action_space = spaces.Box(low=-999.0, high=999.0, shape=(9, 9), dtype=np.float32)

    def step(self, action):
        """Takes a step in the modeling problem. This method must be overwritten by any environment"""
        raise ValueError("This function needs to be implemented by the environment")

    def reset(self):
        """Reset the whole environment. This method must be overwritten by any environment"""
        raise ValueError("This function needs to be implemented by the environment")

    def reward(self):
        """Return the reward of the task. This method must be overwritten by any environment"""
        raise ValueError("This function needs to be implemented by the environment")

    def set_up_score_fxn(self):
        self.scorefxn = pyrosetta.get_fa_scorefxn()
        self.scorefxn_low = create_score_function('score3')
        self.to_centroid = pyrosetta.rosetta.protocols.simple_moves.SwitchResidueTypeSetMover('centroid')
        self.to_fullatom = pyrosetta.rosetta.protocols.simple_moves.SwitchResidueTypeSetMover('fa_standard')
        return None

    def set_up_min_mover(self, move_map, scorefxn):
        self.min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
        self.min_mover.movemap(move_map)
        self.min_mover.score_function(scorefxn)
        return None

    def set_up_pack_mover(self, pose, scorefxn, res_list=None):
        task_pack = pyrosetta.standard_packer_task(pose)
        task_pack.restrict_to_repacking()
        task_pack.temporarily_fix_everything()
        if res_list is None:
            for i in range(1, 1+len(pose.sequence())):
                task_pack.temporarily_set_pack_residue(i, True)
        else:
            for i in res_list:
                task_pack.temporarily_set_pack_residue(i, True)
        self.packmover = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn, task_pack)
        return None

    def set_up_relax(self, scorefxn):
        self.relax = pyrosetta.rosetta.protocols.relax.FastRelax()
        self.relax.set_scorefxn(scorefxn)
        return None

    def set_up_monte_carlo(self, pose, scorefxn, kT):
        self.mc = MonteCarlo(pose, scorefxn, kT)
        return None

    def read_fasta(self, file_name):
        with open(file_name, 'r') as f:
            sequence = f.readlines()
        sequence = [line.strip() for line in sequence if not '>' in line]
        sequence = ''.join(sequence)
        return sequence

    def get_graph(self, pose, score_fxn):
        return prot2graph(pose, score_fxn)
