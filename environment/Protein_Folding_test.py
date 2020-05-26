"""
Code define the protein folding environment
"""
from pyrosetta import *
from pyrosetta.rosetta.protocols.moves import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.protocols.simple_moves import *
from pyrosetta.rosetta.protocols.minimization_packing import *

from environment.Protein_Folding import Protein_Folding_Environment


class Protein_Folding_Test_Environment(Protein_Folding_Environment):

    def step(self, action=None):
        """Takes a step in the modeling problem."""
        if action is None:
            self.rnd_mover.apply(self.pose)
        else:
            res_ind, torsion_type, value = self.read_action(action)
            if torsion_type == 0:
                self.pose.set_phi(res_ind, value * self.max_angle)
            elif torsion_type == 1:
                self.pose.set_psi(res_ind, value * self.max_angle)

        score_before_mc = self.scorefxn_low(self.pose)
        # reward = self.reward()
        if self.if_mc:
            keep = self.mc.boltzmann(self.pose)
        else:
            keep = 0
        score_after_mc = self.scorefxn_low(self.pose)
        rmsd = CA_rmsd(self.native_pose, self.pose)
        reward = 0 - rmsd
        done = 0
        next_state = self.get_graph(self.pose, self.scorefxn_low)
        return next_state, reward, done, score_before_mc, score_after_mc, rmsd, keep


