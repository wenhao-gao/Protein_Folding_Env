"""
Code define the loop modeling environment
"""
from pyrosetta import *
from pyrosetta.rosetta.protocols.moves import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.protocols.simple_moves import *
from pyrosetta.rosetta.protocols.minimization_packing import *

from environment.Protein_Environment import Protein_Modeling_Environment


class Loop_Modeling_Environment(Protein_Modeling_Environment):

    environment_name = "Loop Modeling"

    def __init__(self, ref_pdb=None, sequence=None, max_angle=30.0, kT=1.0, num=16):
        super(Loop_Modeling_Environment, self).__init__()
        self.kT = kT
        self.max_angle = max_angle

        if '.pdb' in ref_pdb:
            self.native_pose = pose_from_pdb(ref_pdb)
        else:
            self.native_pose = ref_pdb

        if isinstance(sequence, str):
            if '.fasta' in sequence:
                self.pose = pose_from_sequence(self.read_fasta(sequence))
            else:
                self.pose = pose_from_sequence(sequence)
        elif sequence is None:
            sequence = self.native_pose.sequence()
            self.pose = pose_from_sequence(sequence)

        self.to_centroid.apply(self.pose)
        self.to_centroid.apply(self.native_pose)
        self.starting_pose = self.pose.clone()

        self.movemap = MoveMap()
        self.movemap.set_bb(True)

        self.set_up_monte_carlo(self.pose, self.scorefxn_low, self.kT)
        self.set_up_min_mover(self.movemap, self.scorefxn_low)

        self.min_native_pose = self.native_pose.clone()
        self.min_mover.apply(self.min_native_pose)
        self.native_score = self.scorefxn_low(self.min_native_pose)

        self.rnd_mover = RandomTorsionMover(move_map=self.movemap, max_angle=self.max_angle, num_moves=1)

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
        reward = self.reward()
        self.mc.boltzmann(self.pose)
        score_after_mc = self.scorefxn_low(self.pose)
        rmsd = CA_rmsd(self.native_pose, self.pose)
        done = 0
        next_state = self.get_graph(self.pose, self.scorefxn_low)
        return next_state, reward, done, score_before_mc, score_after_mc, rmsd

    def reset(self):
        """Reset the whole environment. This function is for debug mode"""
        self.pose = self.starting_pose.clone()
        self.set_up_monte_carlo(self.pose, self.scorefxn_low, self.kT)

        print(f'Starting score: {self.scorefxn_low(self.pose)}, Native score: {self.native_score}')

        return self.get_graph(self.pose, self.scorefxn_low)

    def reward(self):
        """Return the reward of the task. This method must be overwritten by any environment"""
        return - self.scorefxn_low(self.pose)

    def done(self):
        self.mc.recover_low(self.pose)
        self.min_mover.apply(self.pose)
        return self.pose, self.scorefxn_low(self.pose)

    def read_action(self, action):
        res_ind = (action[0] / 2).astype(int) + 1
        torsion_type = action[0] % 2
        value = action[1]
        return res_ind[0], torsion_type[0], value[0]
