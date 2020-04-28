import numpy as np
from pyrosetta import *
init()

from pyrosetta.rosetta.protocols.moves import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.protocols.simple_moves import *
from pyrosetta.rosetta.protocols.minimization_packing import *
from tqdm import tqdm


if __name__ == "__main__":
    pose = pose_from_sequence("VIHCDAATICPDGTTCSLSPYGVWYCSPFS")
    switch = SwitchResidueTypeSetMover("centroid")
    switch.apply(pose)
    sfxn = create_score_function("score3")
    starting_pose = pose.clone()
    
    native_pose = pose_from_pdb("1qgm.pdb")
    switch.apply(native_pose)

    movemap = MoveMap()
    movemap.set_bb(True)

    n_moves = 1
    angles = 30.0
    kT = 1.0
    rnd_mover = RandomTorsionMover(move_map=movemap, max_angle=angles, num_moves=n_moves)

    pose = starting_pose.clone()
    mc = MonteCarlo(pose, sfxn, kT)

    score_womc = []
    score_wmc = []
    rmsd = []
    for i in tqdm(range(10000)):
        rnd_mover.apply(pose)
        score_womc.append(sfxn(pose))
        mc.boltzmann(pose)
        score_wmc.append(sfxn(pose))
        rmsd.append(CA_rmsd(native_pose, pose))

    score_womc = np.array(score_womc)
    score_wmc = np.array(score_wmc)
    rmsd = np.array(rmsd)
    mc.recover_low(pose)

    min_mover = MinMover()
    min_mover.set_movemap(movemap)
    min_mover.score_function(sfxn)

    min_mover.apply(pose)
    print("===========Centroid Score===========")
    print(f"Native Score: {sfxn(native_pose)}")
    print(f"Starting Score: {sfxn(starting_pose)}")
    print(f"Decoy Score: {sfxn(pose)}")
    print(f"CA RMSD: {CA_rmsd(native_pose, pose)}")

    pose.dump_pdb("output_file.pdb")
    np.save('score_womc.npy', score_womc)
    np.save('score_wmc.npy', score_wmc)
    np.save('rmsd.npy', rmsd)
