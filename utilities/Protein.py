"""
Utility function for Protein Environment
"""
import numpy as np
import torch
from torch_geometric.data import Data

aa2index = {}
index2aa = {}
aas = 'RHKDESTNQCGPAVILMFYW'
for i, aa in enumerate(aas):
    aa2index[aa] = i
    index2aa[i] = aa


def min_gaussian(x, mu=3, sigma=3):
    mod_score = np.maximum(x, mu)
    return np.exp(-0.5 * np.power((mod_score - mu) / sigma, 2.))


def sigmoid(x, a=10 / 80, b=20):
    """
    a: about 10 / Range
    b: about the poisiton of center of the score function range
    """
    return 1 - 1 / (1 + np.exp(- a * (x - b)))


def one_hot(x):
    """
    Return the one-hot encoded
    :param x:
    :return:
    """
    if isinstance(x, str):
        seq = x
    else:
        seq = x.sequence()

    feature = []
    for aa in seq:
        f = np.zeros(20)
        f[aa2index[aa]] = 1
        feature.append(f)
    return np.array(feature).astype(int)


def node_feature(pose, res_ind, energy):
    feature = list(one_hot(pose.residue(res_ind).name1())[0])

    feature.append(np.sin(pose.phi(res_ind) * np.pi / 180.))
    feature.append(np.cos(pose.phi(res_ind) * np.pi / 180.))
    feature.append(np.sin(pose.psi(res_ind) * np.pi / 180.))
    feature.append(np.cos(pose.psi(res_ind) * np.pi / 180.))
    feature.append(np.sin(pose.omega(res_ind) * np.pi / 180.))
    feature.append(np.cos(pose.omega(res_ind) * np.pi / 180.))

    xyz = pose.residue(res_ind).xyz("CA") - pose.residue(res_ind).xyz("C")
    xyz = xyz.normalized()
    feature.append(xyz.x)
    feature.append(xyz.y)
    feature.append(xyz.z)

    # try:
    feature = feature + list(energy[res_ind - 1])
    # except:
    #     print(energy.shape, res_ind)

    return feature


def rbf(d, d_count=10, d_max=12., d_min=0.):
    # Distance radial basis function
    d_mu = torch.linspace(d_min, d_max, d_count)
    d_mu = d_mu.view([1, 1, 1, -1])
    d_sigma = (d_max - d_min) / d_count
    d_expand = torch.tensor([d for i in range(d_count)])
    rbf = torch.exp(-((d_expand - d_mu) / d_sigma) ** 2)
    return list(rbf.view(-1, ).numpy())


def coord(pose, res_id):
    N_xyz = pose.residue(res_id).xyz("N")
    CA_xyz = pose.residue(res_id).xyz("CA")
    C_xyz = pose.residue(res_id).xyz("C")

    x1 = N_xyz - CA_xyz
    x1 = x1.normalized()
    x2 = C_xyz - CA_xyz
    x2 = x2.normalized()

    b = x2 - x1
    b = b.normalized()
    n = x2.cross(x1)
    n = n.normalized()
    j = n.cross(b)

    return np.array([
        [b.x, n.x, j.x],
        [b.y, n.y, j.y],
        [b.z, n.z, j.z]
    ])


def quaternions(R):
    Rxx, Ryy, Rzz = R.diagonal()
    magnitudes = 0.5 * np.sqrt(np.abs(1 + np.array([Rxx - Ryy - Rzz,
                                                    - Rxx + Ryy - Rzz,
                                                    - Rxx - Ryy + Rzz])))
    signs = np.sign(np.array([R[2, 1] - R[1, 2],
                              R[0, 2] - R[2, 0],
                              R[1, 0] - R[0, 1]]))
    xyz = signs * magnitudes
    w = np.sqrt(np.maximum((1 + Rxx + Ryy + Rzz), 0)) / 2.
    Q = np.append(xyz, w)
    Q = Q / np.linalg.norm(Q)
    return list(Q)


def edge_feature(pose, res_ind1, res_ind2, energy, hbonds, qs):
    CA1_xyz = pose.residue(res_ind1).xyz("CA")
    CA2_xyz = pose.residue(res_ind2).xyz("CA")
    D_vector = CA1_xyz - CA2_xyz
    D12 = D_vector.norm()
    D_vector = D_vector.normalized()

    feature = rbf(D12) + [D_vector.x, D_vector.y, D_vector.z]

    pos = res_ind1 - res_ind2

    feature = feature + list(qs[res_ind1 - 1, res_ind2 - 1])
    # feature = feature + [np.sin(pos * np.pi / 180.), np.cos(pos * np.pi / 180.)]
    feature.append(pos / 40)
    feature = feature + list(energy[res_ind1 - 1][res_ind2 - 1])

    hbond = 0
    hblist1 = hbonds.residue_hbonds(res_ind1)
    hblist2 = hbonds.residue_hbonds(res_ind2)
    if len(hblist1) != 0 and len(hblist2) != 0:
        for hb in hblist1:
            if hb in hblist2:
                hbond = - hb.energy()
    feature.append(hbond)

    return feature


def prot2graph(pose, score_fxn):
    score_fxn(pose)
    energy = pose.energies().residue_onebody_energies_array()
    x = [node_feature(pose, res_ind, energy) for res_ind in range(1, 1 + len(pose.sequence()))]
    x = torch.tensor(x, dtype=torch.float)

    energy = pose.energies().residue_pair_energies_array()
    hbonds = pose.get_hbonds()
    coords = []
    for i in range(1, len(pose.sequence()) + 1):
        coords.append(coord(pose, i))
    qs = np.zeros((len(coords), len(coords), 4))
    for i in range(len(coords)):
        for j in range(i):
            qs[i, j] = quaternions(coords[i].T * coords[j])
    edge_index = []
    edge_attr = []
    for i in range(1, 1 + len(pose.sequence())):
        for j in range(1, i):
            edge_index.append([i-1, j-1])
            edge_index.append([j-1, i-1])
            edge_f = edge_feature(pose, i, j, energy, hbonds, qs)
            edge_attr.append(edge_f)
            edge_attr.append(edge_f)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)


def get_distance(pose, res_ind1, res_ind2):
    CA1_xyz = pose.residue(res_ind1).xyz("CA")
    CA2_xyz = pose.residue(res_ind2).xyz("CA")
    D_vector = CA1_xyz - CA2_xyz
    return D_vector.norm()


def get_neighbor(pose, loop, cutoff=12.0):
    neighbor = set()
    for i in range(1, len(pose.sequence())+1):
        for j in range(loop.start(), loop.stop()+2):
            if get_distance(pose, i, j) <= cutoff:
                neighbor.add(i)
    loop_list = []
    for i in range(loop.start(), loop.stop()+1):
        neighbor.remove(i)
        loop_list.append(i)
    return loop_list + list(neighbor)


def get_position(pose, res_ind):
    N_xyz = pose.residue(res_ind).xyz("N")
    CA_xyz = pose.residue(res_ind).xyz("CA")
    C_xyz = pose.residue(res_ind).xyz("C")
    return np.array(list(N_xyz) + list(CA_xyz) + list(C_xyz))


def normalize_torsion(degree):
    while degree >= 180.0:
        degree -= 360.0
    while degree < -180.0:
        degree += 360.0
    return degree
