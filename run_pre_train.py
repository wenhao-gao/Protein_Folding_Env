import os
import random
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import pyrosetta
pyrosetta.init()

from environment.Protein_Folding import Protein_Folding_Environment
from networks.protein_folding_pretrain import Net_pre
from utilities.parsing import parse_args
import ipdb


EPOCH = 1000000
POOL_SIZE = 1
LR = 0.001
MINI_BATCH = 1
POSE_SIZE = 64
LOG_PATH = './checkpoints'
SAVE_PATH = './model_parameters/model'
DATA_PATH = 'data/protein_folding_pretrain'
PARAM_PATH = 'model_parameters/model1/model.pth'


def main():

    args = parse_args()

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    writer = SummaryWriter(LOG_PATH)
    env = Protein_Folding_Environment(ref_pdb='./data/protein_folding/1l2y.pdb')

    pdb_list = [pdb for pdb in os.listdir(DATA_PATH) if '.pdb' in pdb]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using Device: {device}')
    net = Net_pre(args, device)

    if PARAM_PATH is not None:
        net.load_state_dict(torch.load(PARAM_PATH))

    net.to(device)
    net.train()

    optimizer = torch.optim.Adam(params=net.parameters(), lr=LR)
    loss_classification = torch.nn.CrossEntropyLoss()
    loss_regression1 = torch.nn.L1Loss()
    loss_regression2 = torch.nn.MSELoss()

    batch = 0

    for epoch in range(EPOCH):
        for pdb_file in tqdm(random.sample(pdb_list, POOL_SIZE)):

            pose = pyrosetta.pose_from_pdb(os.path.join(DATA_PATH, pdb_file))
            env.to_centroid.apply(pose)
            pose_length = len(pose.sequence())

            if pose_length > POSE_SIZE:
                start = random.randint(1, pose_length - POSE_SIZE + 1)
                end = start + POSE_SIZE - 1
                pyrosetta.rosetta.protocols.grafting.delete_region(pose, end + 1, pose_length)
                pyrosetta.rosetta.protocols.grafting.delete_region(pose, 1, start - 1)
                pose_length = POSE_SIZE

            batch += 1
            optimizer.zero_grad()

            datas = []
            t1 = []
            t2 = []
            t3 = []

            for i in range(MINI_BATCH):
                random.seed()
                action = random.randint(0, pose_length * 2 - 1)
                res_ind = int(action / 2) + 1
                torsion_type = action % 2
                action_value = random.uniform(-1, 1)
                score = 0 - env.scorefxn_low(pose)

                try:
                    if torsion_type == 0:
                        pose.set_phi(res_ind, action_value * 30.0)
                    elif torsion_type == 1:
                        pose.set_psi(res_ind, action_value * 30.0)
                except:
                    ipdb.set_trace()

                datas.append(env.get_graph(pose, env.scorefxn_low))
                t1.append([score])
                t2.append(action)
                t3.append([0 if i != action else - action_value for i in range(pose_length * 2)])

            target1 = torch.Tensor(t1)
            target2 = torch.LongTensor(t2)
            target3 = torch.Tensor(t3)

            target1 = target1.to(device)
            target2 = target2.to(device)
            target3 = target3.to(device)
            
            output1, output2, output3 = net.pre_train(datas)

            loss1 = loss_regression1(output1, target1)
            loss2 = loss_classification(output2, target2)
            loss3 = loss_regression2(output3, target3)

            loss = loss1 * 0.01 + loss2 * 5 + loss3

            loss.backward()
            optimizer.step()

            writer.add_scalar('total loss', loss, batch)
            writer.add_scalar('score loss', loss1, batch)
            writer.add_scalar('action1 loss', loss2, batch)
            writer.add_scalar('action2 loss', loss3, batch)

        if epoch % 2000 == 1999:
            torch.save(net.state_dict(), SAVE_PATH + str(epoch+1) + '.pth')

    torch.save(net.state_dict(), SAVE_PATH + '.pth')

    return None


if __name__ == '__main__':
    main()
