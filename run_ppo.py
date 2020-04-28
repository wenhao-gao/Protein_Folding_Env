from environment.Protein_Folding import Protein_Folding_Environment
from networks.mpnn import Net
from utilities.parsing import parse_args
from agents.ppo import PPO


def main():

    args = parse_args()
    env = Protein_Folding_Environment(ref_pdb='./data/protein_folding/1l2y.pdb')
    net = Net()
    agent = PPO(
        task='test',
        model=net,
        env=env,
        args=args,
        param=None,
        keep=100,
        model_path='./checkpoints',
        gen_file='./result_'
    )

    agent.train()
    agent.done()


if __name__ == "__main__":
    main()
