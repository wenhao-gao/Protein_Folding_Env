import torch
from environment.Protein_Folding import Protein_Folding_Environment
from networks.mpnn import Net
from utilities.parsing import parse_args
from agents.random import Random


def main():

    args = parse_args()

    env = Protein_Folding_Environment(ref_pdb=args.ref_pdb)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(args, device)
    print(f'Using Device: {device}')
    if args.parameters is not None:
        net.load_state_dict(torch.load(args.parameters))
    net.to(device)

    agent = Random(
        model=net,
        env=env,
        args=args,
        device=device
    )

    agent.train()
    agent.done()


if __name__ == "__main__":
    main()
