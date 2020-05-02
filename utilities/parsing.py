"""
Code handle the arguments
"""

from argparse import ArgumentParser, Namespace
import json


def add_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.
    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('-p', '--parameters', default=None,
                        help='The network parameters to begin with.')
    parser.add_argument('-t', '--task', default='test',
                        help='The task name.')
    parser.add_argument('-c', '--path_to_config', default=None,
                        help='The JSON file define the hyper parameters.')
    parser.add_argument('-m', '--model_path', default='./checkpoints',
                        help='path to put output files.')
    parser.add_argument('-o', '--gen_path', default='./results'
                        help='The file to store results.')
    parser.add_argument('-n', '--number', default=100,
                        help='Number of molecules to keep')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Whether to see the intermediate information.')
    parser.add_argument('--save_frequency', default=1000,
                        help='The frequency to save a checkpoint.')
    parser.add_argument('--log_frequency', default=100,
                        help='The frequency to log result.')

    # Protein environment arguments
    parser.add_argument('--start_pdb', default=None,
                        help='The path to the pdb file to start with')
    parser.add_argument('--ref_pdb', default=None,
                        help='The path to reference pdb file')
    parser.add_argument('--fastq', default=None,
                        help='The path to the fastq file')
    parser.add_argument('--loop', default=None,
                        help='The path to the loop file')
    parser.add_argument('--max_steps', default=100000000,
                        help='Maximum steps allowed.')
    parser.add_argument('--discount_factor', default=1,
                        help='The discount factor of reward.')

    # Reinforcement Learning arguments
    parser.add_argument('--num_episodes', default=10,
                        help='Number of episodes to run.')
    parser.add_argument('--replay_buffer_size', default=5000,
                        help='The action reward sets to store in the replay buffer.')
    parser.add_argument('--exploration', default='bootstrap',
                        choices=['bootstrap', 'disagreement'],
                        help='Choose the exploration method.')
    # PPO
    parser.add_argument('--batch_size', default=10,
                        help='The training batch size.')
    parser.add_argument('--gamma', default=0.99,
                        help='The normally defined discount factor.')
    parser.add_argument('--tau', default=0.95,
                        help='The GAE discount factor.')
    parser.add_argument('--clip_param', default=0.2,
                        help='The PPO clip factor.')
    parser.add_argument('--update_frequency', default=20,
                        help='The network update frequency.')

    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--optimizer', default='Adam',
                        help='The opitmizer to use.')
    parser.add_argument('--batch_norm', action='store_true', default=False,
                        help='Use the batch normalization or not.')
    parser.add_argument('--dropout', default=0,
                        help='The dropout probability.')
    parser.add_argument('--adam_beta_1', default=0.9,
                        help='The beta_1 in adam optimizer.')
    parser.add_argument('--adam_beta_2', default=0.999,
                        help='The beta_2 in adam optimizer.')
    parser.add_argument('--grad_clipping', default=10,
                        help='The gradient clipping.')
    parser.add_argument('--learning_frequency', default=4,
                        help='The frequency of learning, steps')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='The learning rate to begin with.')
    parser.add_argument('--learning_rate_decay_steps', default=1000,
                        help='Learning rate decay steps.')
    parser.add_argument('--learning_rate_decay_rate', default=0.8,
                        help='Learning rate decay rate.')

    # Graph network part
    parser.add_argument('--node_dim', default=40,
                        help='The dimension of node feature vector. 40 for low-resolution, 49 for high-resolution. Default is 40.')
    parser.add_argument('--edge_dim', default=23,
                        help='The dimension of edge feature vector. 23 for low-resolution, 32 for high-resolution. Default is 23.')
    parser.add_argument('--hidden_dim', default=64,
                        help='The dimension of hidden vector.')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='To use bias in graph network or not.')
    parser.add_argument('--processing_steps', default=3,
                        help='The depth of set2set self-attention layer.')
    parser.add_argument('--depth', default=3,
                        help='The message passing depth of graph network.')
    parser.add_argument('--ffn_hidden_size', default=300,
                        help='The hidden size of following ffn.')
    parser.add_argument('--ffn_num_layers', default=2,
                        help='The number of layers of following ffn.')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='To use atom features instead of concatenation of atom and bond')
    parser.add_argument('--feature_only', action='store_true', default=True,
                        help='Only use the artificial features.')
    parser.add_argument('--use_input_features', action='store_true', default=True,
                        help='Concatenate input features.')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching computation.')


def modify_args(args: Namespace):
    """Modify the arguments and read json configuration file to overwrite."""
    hparams = {}
    if args.path_to_config is not None:
        with open(args.path_to_config, 'r') as f:
            hparams.update(json.load(f))

        for key, value in hparams.items():
            setattr(args, key, value)
    return args


def parse_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).
    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    modify_args(args)

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
