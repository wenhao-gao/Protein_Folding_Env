import torch.nn as nn
import torch.optim as optim
import os
from collections import OrderedDict
from tensorboardX import SummaryWriter


class Policy(object):
    def __init__(self,
                 model,
                 env,
                 args=None,
                 device='cpu'):
        # General attribute
        self.args = args
        self.task = self.args.task
        self.env = env
        self.model = model
        self.device = device

        self.gamma = self.args.gamma
        self.tau = self.args.tau

        # Training attribute
        self.learning_frequency = self.args.learning_frequency
        self.learning_rate_decay_steps = self.args.learning_rate_decay_steps
        self.grad_clipping = self.args.grad_clipping

        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta_1, self.args.adam_beta_2),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
        )

        self.lr_schedule = optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=self.args.learning_rate_decay_rate
        )

        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)
        self.log_path = os.path.join(self.args.model_path, self.task)
        self.writer = SummaryWriter(self.log_path)

        # Logging attribute
        self.save_frequency = self.args.save_frequency
        self.tracker = Storage(keep=self.args.number)

    def train(self):
        raise NotImplementedError('Train function should be implemented by every algorithm')

    def done(self):
        pose, score = self.env.done()
        pose.dump_pdb("output.pdb")
        print(f'The lowest one: {score}')
        print(f'Native score: {self.env.native_score}')

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns


class Storage(object):
    """A class to store the training result"""

    def __init__(self, keep=3):
        self._item = OrderedDict()
        self._lowest = -999999
        self._highest = 999999
        self._keep = keep

    def insert(self, sample):
        """
        Insert a new sample into the tracker.
        :param sample: A tuple with (pose, energy, rmsd)
        """
        pose, energy, rmsd = sample
        self._item[pose] = (energy, rmsd)
        self.renormalize()

    def renormalize(self):
        """
        Keep the order of the protein poses w.r.t reward
        """
        item_in_list = sorted(self._item.items(), key=lambda t: (t[1][0], t[1][1]), reverse=False)[:self._keep]
        self._lowest = item_in_list[-1][1][0]
        self._highest = item_in_list[0][1][0]
        self._item = OrderedDict(item_in_list)
        return None

    def save(self, path, task):
        item_in_list = self._item.items()
        for i in range(len(item_in_list)):
            filename = task + '_' + str(i) + '.pdb'
            save_name = os.path.join(path, filename)
            pose = item_in_list[i][0]
            pose.dump_pdb(save_name)

    @property
    def content(self):
        return self._item

    @property
    def highest(self):
        return self._highest

    @property
    def lowest(self):
        return self._lowest


class MyDataParallel(nn.DataParallel):
    """Class for multi-GPU training network wrapper"""
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.module, item)
