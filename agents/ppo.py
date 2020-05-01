import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from collections import OrderedDict
import ipdb
from tensorboardX import SummaryWriter


class PPO(object):
    def __init__(self,
                 task,
                 model,
                 env,
                 args=None,
                 param=None,
                 keep=100,
                 model_path='./checkpoints',
                 gen_file='./result_'):
        """
        Initialization of the agent

        :param task:
        :param env:
        :param args:
        :param param:
        :param keep:
        :param model_path:
        :param gen_file:
        """
        # General attribute
        self.task = task
        self.env = env
        self.model = model
        self.args = args
        self.num_episodes = self.args.num_episodes
        self.max_steps_per_episode = self.args.max_steps_per_episode
        self.batch_size = self.args.batch_size
        self.gamma = self.args.gamma
        self.ppo_epochs = 4

        # Deep Network Attribute
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        print(f'Using Device: {self.device}')
        self.model.to(self.device)

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

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.log_path = os.path.join(model_path, self.task)
        self.writer = SummaryWriter(self.log_path)

        # Logging attribute
        self.save_frequency = self.args.save_frequency
        self.tracker = Storage(keep=keep)

    def train(self):
        max_frames = 100000000
        step = 0
        num_steps = 20
        mini_batch_size = 5

        state = self.env.reset()

        while step < max_frames:

            log_probs = []
            values = []
            states = []
            actions = []
            act_vals = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(num_steps):

                state.to(self.device)
                dist, value = self.model(state)
                action = dist.sample()
                dist_value = self.model.get_value(action)
                # ipdb.set_trace()
                action_value = dist_value.sample()

                next_state, reward, done, score_before_mc, score_after_mc, rmsd = self.env.step(
                    (action.cpu().numpy(), action_value.cpu().numpy()))

                log_prob = dist.log_prob(action) + dist_value.log_prob(action_value)
                entropy += dist.entropy().mean() + dist_value.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(np.array([reward])).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor(1 - np.array([done])).unsqueeze(1).to(self.device))

                states.append(state)
                actions.append(action)
                act_vals.append(action_value)

                state = next_state
                step += 1

                if step % 1 == 0:
                    print(f'frame_idx: {step}   Rosetta score: {score_after_mc}   RMSD: {rmsd}')

                    # Keep track the result
                    # if reward > self.tracker.lowest:
                    self.tracker.insert((self.env.pose.clone(), score_after_mc, rmsd))

                    # Log result
                    self.writer.add_scalar('score_before_mc', score_before_mc, step)
                    self.writer.add_scalar('score_after_mc', score_after_mc, step)
                    self.writer.add_scalar('rmsd', rmsd, step)

            next_state = next_state.to(self.device)
            _, next_value = self.model(next_state)
            returns = self.compute_gae(next_value, rewards, masks, values)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = states
            actions1 = torch.cat(actions).detach()
            actions2 = torch.cat(act_vals).detach()
            actions = (actions1, actions2)
            advantage = returns - values

            self.ppo_update(self.ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

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

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = len(states)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield [states[i] for i in rand_ids], (actions[0][rand_ids], actions[1][rand_ids]), log_probs[rand_ids], \
                  returns[rand_ids], advantage[rand_ids]

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions,
                                                                             log_probs, returns, advantages):
                state = [s.to(self.device) for s in state]
                dist, value = self.model(state)
                action1, action2 = action
                dist_value = self.model.get_value(action1)
                entropy = dist.entropy().mean() + dist_value.entropy().mean()
                new_log_probs = dist.log_prob(action1) + dist_value.log_prob(action2)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


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
