import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from agents.basic_policy import Policy
import ipdb
from tensorboardX import SummaryWriter


class PPO(Policy):
    def __init__(self, model, env, args=None, device='cpu'):
        super(PPO, self).__init__(model, env, args, device)

        self.ppo_epochs = 4
        self.max_steps = args.max_steps
        self.mini_batch_size = args.batch_size
        self.clip_param = self.args.clip_param
        self.update_frequency = self.args.update_frequency

    def train(self):

        action1 = []
        action2 = []

        step = 0
        state = self.env.reset()

        while step < self.max_steps:

            log_probs = []
            values = []
            states = []
            actions = []
            act_vals = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(self.update_frequency):

                state.to(self.device)
                dist, value = self.model(state)
                action = dist.sample()
                dist_value = self.model.get_value(action)
                # ipdb.set_trace()
                action_value = dist_value.sample()

                action1.append(action.cpu().numpy())
                action2.append(action_value.cpu().numpy())

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

                # Keep track the result
                self.tracker.insert((self.env.pose.clone(), score_after_mc, rmsd))

                if step % self.args.log_frequency == 0:

                    print(f'frame_idx: {step}   Rosetta score: {score_after_mc}   RMSD: {rmsd}')
                    self.writer.add_scalar('score_before_mc', score_before_mc, step)
                    self.writer.add_scalar('score_after_mc', score_after_mc, step)
                    self.writer.add_scalar('rmsd', rmsd, step)
                    self.writer.add_scalar('action1', action, step)
                    self.writer.add_scalar('action2', action_value, step)

                if step % self.args.save_frequency == 0:

                    self.tracker.save(self.args.gen_path, self.task)

            next_state = next_state.to(self.device)
            _, next_value = self.model(next_state)
            returns = self.compute_gae(next_value, rewards, masks, values, gamma=self.gamma, tau=self.tau)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = states
            actions1 = torch.cat(actions).detach()
            actions2 = torch.cat(act_vals).detach()
            actions = (actions1, actions2)
            advantage = returns - values

            self.ppo_update(ppo_epochs=self.ppo_epochs,
                            mini_batch_size=self.mini_batch_size,
                            states=states,
                            actions=actions,
                            log_probs=log_probs,
                            returns=returns,
                            advantages=advantage,
                            clip_param=self.clip_param)

        action1 = np.array(action1)
        action2 = np.array(action2)
        np.save(self.task + '_action1.npy', action1)
        np.save(self.task + '_action2.npy', action2)

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
