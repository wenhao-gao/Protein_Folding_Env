from agents.basic_policy import Policy


class Random(Policy):
    def __init__(self, model, env, args=None, device='cpu'):
        super(Random, self).__init__(model, env, args, device)

        self.ppo_epochs = 4
        self.max_steps = args.max_steps
        self.mini_batch_size = args.batch_size
        self.clip_param = self.args.clip_param
        self.update_frequency = self.args.update_frequency

    def train(self):

        step = 0
        state = self.env.reset()

        while step < self.max_steps:

            state.to(self.device)
            dist, value = self.model(state)
            action = dist.sample()
            dist_value = self.model.get_value(action)
            # ipdb.set_trace()
            action_value = dist_value.sample()

            next_state, reward, done, score_before_mc, score_after_mc, rmsd = self.env.step(
                (action.cpu().numpy(), action_value.cpu().numpy()))

            state = next_state
            step += 1

            # Keep track the result
            self.tracker.insert((self.env.pose.clone(), score_after_mc, rmsd))

            if step % self.args.log_frequency == 0:

                print(f'frame_idx: {step}   Rosetta score: {score_after_mc}   RMSD: {rmsd}')
                self.writer.add_scalar('score_before_mc', score_before_mc, step)
                self.writer.add_scalar('score_after_mc', score_after_mc, step)
                self.writer.add_scalar('rmsd', rmsd, step)
                self.writer.add_scalar('lowest', self.tracker.lowest, step)
                self.writer.add_scalar('highest', self.tracker.highest, step)
                self.writer.add_scalar('action1', action, step)
                self.writer.add_scalar('action2', action_value, step)

            if step % self.args.save_frequency == 0:

                self.tracker.save(self.args.gen_path, self.task)
                self.env.pose.dump_pdb(os.path.join(self.args.gen_path, self.task + '_traj_' + str(step) + '.pdb'))
