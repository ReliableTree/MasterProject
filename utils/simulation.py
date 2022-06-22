from cProfile import label
import torch
import gym
from torch.utils.data import DataLoader

import torch
from torch.utils.data import DataLoader
from zmq import device
import numpy as np

class ToySimulation():
    def __init__(self, neg_tol, pos_tol, check_outpt_fct, dataset, result_size, window = 9) -> None:
        self.dataset = dataset     
        self.neg_tol = neg_tol
        self.pos_tol = pos_tol
        self.check_outpt_fct = check_outpt_fct
        self.window = window
        self.result_size = result_size

    def get_seeds(self, n):
        indices = torch.randperm(len(self.dataset))[:n]
        return indices

    def simulate(self, policy, seeds):
        envs = seeds
        trajectories = []
        inpt_obs = []
        success = []
        labels = []
        critic_scores = []
        subset = torch.utils.data.Subset(self.dataset, envs)
        loader = DataLoader(subset, batch_size=200, shuffle=False)
        for succ, fail in loader:
            obsv, label, succ = succ
            output_seq = policy.forward(task_embedding=obsv)
            critic_score = policy.get_critic_score(task_embedding=obsv, last_seq=output_seq)
            trajectories += [output_seq.detach()]
            inpt_obs.append(obsv.detach())
            success.append(self.check_outpt_fct(label=label, outpt=output_seq, tol_neg=self.neg_tol, tol_pos=self.pos_tol, window=self.window))
            labels.append(label.detach())
            critic_scores += [critic_score.detach()]

        trajectories = torch.cat([*trajectories], dim=0)
        inpt_obs = torch.cat([*inpt_obs], dim=0)
        success = torch.cat([*success], dim=0)
        labels = torch.cat([*labels], dim=0)
        critic_scores = torch.cat([*critic_scores], dim=0)

        return trajectories, inpt_obs, labels, success, critic_scores


class HERSimulation():
    def __init__(self) -> None:
        self.neg_tol = None
        self.pos_tol = None
        self.window = 0

    def pars_obsv(self, result):
        obsv = result['observation']
        goal = result['desired_goal']
        parsed = {}
        parsed['hand_pos'] = obsv[:3]
        parsed['puck_pos'] = obsv[3:6]
        parsed['puck_rot'] = obsv[9:13]
        parsed['goal_pos'] = goal
        return parsed

    def make_obsv(self, result):
        obs_dict = self.pars_obsv(result)
        return(np.concatenate((obs_dict['hand_pos'], obs_dict['puck_pos'], obs_dict['puck_rot'], obs_dict['goal_pos']), axis=0))


    def check_outpt_fct(self, env, outpt, render = False):
        #success, label
        labels = []
        device = outpt.device
        outpt = outpt.cpu().numpy()
        for pred in outpt[0]:
            #pred = 29
            action = pred
            env_action = np.copy(action)
            env_action[-1] = env_action[-1]/10
            result = env.step(action)
            labels += [torch.tensor(action).unsqueeze(0)]
        labels = torch.cat([*labels], dim=0).to('cuda')
        success = torch.tensor(result[-1]['is_success']).type(torch.bool).to(device)
        return success.unsqueeze(0), labels.unsqueeze(0)

    def get_env(self, n, env_tag):
        seeds = torch.randint(0,int(1e10), [n])
        return seeds

    def get_simulation_input(self, seed, device):
        env = gym.make('FetchPickAndPlace-v1')
        env.seed(int(seed))

        result = env.reset()
        obsv = self.make_obsv(result=result)
        return torch.tensor(obsv, dtype=torch.float).to(device).reshape(1,-1), env

    def get_success(self, policy, envs, device='cuda'):
        trajectories = []
        inpt_obs = []
        successes = []
        labels = []

        for seed in envs:
            inpt_obsv, env = self.get_simulation_input(seed, device=device)
            try:
                policy = policy.to(device)
            except:
                pass
            output_seq = policy.forward(inpt_obsv.unsqueeze(1))['gen_trj'].detach()
            #print(f'output_seq.shape: {output_seq.shape}')
            success, label = self.check_outpt_fct(env=env, outpt=output_seq, render=False)

            trajectories += [output_seq]
            inpt_obs += [inpt_obsv]
            successes += [success.reshape(1,1)]
            labels += [label]
            #print(f'labels shape: {label.shape}')


        trajectories = torch.cat([*trajectories], dim=0).to('cuda')
        inpt_obs = torch.cat([*inpt_obs], dim=0).to('cuda')
        successes = torch.cat([*successes], dim=0).to('cuda')
        labels = torch.cat([*labels], dim=0).to('cuda')


        return trajectories, inpt_obs.unsqueeze(1), labels, successes.squeeze(), trajectories
