import torch
import gym
from torch.utils.data import DataLoader

import torch
from torch.utils.data import DataLoader
import numpy as np
from MasterProject.utils.utils import make_obsv_seq, append_obsv_seq, pars_obsv, concat_obsv, make_obsv

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

    def check_outpt_fct(self, env, policy, device = 'cuda'):
        #success, label
        labels = []
        obsvs = append_obsv_seq(env.reset(), device=device)
        success = False
        step = 0
        while not success and (step < 65):
            action = policy.forward(obsvs)[0,-1]
            env_action = np.copy(action)
            env_action[-1] = env_action[-1]/10
            n_obsv = env.step(env_action.detach().cpu().numpy())
            obsvs = append_obsv_seq(obsv=n_obsv[0], seq=obsvs, device=device)
            labels += [torch.tensor(action).unsqueeze(0)]
            success = n_obsv[-1]['is_success']
            step += 1

        labels = torch.cat([*labels], dim=0).to('cuda')
        obsvs = torch.cat([*obsvs], dim=0).to('cuda')
        return success.unsqueeze(0), labels.unsqueeze(0), obsvs

    def get_env(self, n, env_tag):
        seeds = torch.randint(0,int(1e10), [n])
        return seeds

    def get_simulation_input(self, seed, device):
        env = gym.make('FetchPickAndPlace-v1')
        env.seed(int(seed))

        result = env.reset()
        obsv_dict = pars_obsv(result=result)
        obsv = make_obsv(result=result)
        return torch.tensor(obsv, dtype=torch.float).to(device).reshape(1,-1), env, obsv_dict

    def get_success(self, policy, envs, device='cuda'):
        successes = []
        labels = []
        obsvs = []

        for seed in envs:
            env = gym.make('FetchPickAndPlace-v1')
            env.seed(int(seed))
            try:
                policy = policy.to(device)
            except:
                pass
            success, label, obsv = self.check_outpt_fct(env, policy=policy, device=device)
            successes += [success.reshape(1,1)]
            labels += [label]
            obsvs += [obsv]


        obsvs = torch.cat([*obsvs], dim=0)
        successes = torch.cat([*successes], dim=0).squeeze()
        labels = torch.cat([*labels], dim=0)

        '''for i in range((~successes).sum()):
            HER_dict = obsv_dicts[i]
            HER_dict['goal_pos'] = goals[i].squeeze()
            new_inpt = torch.tensor(concat_obsv(HER_dict), dtype=torch.float).to('cuda')
            inpt_obs = torch.cat((inpt_obs, new_inpt.unsqueeze(0)), dim=0)'''

        '''trajectories = torch.cat((trajectories, trajectories[~successes]), dim=0).to('cuda')
        labels = torch.cat((labels, labels[~successes]), dim=0).to('cuda')
        successes = torch.cat((successes, ~successes[~successes]), dim=0).to('cuda')'''
        '''print(f'trajectories: {trajectories.shape}')
        print(f'labels: {labels.shape}')
        print(f'inpt_obs: {inpt_obs.shape}')
        print(f'successes: {successes.shape}')'''

        return obsvs, labels, successes
