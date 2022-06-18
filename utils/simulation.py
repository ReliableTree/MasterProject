from cProfile import label
import torch
import gym
from torch.utils.data import DataLoader

import torch
from torch.utils.data import DataLoader
from zmq import device

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

    def check_outpt_fct(self, env, outpt, render = False):
        #success, label
        labels = []
        device = outpt.device
        outpt = outpt.cpu().numpy()
        for pred in outpt[0]:
            #pred = 29
            action = pred[-4:]
            result = env.step(action)
            obsv = torch.tensor(result[0]['observation']) 
            label = torch.cat((obsv.unsqueeze(0), torch.tensor(action).unsqueeze(0)), dim=-1)
            labels += [label]

        labels = torch.cat([*labels], dim=0).to(device)
        success = torch.tensor(result[-1]['is_success']).type(torch.bool).to(device)
        return success.unsqueeze(0), labels.unsqueeze(0)

    def get_seeds(self, n):
        seeds = torch.randint(0,int(1e10), [n])
        return seeds

    def get_simulation_input(self, seed, device):
        env = gym.make('FetchPickAndPlace-v1')
        env.seed(int(seed))

        result = env.reset()
        obsv = torch.tensor(result['observation']) 
        goal = torch.tensor(result['desired_goal'])
        #print('shapes:')
        #print(obsv.shape)
        #print(goal.shape)
        inpt_obsv = torch.cat((obsv.unsqueeze(0), goal.unsqueeze(0)), dim=-1).to(device)
        #print(inpt_obsv.shape)
        return inpt_obsv, env

    def simulate(self, policy, seeds, device):
        trajectories = []
        inpt_obs = []
        successes = []
        labels = []
        critic_scores = []

        HER_trajectories = []
        HER_inpt_obs = []
        HER_successes = []
        HER_labels = []
        HER_critic_scores = []

        for seed in seeds:
            inpt_obsv, env = self.get_simulation_input(seed, device=device)
            output_seq = policy.forward(task_embedding=inpt_obsv).detach()
            #print(f'output_seq.shape: {output_seq.shape}')
            critic_score = policy.get_critic_score(task_embedding=inpt_obsv, last_seq=output_seq, detach=True).detach()
            success, label = self.check_outpt_fct(env=env, outpt=output_seq, render=False)

            trajectories += [output_seq]
            inpt_obs += [inpt_obsv]
            successes += [success.reshape(1,1)]
            labels += [label]
            critic_scores += [critic_score.reshape(1,1)]
            #print(f'labels shape: {label.shape}')

            if not success.reshape(1)[0]:
                HER_Observation = inpt_obsv.clone()
                HER_Observation[:,-3:] = label[:,-1,3:6]
                HER_label = label.clone()
                HER_success = torch.ones_like(success).type(torch.bool)
                HER_critic_score = policy.get_critic_score(task_embedding=HER_Observation, last_seq=label, detach=True).detach()
                HER_trajectories += [output_seq]
                HER_inpt_obs += [HER_Observation]
                HER_successes += [HER_success.reshape(1,1)]
                HER_labels += [HER_label]
                HER_critic_scores += [HER_critic_score.reshape(1,1)]


        trajectories = torch.cat([*trajectories], dim=0)
        inpt_obs = torch.cat([*inpt_obs], dim=0)
        successes = torch.cat([*successes], dim=0)
        labels = torch.cat([*labels], dim=0)
        critic_scores = torch.cat([*critic_scores], dim=0)

        HER_trajectories = torch.cat([*HER_trajectories], dim=0)
        HER_inpt_obs = torch.cat([*HER_inpt_obs], dim=0)
        HER_successes = torch.cat([*HER_successes], dim=0)
        HER_labels = torch.cat([*HER_labels], dim=0)
        HER_critic_scores = torch.cat([*HER_critic_scores], dim=0)

        return (trajectories, inpt_obs, labels, successes.squeeze(), critic_scores), \
            (HER_trajectories, HER_inpt_obs, HER_labels, HER_successes.squeeze(), HER_critic_scores)
