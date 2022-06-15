import torch
from torch.utils.data import DataLoader

class ToySimulation():
    def __init__(self, neg_tol, pos_tol, check_outpt_fct, dataset, result_size, window = 9) -> None:
        self.dataset = dataset     
        self.neg_tol = neg_tol
        self.pos_tol = pos_tol
        self.check_outpt_fct = check_outpt_fct
        self.window = window
        self.result_size = result_size

    def get_env(self, n, env_tag):
        indices = torch.randperm(len(self.dataset))[:n]
        return indices

    def simulate(self, policy, envs):
        trajectories = []
        inpt_obs = []
        success = []
        labels = []
        critic_scores = []
        subset = torch.utils.data.Subset(self.dataset, envs)
        loader = DataLoader(subset, batch_size=200, shuffle=False)
        for succ, fail in loader:
            obsv, label, succ = succ
            result = policy.forward(task_embedding=obsv)
            output_seq = result
            #output_seq = policy.forward(task_embedding=obsv)
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