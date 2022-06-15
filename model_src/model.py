import torch
import torch.nn as nn
from pathlib import Path
import sys
from MasterProject.utils.utils import right_stack_obj_trj, make_inpt_seq
from MasterProject.model_src.transformer import TransformerModel, TransformerDecoder


class Model(nn.Module):
    def __init__(self, model_setup, opt_lr, writer, iterations=1, threshold = 0) -> None:
        super().__init__()
        self.self_init = False
        self.model_setup = model_setup
        self.opt_lr = opt_lr
        self.transformer = TransformerModel(model_setup=self.model_setup['transformer'])
        self.critic = TransformerDecoder(model_setup=self.model_setup['critic']) 
        self.seq_len = model_setup['seq_len'] + 1
        self.writer = writer
        self.iterations = iterations
        self.memory = {}
        self.threshold = threshold
        self.epochs = 1
        self.set_mode(0)

    def set_mode(self, mode):
        if mode == 0:
            self.forward = self.iter_forward
        else:
            self.forward = self.optimize

    def iter_forward(self, task_embedding):
        trj_size = self.get_trj_size(task_embedding=task_embedding)
        last_seq = make_inpt_seq(task_embedding=task_embedding, seq_size=trj_size)
        for it in range(self.iterations):
            last_seq = last_seq.detach()
            last_seq = self.get_trj(task_embedding=task_embedding, last_seq=last_seq)
        return last_seq

    def get_trj(self, task_embedding, last_seq):
        seq_transformer = self.get_seq_trans(task_embedding=task_embedding, last_seq=last_seq)
        last_seq = self.decode(seq_transformer=seq_transformer)
        return last_seq

    def get_seq_trans(self, task_embedding, last_seq):
        inpt_seq_size = self.get_seq_size(task_embedding=task_embedding)
        inpt_seq = right_stack_obj_trj(task_embedding, last_seq, inpt_seq_size)
        seq_transformer = self.transformer(inpt_seq)
        return seq_transformer

    def get_seq_size(self, task_embedding):
        return (task_embedding.size(0), self.seq_len, self.model_setup['transformer']['d_model'])

    def get_trj_size(self, task_embedding):
        return (task_embedding.size(0), self.seq_len - 1, self.model_setup['decoder']['d_output'])


    def get_critic_score(self, task_embedding, last_seq):
        seq_trans = self.get_seq_trans(task_embedding=task_embedding, last_seq=last_seq)
        critic_score = self.get_critic(seq_transformer=seq_trans)
        return critic_score

    def decode(self, seq_transformer):
        output_seq = seq_transformer[:,:-1,:self.model_setup['decoder']['d_output']] #N,l,d_out
        return output_seq

    def get_critic(self, seq_transformer):
        critic_score = self.critic(seq_transformer).squeeze() #N,1
        return critic_score

    def critic_loss_fct(self, critic_score, reward_label):
        return ((critic_score.reshape(-1) - reward_label.reshape(-1))**2).mean()

    def optimize(self, task_embedding):
        threshold = self.threshold
        epochs = self.epochs

        output_seq = self.iter_forward(task_embedding=task_embedding)
        opt_trj = output_seq.clone().detach()
        opt_trj.requires_grad = True
        trj_optimizer = torch.optim.AdamW([opt_trj], lr=1)
        critic_score = self.get_critic_score(task_embedding=task_embedding, last_seq=opt_trj)

        best_expected_success = torch.clone(critic_score).detach()
        best_expected_mean = best_expected_success.mean()

        improve_mask_opt = torch.ones_like(critic_score).type(torch.bool) * (best_expected_success < threshold)
        print(f'imporvement mask: {improve_mask_opt.sum()}')
        reward_label = torch.ones(task_embedding.size(0), device=task_embedding.device)
        best_trj = torch.clone(output_seq).detach()

        step = 0
        critic_loss = self.critic_loss_fct(critic_score=critic_score, reward_label=reward_label)
        while (step <= epochs):# and (best_expected_mean < threshold):
            critic_loss.backward()
            trj_optimizer.step()

            trj_optimizer.zero_grad()
            critic_score = self.get_critic_score(task_embedding=task_embedding, last_seq=opt_trj)

            critic_loss = self.critic_loss_fct(critic_score=critic_score, reward_label=reward_label)
            
            improve_mask = (critic_score > best_expected_success)*improve_mask_opt

            best_expected_success[improve_mask]= critic_score.clone()[improve_mask].detach()
            best_trj[improve_mask] = opt_trj.clone()[improve_mask].detach()
            improve_mask_opt = improve_mask_opt * (best_expected_success < threshold)
            best_expected_mean = best_expected_success.mean()
            self.writer.write_tboard_scalar({'in optimisation ':best_expected_mean}, train=False, step=step)
            step += 1

        return best_trj
