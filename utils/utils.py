import torch
import numpy as np
from torch import Tensor
import scipy.stats

def stack_trj(task_embedding, sequence):
    #inpt = N,L+1,D
    seq_size = [task_embedding.size(0), sequence.size(1), sequence.size(-1)+task_embedding.size(-1)]
    result = make_inpt_seq(task_embedding=task_embedding, seq_size=seq_size)
    embed_size = task_embedding.size(-1)
    result[:,:,embed_size:embed_size+sequence.size(-1)] = sequence
    return result

def make_inpt_seq(task_embedding, seq_size):
    rep_task_embedding = task_embedding.unsqueeze(1).repeat([1,seq_size[1],1])
    embed_size = task_embedding.size(-1)
    result = torch.zeros(size=seq_size, device=task_embedding.device)
    result[:,:,:embed_size] = rep_task_embedding
    return result

def pars_obsv(result):
    obsv = result['observation']
    goal = result['desired_goal']
    parsed = {}
    parsed['hand_pos'] = obsv[:3]
    parsed['puck_pos'] = obsv[3:6]
    parsed['puck_rot'] = obsv[9:13]
    parsed['goal_pos'] = goal
    return parsed

def make_obsv(result):
    obs_dict = pars_obsv(result)
    return concat_obsv(obs_dict=obs_dict)

def concat_obsv(obs_dict):
    return(np.concatenate((obs_dict['hand_pos'], obs_dict['puck_pos'], obs_dict['puck_rot'], obs_dict['goal_pos']), axis=0))

def make_obsv_seq(obsvs, device='cuda'):
    result = None
    for obsv in obsvs:
        result = append_obsv_seq(obsv, result, device)
    return result

def append_obsv_seq(obsv, seq=None, device = 'cuda'):
    result = make_obsv(obsv)
    result = torch.tensor(result, dtype=torch.float, device=device).unsqueeze(0)
    if seq is None:
        seq = result
    else:
        seq = torch.cat((seq, result), dim=0)
    return seq



def add_data_to_seq(data, seq=None, length = 0):
    inpt = pad_to_len(data, length).unsqueeze(0)
    if (seq is None) or (len(seq) == 0):
        return inpt
    else:
        return torch.cat((seq,  inpt), dim=0)

def pad_to_len(ten, length):
    if length == 0:
        return ten
    elif len(ten) == 0:
        return torch.zeros([length])
    else:
        num_dim = len(ten.shape)
        rep_arr = [1]*(num_dim-1)
        result = torch.cat((ten, (ten[-1].unsqueeze(0).repeat(length - len(ten), *rep_arr))))
        return result

def calc_MSE(inpt, label):
    if torch.numel(inpt) == 0:
        return torch.zeros(1, device=inpt.device).mean()
    else:
        return ((inpt.reshape(-1).type(torch.float) - label.reshape(-1).type(torch.float))**2).mean()

def check_outpt(label, outpt, tol_neg, tol_pos, window = 0):
    if window > 0:
        tol_neg, tol_pos, inpt= make_sliding_tol(label=inpt, neg_tol=tol_neg, pos_tol=tol_pos, window=window)

    diff = outpt - label
    

    if window > 0:
        neg_acc = diff > tol_neg
        pos_acc = diff < tol_pos
    else:
        neg_acc = diff > tol_neg[None,None,:]
        pos_acc = diff < tol_pos[None,None,:]
    acc = neg_acc*pos_acc
    acc = acc.reshape(diff.size(0), -1)
    return torch.all(acc, dim=1)

def make_sliding_tol(label, neg_tol, pos_tol, window=9):
    tols_pos, tols_neg = [], []
    for dim in range(label.size(-1)):
        tol_pos, tol_neg = make_sliding_tol_dim(label=label[:,:,dim], window=window)
        tols_pos.append(tol_pos.unsqueeze(-1))
        tols_neg.append(tol_neg.unsqueeze(-1))
    sliding_tol_pos, sliding_tol_neg = torch.cat(tuple(tols_pos), dim=-1), torch.cat(tuple(tols_neg), dim=-1)
    neg_inpt = (sliding_tol_neg[0] + neg_tol[None,:])
    pos_inpt = (sliding_tol_pos[0] + pos_tol[None,:])
    inpt = label[:, int(window/2):-(int(window/2) + 1)]
    result = pos_inpt, neg_inpt, inpt
    return result

def make_sliding_tol_dim(label, window = 9):
    batch_size = label.size(0)
    batch_counter = torch.arange(batch_size)
    counter = torch.arange(label.size(-1) - window) + int(window/2)
    window_counter = torch.arange(window) - int(window/2)
    s_ind = counter.repeat([batch_size,window,1]).transpose(-1,-2)
    f_ind = (counter[:,None] + window_counter[None,:]).repeat([batch_size, 1,1])
    batch_ind = batch_counter.reshape(-1,1,1).repeat([1,f_ind.size(-2), f_ind.size(-1)])
    ind = tuple((batch_ind, f_ind, s_ind))
    label_repeated = label.unsqueeze(-1).repeat([1,1,label.size(-1)])
    label_ind = label_repeated[ind]
    result = label_ind.max(dim=-1)[0], label_ind.min(dim=-1)[0]
    return result

def get_her_mask(device):
    mask = torch.zeros([25], dtype=torch.bool, device=device)
    #mask[:6] = 1
    #mask[9] = 1
    return mask

def get_input_mask(device):
    mask = torch.zeros([25], dtype=torch.bool, device=device)
    mask[:6] = 1
    return mask

class ConvergenceDetector():
    def __init__(self, setup, set_meta) -> None:
        self.loss_history = None
        self.mean_loss_history = []
        self.setup = setup
        self.set_meta = set_meta
        self.num_convs = 0

    def add_loss(self, loss, ):
        if self.loss_history is None:
            self.loss_history = loss
        else:
            self.loss_history = torch.cat((self.loss_history, loss))
        conv_detected = self.detect_convergence()
        if conv_detected:
            if self.num_convs == 0:
                self.setup()
                self.set_meta()
                self.reset_history()
            self.num_convs += 1
        return self.num_convs, conv_detected

    def reset_history(self):
        self.loss_history = None
        self.mean_loss_history = []

    def torch_compute_confidence_interval(self, data: Tensor,
                                           confidence: float = 0.95
                                           ) -> Tensor:
        """
        Computes the confidence interval for a given survey of a data set.
        """
        n = len(data)
        mean: Tensor = data.mean()
        # se: Tensor = scipy.stats.sem(data)  # compute standard error
        # se, mean: Tensor = torch.std_mean(data, unbiased=True)  # compute standard error
        se: Tensor = data.std(unbiased=True) / (n**0.5)
        t_p: float = float(scipy.stats.t.ppf((1 + confidence) / 2., n - 1))
        ci = t_p * se
        return mean, ci

    def detect_convergence(self):
        if len(self.loss_history) > 3000:
            resent_interval = self.torch_compute_confidence_interval(data=self.loss_history[-300:], confidence=0.02)
            whole_interval = self.torch_compute_confidence_interval(data=self.loss_history[-3000:], confidence=0.02)
            return whole_interval[0] - whole_interval[1] > resent_interval[0]
        else:
            return False