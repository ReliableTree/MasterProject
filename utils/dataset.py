from cProfile import label
from tkinter import Label
import torch
import numpy as np
from utils.utils import add_data_to_seq
from utils.utils import get_her_mask, get_input_mask
from utils.simulation import HERSimulation

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        

    def add_data(self, trajectories, obsv, success):
        self.s_trajectories = torch.cat((self.s_trajectories , trajectories[success]), dim=0) 
        self.s_obsv = torch.cat((self.s_obsv , obsv[success]), dim=0) 
        self.success = torch.cat((self.success , success[success]), dim=0) 

        self.f_trajectories = torch.cat((self.f_trajectories , trajectories[~success]), dim=0) 
        self.f_obsv = torch.cat((self.f_obsv , obsv[~success]), dim=0) 
        self.fail = torch.cat((self.fail , success[~success]), dim=0) 

        self.set_len()

    def set_len(self):
        self.s_len = len(self.success)
        self.f_len = len(self.fail)
        self.len = max(self.s_len, self.f_len) 


    def __len__(self):
        return self.len

    def num_ele(self):
        return self.s_len + self.f_len

    def __getitem__(self, index):
        if self.f_len == 0:
            return (self.s_obsv[index%self.s_len], self.s_trajectories[index%self.s_len], self.success[index%self.s_len]),\
                False
        else:
            return (self.s_obsv[index%self.s_len], self.s_trajectories[index%self.s_len], self.success[index%self.s_len]),\
                (self.f_obsv[index%self.f_len], self.f_trajectories[index%self.f_len], self.fail[index%self.f_len])


class HERDataset(Dataset):
    def __init__(self, path, device, num_ele=-0):
        super().__init__()
        self.load_data(path=path, device=device, num_ele=num_ele)

    def load_data(self, path, device, num_ele):
        data = np.load(path, allow_pickle=True)
        max_len = 0
        for sample in data['obs']:
            if len(sample) > max_len:
                max_len = len(sample)

        trajectories = None
        traj = None

        goals = None
        des_goal = None
        for sample in data['obs']:
            for step in sample:
                obsv = torch.tensor(step['observation'])
                traj = add_data_to_seq(obsv[mask], traj, length=len(obsv[mask]) + 4)
                traj[:,-4:] = 0
                if des_goal is None:
                    des_goal = torch.cat((obsv[inpt_mask], torch.tensor(step['desired_goal'])))

            trajectories = add_data_to_seq(traj, trajectories, length=max_len)
            traj = None

            goals = add_data_to_seq(des_goal, goals)
            des_goal = None

        for i, sample in enumerate(data['acs']):
            for j, step in enumerate(sample):
                action = torch.tensor(step)
                trajectories[i,j+1,-4:] = action

        #trajectories = batch x seq_len x [obsv[25], acts[26:29]]
        
        success = torch.ones(trajectories.size(0), dtype=torch.bool)

        self.s_trajectories = trajectories[success==1].to(device)[-num_ele:]
        self.s_obsv = goals[success==1].to(device)[-num_ele:]
        self.success = success[success==1].to(device)[-num_ele:]

        self.f_trajectories = trajectories[success==0].to(device)
        self.f_obsv = goals[success==0].to(device)
        self.fail = success[success==0].to(device)
        self.set_len()
        


class ToyDataset(Dataset):
    def __init__(self, path, device, num_ele=0) -> None:
        super().__init__()
        self.load_data(path=path, device=device, num_ele=num_ele)
        self.set_len()

    def load_data(self, path, device, num_ele):
        path_data = path + 'inpt'
        path_label = path + 'label'
        obsv = torch.load(path_data).to(torch.float32).squeeze()
        trajectories = torch.load(path_label).to(torch.float32)
        success = torch.ones(trajectories.size(0), dtype=torch.bool)

        self.s_trajectories = trajectories[success==1].to(device)[-num_ele:]
        self.s_obsv = obsv[success==1].to(device)[-num_ele:]
        self.success = success[success==1].to(device)[-num_ele:]

        self.f_trajectories = trajectories[success==0].to(device)
        self.f_obsv = obsv[success==0].to(device)
        self.fail = success[success==0].to(device)
        self.set_len()

class HERDatasetBaseline(Dataset):
    def __init__(self, path, device, num_ele=-0):
        super().__init__()
        self.load_data(path=path, device=device, num_ele=num_ele)

    def load_data(self, path, device, num_ele):
        hs = HERSimulation()
        np_data = np.load(path, allow_pickle=True)
        max_len = 0
        for sample in np_data["obs"]:
            if len(sample) > max_len:
                max_len = len(sample)

        print(f'max length: {max_len}')

        label = None
        labels = None

        data = None
        datas = None

        for sample in np_data['obs']:
            for step in sample:
                obsv = torch.tensor(hs.make_obsv(step), dtype=torch.float)
                label = add_data_to_seq(obsv, label, length=len(obsv))
            labels = add_data_to_seq(label, labels, length=max_len)
            label = None
                
        for i, sample in enumerate(np_data['acs']):
            for j, step in enumerate(sample):
                step[-1] = step[-1]*10
                action = torch.tensor(step, dtype=torch.float)
                data = add_data_to_seq(action, data, len(action))
            datas = add_data_to_seq(data, datas, length=max_len)
            data = None
        self.label = datas[num_ele:2*num_ele].to(device)
        self.data = labels[num_ele:2*num_ele].to(device)
        print(f'self.label: {self.label.shape}')
        print(self.label[...,:])
        print(f'self.data: {self.data.shape}')
        print(self.data[...,:])

    def add_data(self, data, label):
        print(f'data: {data.shape}')
        print(f'label: {label.shape}')
        print(f'self.data: {self.data.shape}')
        print(f'self.label: {self.label.shape}')

        if data.size(1) == 1:
            data = data.repeat([1,self.data.size(1), 1])
        if len(data.shape) == 2:
            data = data.unsqueeze(1).repeat([1,self.data.size(1), 1])
        self.data = torch.cat((self.data.to('cuda'), data.to('cuda')), dim=0)
        self.label = torch.cat((self.label.to('cuda'), label.to('cuda')), dim=0)

    def __getitem__(self, index):
            'Generates one sample of data'
            return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
        