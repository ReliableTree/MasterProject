import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, device = 'cpu', num_ele = 0):
        super().__init__()
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

    def add_data(self, trajectories, obsv, success):
        self.s_trajectories = torch.cat((self.s_trajectories , trajectories[success==1]), dim=0) 
        self.s_obsv = torch.cat((self.s_obsv , obsv[success==1]), dim=0) 
        self.success = torch.cat((self.success , success[success==1]), dim=0) 

        self.f_trajectories = torch.cat((self.f_trajectories , trajectories[success==0]), dim=0) 
        self.f_obsv = torch.cat((self.f_obsv , obsv[success==0]), dim=0) 
        self.fail = torch.cat((self.fail , success[success==0]), dim=0) 

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