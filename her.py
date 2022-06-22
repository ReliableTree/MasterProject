from utils.dataset import HERDataset
from model_src.model_setup import model_setup
from utils.tensorboard import TBoardGraphs
import hashids
import time
from model_src.network import Network
import torch
from utils.simulation import HERSimulation
from prettytable import PrettyTable
import copy

torch.manual_seed(0)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__ == '__main__':
    hid = hashids.Hashids()
    logname = hid.encode(int(time.time() * 1000000))
    data_path = '/home/hendrik/Documents/master_project/LokalData/'
    tboard = TBoardGraphs(logname=logname, data_path=data_path)

    train_data = HERDataset(path='/home/hendrik/Documents/master_project/Code/MasterProject/data_fetch_random_100.npz', device='cuda', num_ele=-0)
    val_data = copy.deepcopy(train_data)

    model_setup['seq_len'] = 53
    model_setup['decoder']['d_output'] = 4
    #model_setup['transformer']['d_output']=4
    model_setup['transformer']['d_inpt'] = 13
    new_model_setup = model_setup
    datasets = {'train':train_data, 'val':val_data}
    model_lr = 5e-4
    critic_lr = 6e-5
    batch_size = 32
    simulation = HERSimulation()
    network = Network(model_setup=model_setup, data_sets=datasets, model_lr=model_lr, critic_lr=critic_lr, batchsize=16, tboard=tboard, simulation=simulation, batch_size=batch_size, save_path=data_path, logname = logname)
    count_parameters(network)

    network.train(epochs=100000)