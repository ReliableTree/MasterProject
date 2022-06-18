from pathlib import Path
import sys
parent_path = str(Path(__file__).parent.absolute())
parent_path += '/../../'
sys.path.append(parent_path)

import torch
from torch.utils.data import DataLoader
from yaml import load
from MasterProject.model_src.model import Model
from MasterProject.utils.utils import add_data_to_seq, calc_MSE
class Network(torch.nn.Module):
    def __init__(self, model_setup, data_sets, model_lr, critic_lr, batchsize, tboard, simulation, batch_size):
        super().__init__()
        self.tboard = tboard
        self.model_setup = model_setup

        self.train_ds = data_sets['train']
        self.val_ds = data_sets['val']
        self.train_loader = DataLoader(self.train_ds, batch_size=batchsize, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=batchsize, shuffle=True)        

        self.model_lr = model_lr
        self.critic_lr = critic_lr
        self.threshold = 0.95
        self.init_model()
        self.global_step = 0

        self.simulation = simulation
        self.batch_size = batch_size

        
    def init_model(self):
        self.model = Model(model_setup=self.model_setup, opt_lr=1, writer=self.tboard, iterations=2, threshold=self.threshold).to('cuda')

        for succ, fail in self.train_loader:
            obsv, traj, success = succ
            print(f'obsv:{obsv.shape}')
            self.model.forward(obsv)
            break
        self.optimizers = []
        self.optimizers.append(torch.optim.AdamW(self.model.transformer.parameters(), lr=self.model_lr, weight_decay=1e-2))
        self.optimizers.append(torch.optim.AdamW(self.model.critic_transformer.parameters(), lr=self.model_lr, weight_decay=1e-2))
    
    
    def train(self, epochs):
        init_train = True
        max_ll = 0
        print(f'len succ: {len(self.train_ds.success)}')
        print(f'len fail: {len(self.train_ds.fail)}')

        for epoch in range(epochs):
            self.model.train()
            self.model.set_mode(0)
            loss = 1
            critic_loss = 2
            trj_loss = 1
            learning_loops = 0
            while (trj_loss > 0.005)or (critic_acc < 1):
                debug_dict = self.run_epoch(train=True)
                critic_acc = debug_dict['main critic acc']
                trj_loss = debug_dict['main trj loss']
                learning_loops += 1

                '''if (learning_loops > max_ll) and (not init_train):
                    learning_loops = 0
                    self.init_model()
                    init_train = True
                    print('init model')'''
                            
            if init_train:
                max_ll = 4*learning_loops
            init_train = False
            self.model.eval()
            self.validate()
            if (epoch+1) % 200 == 0:
                self.model.set_mode(1)
                self.simulate(num_envs=20, prefix='optimisation ', add_data=True, train=True)
                self.model.set_mode(0)
                self.simulate(num_envs=20, prefix='iteration ', add_data=True, train=True)
                if (epoch+1)%2000 == 0:
                    self.model.set_mode(1)
                    self.simulate(num_envs=200, prefix='optimisation ', add_data=True, train=False)
                    self.model.set_mode(0)
                    self.simulate(num_envs=200, prefix='iteration ', add_data=True, train=False)

            print(f'epoch: {epoch}')

    def run_epoch(self, train):
        losses = None
        trj_losses = None
        critic_losses = None
        critic_acc = None
        learn_critic = False
        if train:
            loader = self.train_loader
            dataset = self.train_ds
        else:
            loader = self.val_loader
            dataset = self.val_ds

        for succ, fail in loader:
            if train:
                self.global_step += 1

            learn_critic = dataset.f_len > 0


            if learn_critic:
                loss, debug_dict = self.step(fail, train=train, learn_critic=learn_critic)
                self.tboard.write_tboard_scalar(debug_dict=debug_dict, train=train, step=self.global_step, prefix='fail ')
                losses = add_data_to_seq(loss.reshape(-1), losses)
                critic_acc = add_data_to_seq(debug_dict['acc'].reshape(-1), critic_acc)


            loss, debug_dict = self.step(succ, train=train, learn_critic=learn_critic)
            self.tboard.write_tboard_scalar(debug_dict=debug_dict, train=train, step=self.global_step, prefix='succ ')
            trj_losses = add_data_to_seq(debug_dict['trj loss'].reshape(-1), trj_losses)
            critic_losses = add_data_to_seq(debug_dict['critic loss'].reshape(-1), critic_losses)
            critic_acc = add_data_to_seq(debug_dict['acc'].reshape(-1), critic_acc)

            losses = add_data_to_seq(loss.reshape(-1), losses)
        
        if not learn_critic:
            critic_losses = torch.zeros_like(critic_losses)
            critic_acc = torch.ones_like(critic_acc)

        debug_dict = {
            'main loss':losses.mean().detach(),
            'main critic loss':critic_losses.mean().detach(),
            'main trj loss':trj_losses.mean().detach(),
            'main critic acc': critic_acc.mean().detach()
        }

        self.tboard.write_tboard_scalar(debug_dict=debug_dict, train=train, step=self.global_step)
        return debug_dict

    def backprop(self, loss):
        for opt in self.optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in self.optimizers:
            opt.step()
                    
    def step(self, inpt, train, learn_critic):
        obsv, label, succ = inpt
        output_seq = self.model.forward(task_embedding=obsv)
        critic_scores = self.model.get_critic_score(task_embedding=obsv, last_seq=output_seq, detach=True)
        loss, debug_dict = self.calculate_loss(result=(output_seq, critic_scores), label=label, succ=succ, learn_critic=learn_critic)
        if train:
            self.backprop(loss)
        return loss, debug_dict

    def calculate_loss(self, result, label, succ, learn_critic):
        output_seq, critic_score = result
        trj_loss = calc_MSE(output_seq[succ], label[succ])
        critic_loss = calc_MSE(critic_score, succ)
        pred_stats = self.get_prediction_stats(label=succ, pred=critic_score)
        debug_dict = {
            'trj loss':trj_loss.clone().detach(),
            'critic loss':critic_loss.clone().detach(),
            'critic acc':pred_stats['acc']
        }
        debug_dict.update(self.get_prediction_stats(label=succ, pred=critic_score))
        loss = trj_loss
        if learn_critic:
            loss = loss + critic_loss
        return loss, debug_dict
        
    def get_prediction_stats(self, label, pred):
        label = label.type(torch.bool)
        expected_success = (pred > 0.5).type(torch.bool)
        expected_fail = ~ expected_success
        fail = ~label

        tp = (expected_success == label)[label==1].type(torch.float).mean()
        if label.sum() == 0:
            tp = torch.tensor(0)
        fp = (expected_success == fail)[fail==1].type(torch.float).mean()
        tn = (expected_fail == fail)[fail==1].type(torch.float).mean()
        fn = (expected_fail == label)[label==1].type(torch.float).mean()
        acc = (expected_success == label).type(torch.float).mean()
        debug_dict = {
            'expedcet success': expected_success.type(torch.float).mean(),
            'tp' : tp,
            'fp' : fp,
            'tn' : tn,
            'fn' : fn,
            'acc' : acc
                    }
        return debug_dict

    def validate(self):
        with torch.no_grad():
            self.run_epoch(train=False)

    def simulate(self, num_envs, prefix, add_data = False, device='cuda', train = True):
        gt_policy_success = False
        while not gt_policy_success:
            seeds = self.simulation.get_seeds(n=num_envs)
            result = self.simulation.simulate(policy = self.model, seeds=seeds, device=device)
            if result is not False:
                i_result, HER_result = result
                trajectories, inpt_obs, labels, success, critic_scores = i_result
                HER_trajectories, HER_inpt_obs, HER_labels, HER_success, HER_critic_scores = HER_result
                gt_policy_success = True

        debug_dict = self.get_prediction_stats(label=success, pred=critic_scores)

        debug_dict.update({
            'success rate' : success.type(torch.float).mean(),
                    })
        self.tboard.write_tboard_scalar(debug_dict=debug_dict, train=train, step=self.global_step, prefix=prefix)

        if add_data:
            num_exp = 5
            self.train_ds.add_data(trajectories=labels[:num_exp], obsv=inpt_obs[:num_exp], success=success[:num_exp].type(torch.bool))
            self.val_ds.add_data(trajectories=labels[num_exp:2*num_exp], obsv=inpt_obs[num_exp:2*num_exp], success=success[num_exp:2*num_exp].type(torch.bool))
            
            self.train_ds.add_data(trajectories=HER_labels[:num_exp], obsv=HER_inpt_obs[:num_exp], success=HER_success[:num_exp].type(torch.bool))
            self.val_ds.add_data(trajectories=HER_labels[num_exp:2*num_exp], obsv=HER_inpt_obs[num_exp:2*num_exp], success=HER_success[num_exp:2*num_exp].type(torch.bool))
            print('added data')
            print(f'data len: {self.train_ds.num_ele()}')
            print(f'len succ: {len(self.train_ds.success)}')
            print(f'len fail: {len(self.train_ds.fail)}')


        self.tboard.plotTrajectory(y_true = labels[0], y_pred=trajectories[0], opt_y_pred=None,inpt = inpt_obs[0], stepid= self.global_step, name = "Trajectory", save = False, \
            name_plot = None, path=None, tol_neg = self.simulation.neg_tol, tol_pos=self.simulation.pos_tol)
