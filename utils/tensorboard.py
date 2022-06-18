import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import math


class TBoardGraphs():
    def __init__(self, logname= None, data_path = None):
        if logname is not None:
            self.logdir              = os.path.join(data_path, "gboard/" + logname + "/")
            self.__tboard_train      = tf.summary.create_file_writer(self.logdir + "train/")
            self.__tboard_validation = tf.summary.create_file_writer(self.logdir + "validate/")
        self.subplots = False

    def set_subplots(self, dims):
        num_rows = math.ceil((dims+1) / 3) 
        self.fig, self.ax = plt.subplots(num_rows,3)
        self.subplots = True

    def finishFigure(self, fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def addTrainScalar(self, name, value, stepid):
        with self.__tboard_train.as_default():
            tf.summary.scalar(name, value.detach().cpu().numpy(), step=stepid)

    def addValidationScalar(self, name, value, stepid):
        with self.__tboard_validation.as_default():
            tf.summary.scalar(name, value.detach().cpu().numpy(), step=stepid)

    def plotTrajectory(self, y_true, y_pred, opt_y_pred=None,inpt = None, stepid= None, name = "Trajectory", save = False, \
            name_plot = None, path=None, tol_neg = None, tol_pos=None):
        num_dims = len(y_true[0])
        if not self.subplots:
            self.set_subplots(dims=num_dims)
        trj_len      = y_true.shape[0]
        
        fig, ax = self.fig, self.ax
        if tol_neg is not None:
            neg_inpt = (y_true + tol_neg[None,:]).cpu().numpy()
            pos_inpt = (y_true + tol_pos[None,:]).cpu().numpy()
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        inpt = inpt.cpu().numpy()
        if opt_y_pred is not None:
            opt_y_pred = opt_y_pred.cpu().numpy
        for sp in range(num_dims):
            idx = sp // 3
            idy = sp  % 3
            ax[idx,idy].clear()

            # GT Trajectory:
            if tol_neg is not None:
                ax[idx,idy].plot(range(y_pred.shape[0]), neg_inpt[:,sp], alpha=0.75, color='orangered')
                ax[idx,idy].plot(range(y_pred.shape[0]), pos_inpt[:,sp], alpha=0.75, color='orangered')
            ax[idx,idy].plot(range(trj_len), y_true[:,sp],   alpha=1.0, color='forestgreen')            
            ax[idx,idy].plot(range(trj_len), y_pred[:,sp], alpha=0.75, color='mediumslateblue')
            if opt_y_pred is not None:
                ax[idx,idy].plot(range(y_pred.shape[0]), opt_y_pred[:,sp], alpha=0.75, color='lightseagreen')
                diff_vec = opt_y_pred - y_pred
                ax[idx,idy].plot(range(y_pred.shape[0]), diff_vec[:,sp], alpha=0.75, color='pink')

        if inpt is not None:
            ax[-1,-1].clear()
            ax[-1,-1].plot(range(inpt.shape[-1]), inpt.squeeze(),   alpha=1.0, color='forestgreen')     

        result = np.expand_dims(self.finishFigure(fig), 0)
        if save:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + name_plot + '.png')
        else:
            with self.__tboard_validation.as_default():
                tf.summary.image(name, data=result, step=stepid)

    def write_tboard_scalar(self, debug_dict, train, step, prefix = ''):
        for para, value in debug_dict.items():
            name_para = prefix + para
            if train:
                self.addTrainScalar(name_para, value, step)
            else:
                self.addValidationScalar(name_para, value, step)