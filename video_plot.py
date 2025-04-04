from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import glob
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import vmap
import numpy as np

import json
from scipy.io import savemat, loadmat
import time

name = "output"
experiment_name = "membrane_5inputs_force_time_damping_ic0hard_icv0_t5.0_MLP_rff1.0_12000epochs_2"
current_file = os.path.abspath(__file__)
output_dir = os.path.join(os.path.dirname(current_file), name)
output_dir = os.path.join(output_dir, experiment_name)

model_dir = os.path.join(output_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'model_600.pt')

video_output = False



def generate_video(folder):
    os.chdir(folder)
    subprocess.call([
        'ffmpeg', '-framerate', '10', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    for file_name in glob.glob("file*.png"):
        os.remove(file_name)

def load_params(path):
    global u_min, u_max, f_min, f_max, x_min, x_max, y_min, y_max, t_f, delta_u, delta_x, delta_f, delta_y
    with open(path, "r") as fp:
        params = json.load(fp)["additionalData"]
        u_min = params["u_min"]
        u_max = params["u_max"]
        x_min = params["x_min"]
        x_max = params["x_max"]
        y_min = params["y_min"]
        y_max = params["y_max"]
        f_min = params["f_min"]
        f_max = params["f_max"]
        t_f = params["t_f"]
        delta_u = u_max - u_min
        delta_x = x_max - x_min
        delta_f = f_max - f_min
        delta_y = y_max - y_min
    return

load_params(os.path.join(output_dir, "params.json"))


def hard_constraint(x, y_out):
    X = x[:, 0].reshape(-1, 1)
    Y = x[:, 1].reshape(-1, 1)
    tau = x[:, -1].reshape(-1, 1)

    x = X*delta_x + x_min
    y = Y*delta_y + y_min
    t = tau*t_f
    u = y_out*delta_u + u_min

    u = u*(x-x_max)*(x-x_min)*(y-y_max)*(y-y_min)*t

    U = (u-u_min)/delta_u
    return U



def compose_input(x, y, tt, meshgrid=True):
    # Compose input from meshgrid of x and y, repeat as the size of tt and attach in order x_f1, y_f1, x_f2, y_f2, tt
    X_ = (x - x_min) / delta_x
    Y_ = (y - y_min) / delta_y
    X_ = np.tile(X_, (tt.shape[0], 1)).reshape(-1, 1)
    Y_ = np.tile(Y_, (tt.shape[0], 1)).reshape(-1, 1)


    if meshgrid:
        T_ = np.repeat(tt / t_f, x.shape[0]*y.shape[0]).reshape(-1, 1)
    else:
        T_ = np.repeat(tt / t_f, x.shape[0]).reshape(-1, 1)
    
    X = np.hstack((X_, Y_, T_))
    X = torch.Tensor(X).to(torch.device("cpu")).requires_grad_()
    return X

#model = TimeFourierMLP([3] + [308]*8 + [1], nn.SiLU, sigma = 10.0, encoded_size=154, hard_constraint_fn = hard_constraint, p_dropout=0.0)
#model.load_state_dict(torch.load(model_path))
model = torch.load(model_path, map_location=torch.device("cpu"))



model.train(False)
#fig, axes = plt.subplots(1, 3, figsize=(15, 5))

tt = np.linspace(0, t_f, num=1001, endpoint=True)
x = np.linspace(x_min, x_max, num=101, endpoint=True).reshape(-1, 1)
y = np.linspace(y_min, y_max, num=101, endpoint=True).reshape(-1, 1)
x, y = np.meshgrid(x, y)


""" ttrue = exact()

xx, yy, ttt = np.meshgrid(x,y,tt)
X = np.vstack((np.ravel(xx), np.ravel(yy), np.ravel(ttt))).T
la = len(np.unique(X[:, 0]))
le = len(np.unique(X[:, -1]))

true = ttrue.reshape((le, la), order="F")  """ 

X = compose_input(x, y, tt)
preds = np.zeros(len(x)*len(y)*len(tt))
batch = 10000
for i in range(0, len(X), batch):
    elem = X[i:i+batch]
    pred = model(elem)
    pred = pred.cpu().detach().numpy()
    pred = pred*delta_u + u_min
    preds[i:i+batch] = pred.reshape(-1)

preds = preds.reshape(len(tt), len(x), len(y))



counter = 0
if video_output:
    for index, t in enumerate(tt):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x, y, preds[index, :, :], cmap=cm.coolwarm, linewidth=1, antialiased=False)
        #plt.plot(x, true[counter])
        #ax = plt.gca()
        ax.set_zlim([-0.2, 0.2])
        ax.legend(["t = {:.2f}".format(t)])
        #plt.show()
        plt.savefig(output_dir + "/file%02d.png" % counter)
        plt.close()
        counter += 1
    generate_video(output_dir)
    
preds = np.array(preds)
mdic = {"pinn_data": preds, "X_pinn": x, "Y_pinn": y}
savemat(output_dir+"/data_all.mat", mdic)


node_path = "nodes.mat"
if os.path.exists(node_path):
    start_time = time.time()
    nodes = loadmat(node_path)["nodes"]
    x = nodes[0, :].reshape(-1, 1)
    y = nodes[1, :].reshape(-1, 1)
    X = compose_input(x, y, tt, meshgrid=False)
    preds = np.zeros(len(x)*len(tt))
    batch = 10000
    batch = min(batch, len(X))
    for i in range(0, len(X), batch):
        if i+batch > len(X):
            elem = X[i:]
        else:
            elem = X[i:i+batch]
        pred = model(elem)
        pred = pred.cpu().detach().numpy()
        pred = pred*delta_u + u_min
        if i+batch > len(X):
            preds[i:] = pred.reshape(-1)
        else:
            preds[i:i+batch] = pred.reshape(-1)
    end_time = time.time()
    print("Time elapsed for prediction: ", end_time - start_time)    

    preds = preds.reshape(len(tt), len(x))
    preds = np.array(preds)
    mdic = {"pinn_data": preds, "X_pinn": x, "Y_pinn": y}
    savemat(output_dir+"/data.mat", mdic)






'''
# Plot Theta Predicted
im1 = axes[0].imshow(pred, cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])#, vmin=true.min(), vmax = true.max())
axes[0].set_title(f'Predicted')
axes[0].set_xlabel('X')
axes[0].set_ylabel('T')
plt.colorbar(im1, ax=axes[0])

# Plot Theta True
im2 = axes[1].imshow(true, cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])
axes[1].set_title(f'True')
axes[1].set_xlabel('X')
axes[1].set_ylabel('T')
plt.colorbar(im2, ax=axes[1])

# Plot Difference
im3 = axes[2].imshow(np.abs(pred-true), cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])
axes[2].set_title(f'Difference')
axes[2].set_xlabel('X')
axes[2].set_ylabel('T')
plt.colorbar(im3, ax=axes[2])

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/plot_corda_semplice.png')

plt.show()

'''

