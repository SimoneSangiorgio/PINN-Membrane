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
from pinns_v2.model import TimeFourierMLP
import json
from scipy.io import savemat

name = "output"
experiment_name = "membrane_6inputs_nostiffness_force_damping_ic0hard_icv0_causality_t10.0_timerff10.0_2000epochs"
current_file = os.path.abspath(__file__)
output_dir = os.path.join(os.path.dirname(current_file), name)
output_dir = os.path.join(output_dir, experiment_name)

model_dir = os.path.join(output_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'model_10k.pt')

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

    u = u*(x-x_max)*(x-x_min)*(y-y_max)*(y-y_min)

    U = (u-u_min)/delta_u
    return U

def compose_input(x, y, x_f, y_f, h, t):
    X = (x-x_min)/delta_x
    Y = (y-y_min)/delta_y
    tau = np.ones_like(x)*(t/t_f)
    X_f = np.ones_like(x)*((x_f-x_min)/delta_x)
    Y_f = np.ones_like(x)*((y_f-y_min)/delta_y)
    H = np.ones_like(x)*((h-f_min)/delta_f)
    x_in = np.vstack((np.ravel(X), np.ravel(Y), np.ravel(X_f), np.ravel(Y_f), np.ravel(H), np.ravel(tau))).T
    x_in = torch.Tensor(x_in).to(torch.device("cpu")).requires_grad_()
    return x_in

#model = TimeFourierMLP([3] + [308]*8 + [1], nn.SiLU, sigma = 10.0, encoded_size=154, hard_constraint_fn = hard_constraint, p_dropout=0.0)
#model.load_state_dict(torch.load(model_path))
model = torch.load(model_path, map_location=torch.device("cpu"))

x = torch.randn(1, 6).to(torch.device("cpu"))


model.train(False)
#fig, axes = plt.subplots(1, 3, figsize=(15, 5))

tt = np.linspace(0, t_f, num=1001, endpoint=True)
x = np.linspace(x_min, x_max, num=101, endpoint=True).reshape(-1, 1)
y = np.linspace(y_min, y_max, num=101, endpoint=True).reshape(-1, 1)
x, y = np.meshgrid(x, y)
x_f = 0.5
y_f = 0.5
h = -3.0


""" ttrue = exact()

xx, yy, ttt = np.meshgrid(x,y,tt)
X = np.vstack((np.ravel(xx), np.ravel(yy), np.ravel(ttt))).T
la = len(np.unique(X[:, 0]))
le = len(np.unique(X[:, -1]))

true = ttrue.reshape((le, la), order="F")  """ 

preds = []
counter = 0
for t in tt:     
    X = compose_input(x, y, x_f, y_f, h, t)
    pred = model(X)
    pred = pred.cpu().detach().numpy()
    pred = pred*delta_u + u_min
    pred = pred.reshape(len(x), len(y))
    preds.append(pred)

    if video_output:
        #plt.cla()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x, y, pred, cmap=cm.coolwarm, linewidth=1, antialiased=False)
        #plt.plot(x, true[counter])
        #ax = plt.gca()
        ax.set_zlim([-0.2, 0.2])
        ax.legend(["t = {:.2f}".format(t)])
        #plt.show()
        plt.savefig(output_dir + "/file%02d.png" % counter)
        plt.close()
        counter += 1

if video_output:
    generate_video(output_dir)
preds = np.array(preds)
print(preds.shape)
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
