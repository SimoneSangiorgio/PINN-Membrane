from pinns_v2.model import MLP, ModifiedMLP
from pinns_v2.components import ComponentManager, ResidualComponent, ICComponent, SupervisedComponent
from pinns_v2.rff import GaussianEncoding 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pinns_v2.train import train
from pinns_v2.gradient import _jacobian, _hessian
from pinns_v2.dataset import DomainDataset, ICDataset, DomainSupervisedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 5
num_inputs = 3 #x, y, t

u_min = -0.21
u_max = 0.0
x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0
t_f = 10
f_min = -3.0
f_max = 0.0
delta_u = u_max - u_min
delta_x = x_max - x_min
delta_y = y_max - y_min
delta_f = f_max - f_min

params = {
    "u_min": u_min,
    "u_max": u_max,
    "x_min": x_min,
    "x_max": x_max,
    "y_min": y_min,
    "y_max": y_max,
    "t_f": t_f,
    "f_min": f_min,
    "f_max": f_max
}

def hard_constraint(input, output):
    X = input[0]
    Y = input[1]
    tau = input[-1]
    U = ((X-1)*X*(Y-1)*Y*t_f*tau)*(output+(u_min/delta_u)) - (u_min/delta_u)
    return U

def f(sample):
    x = sample[0]*(delta_x) + x_min
    y = sample[1]*(delta_y) + y_min

    h = f_min
    
    z = h * torch.exp(-100*((x-delta_x/2)**2+(y-delta_y/2)**2))
    return z


def pde_fn(model, sample):

    # Physics Parameters
    sigma = 1.0 #kg/m^2
    T = 25.0  #N/m
    v = (T / sigma)**0.5 
    k = 1 #damping term (viscosity)

    a = (v**2)*(t_f**2)/(delta_x**2)
    b = (v**2)*(t_f**2)/(delta_y**2)
    c = (t_f**2)/delta_u

    K = k*v #damping coefficient

    J, d = _jacobian(model, sample)

    ddX = _jacobian(d, sample, i=0, j=0)[0][0]
    ddY = _jacobian(d, sample, i=1, j=1)[0][0]
    ddtau = _jacobian(d, sample, i=2, j=2)[0][0]
    dtau = J[0][-1]

    return ddtau - (a*ddX + b*ddY) - c* f(sample) + K*dtau


def ic_fn_vel(model, sample):
    J, d = _jacobian(model, sample)
    dtau = J[0][-1]
    dt = dtau*delta_u/t_f
    ics = torch.zeros_like(dt)
    return dt, ics

batchsize = 500
learning_rate = 0.002203836177626117

print("Building Domain Dataset")
domainDataset = DomainDataset([0.0]*num_inputs,[1.0]*num_inputs, 10000, period = 3)
print("Building IC Dataset")
icDataset = ICDataset([0.0]*(num_inputs-1),[1.0]*(num_inputs-1), 10000, period = 3)

print("Building Validation Dataset")
validationDataset = DomainDataset([0.0]*num_inputs,[1.0]*num_inputs, batchsize, shuffle = False)
print("Building Validation IC Dataset")
validationicDataset = ICDataset([0.0]*(num_inputs-1),[1.0]*(num_inputs-1), batchsize, shuffle = False)

encoding = GaussianEncoding(sigma = 1.0, input_size=num_inputs, encoded_size=154)
model = ModifiedMLP([num_inputs] + [308]*8 + [1], nn.SiLU, hard_constraint, p_dropout=0.1, encoding = None)

component_manager = ComponentManager()
r = ResidualComponent(pde_fn, domainDataset)
component_manager.add_train_component(r)
ic = ICComponent([ic_fn_vel], icDataset)
component_manager.add_train_component(ic)
r = ResidualComponent(pde_fn, validationDataset)
component_manager.add_validation_component(r)
ic = ICComponent([ic_fn_vel], validationicDataset)
component_manager.add_validation_component(ic)


def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

model = model.apply(init_normal)
model = model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1721, gamma=0.15913059595003437)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

data = {
    "name": "string_2inputs_nostiffness_force_damping_ic0hard_icv0_causality_t10.0_optimized_modifiedMLP",
    #"name": "prova",
    "model": model,
    "epochs": epochs,
    "batchsize": batchsize,
    "optimizer": optimizer,
    "scheduler": scheduler,
    "component_manager": component_manager,
    "additional_data": params
}

train(data, output_to_file=False)
