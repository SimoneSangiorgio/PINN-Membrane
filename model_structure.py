from pinns_v2.implementations import *
from pinns_v2.model import SimpleSpatioTemporalFFN
#from main_2_inputs import num_inputs,hard_constraint
import torch

u_min = -0.21
u_max = 0.21
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

def hard_constraint(input, output):
    X, Y, tau = input
    U = ((X-1)*X*(Y-1)*Y*t_f*tau)*(output + (u_min/delta_u)) - (u_min/delta_u)
    return U-0.5

def hard_constraint2(input, output):
    X, Y, tau = input
    U = ((X-1)*X*(Y-1)*Y)*(output + (u_min/delta_u)) - (u_min/delta_u)
    return U-0.5

layers = [3] + [308]*8 + [1]

model = SimpleSpatioTemporalFFN(
    spatial_sigmas=[1.0],  # From paper section 4.3
    temporal_sigmas=[1.0,10.0],
    hidden_layers=[200]*3, 
    activation=nn.Tanh,
    hard_constraint_fn = hard_constraint2
)

#model = ImprovedMLP(layers, nn.SiLU, hard_constraint, p_dropout=0.0, encoding = None)


