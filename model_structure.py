from pinns_v2.model import *
from pinns_v2.rff import *
#from main_2_inputs import num_inputs,hard_constraint
import torch

num_inputs = 3
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
    return U


layers = [num_inputs] + [308]*8 + [1]

encoding = GaussianEncoding(sigma = 1.0, input_size=num_inputs, encoded_size=150)
model = MLP(layers, nn.SiLU, hard_constraint, p_dropout=0.0, encoding = encoding)

#model = ImprovedMLP(layers, nn.SiLU, hard_constraint, p_dropout=0.0, encoding = None)


