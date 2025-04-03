import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from model import SimpleSpatioTemporalFFN

# --- Configurazione ---
MODEL_PATH = 'C:/Users/saram/OneDrive/Documenti/GitHub/PINN-Membrane/result/FFT-1000.pth'
CSV_PATH = 'C:/Users/saram/OneDrive/Documenti/GitHub/PINN-Membrane/membrane.csv'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # Cartella dello script
N_X, N_Y, N_T = 50, 50, 50

# --- Parametri Fisici ---
u_min, u_max = -0.21, 0.21
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
t_f = 10.0
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
# --- Funzione Vincolo ---
def hard_constraint2(x_in, y_out):
    X = x_in[0]
    Y = x_in[1]
    tau = x_in[-1]

    x = X*delta_x + x_min
    y = Y*delta_y + y_min
    t = tau*t_f
    u = y_out*delta_u + u_min

    u = u*(x-x_max)*(x-x_min)*(y-y_max)*(y-y_min)

    U = (u-u_min)/delta_u
    return U

# --- Modello ---
model = SimpleSpatioTemporalFFN(
    spatial_sigmas=[1.0], temporal_sigmas=[1.0, 10.0],
    hidden_layers=[200] * 3, activation=nn.Tanh, hard_constraint_fn=None
)

# --- Caricamento Pesi ---
checkpoint = torch.load(MODEL_PATH)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# --- Griglia ---
x = np.linspace(x_min, x_max, N_X)
y = np.linspace(y_min, y_max, N_Y)
t = np.linspace(0, t_f, N_T)
grid = np.stack(np.meshgrid(x, y, t, indexing='ij'), axis=-1).reshape(-1, 3)

# --- Predizione Griglia ---
with torch.no_grad():
    input_tensor = torch.tensor(grid / [x_max, y_max, t_f], dtype=torch.float32)
    u_pred_norm = model(input_tensor).numpy().flatten()
    u_constrained = np.array([hard_constraint2(g, u) for g, u in zip(grid / [x_max, y_max, t_f], u_pred_norm)])

print(f"Predizione Griglia: {u_constrained.reshape(N_X, N_Y, N_T).shape}")

# --- Validazione CSV ---
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, header=None, names=['x', 'y', 't', 'u'])
    with torch.no_grad():
        inputs = torch.tensor(df[['x', 'y', 't']].values / [x_max, y_max, t_f], dtype=torch.float32)
        u_pred_norm_csv = model(inputs).numpy().flatten()
        u_constrained_csv = np.array([hard_constraint2(g, u) for g, u in zip(df[['x', 'y', 't']].values / [x_max, y_max, t_f], u_pred_norm_csv)])

    # Calcola errori
    mse = (u_constrained_csv - df['u'].values) ** 2
    l2_diff = np.abs(u_constrained_csv - df['u'].values)

    # Crea DataFrame e salva in CSV
    results = pd.DataFrame({
        'u_real': df['u'],
        'u_pred': u_constrained_csv,
        'MSE': mse,
        'L2_diff': l2_diff
    })

    # Statistiche riassuntive
    results['MSE_mean'] = mse.mean()
    results['L2_diff_mean'] = l2_diff.mean()
    results['MSE_std'] = mse.std()
    results['L2_diff_std'] = l2_diff.std()

    results.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_results.csv'), index=False)
    print("Risultati CSV salvati in 'evaluation_results.csv'")

    # --- Grafici ---
    plt.figure(figsize=(12, 6))

    # Grafico u_real vs u_pred
    plt.subplot(1, 2, 1)
    plt.scatter(df['u'], u_constrained_csv)
    plt.xlabel('u_real')
    plt.ylabel('u_pred')
    plt.title('u_real vs u_pred')

    # Grafico errori
    plt.subplot(1, 2, 2)
    plt.plot(mse, label='MSE')
    plt.plot(l2_diff, label='L2_diff')
    plt.xlabel('Punto')
    plt.ylabel('Errore')
    plt.title('Errori per Punto')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'evaluation_plots.png'))
    plt.show()