import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from timeit import default_timer as timer

from pinns_v2.model import SimpleSpatioTemporalFFN, MLP, ModifiedMLP  # Useremo MLP come da tuo codice
from pinns_v2.rff import GaussianEncoding

def hard_constraint2(x_in, y_out):
    X = x_in[0, 0]  # Accedi al primo elemento della seconda dimensione
    Y = x_in[0, 1]  # Accedi al secondo elemento della seconda dimensione
    tau = x_in[0, 2]  # Accedi al terzo elemento della seconda dimensione

    x = X * delta_x + x_min
    y = Y * delta_y + y_min
    t = tau * t_f
    u = y_out * delta_u + u_min

    u = u * (x - x_max) * (x - x_min) * (y - y_max) * (y - y_min)

    U = (u - u_min) / delta_u
    return U

num_inputs = 3  # x, y, t

u_min = -0.21
u_max = 0.0
x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0
t_f = 10.0  # Assicurati sia float
delta_u = u_max - u_min
delta_x = x_max - x_min
delta_y = y_max - y_min

# --- Configurazione Utente ---
# 1. Specifica il percorso del file .pth con i pesi del modello ADDRESTRATO
#   Assicurati che questo file .pth corrisponda all'architettura MLP definita sotto.
MODEL_WEIGHTS_PATH = 'C:/Users/saram/OneDrive/Documenti/GitHub/PINN-Membrane/result/FFT-1000.pth'
# 2. Specifica il percorso del file .csv con i dati di validazione (opzionale)
DATA_CSV_PATH = 'C:/Users/saram/OneDrive/Documenti/GitHub/PINN-Membrane/membrane.csv'

# 3. Risoluzione della griglia mesh (es. 50 punti per dimensione)
N_X = 50
N_Y = 50
N_T = 50
# --------------------------

# --- Architettura Modello (dal tuo codice) ---
# Assicurati che corrisponda a quella usata per salvare MODEL_WEIGHTS_PATH
print("Istanziazione del modello...")

# Parametri per l'encoding e MLP come nel tuo codice di training fornito
encoding_sigma = 1.0
encoding_size = 154
mlp_hidden_layers = [308] * 8
mlp_activation = nn.SiLU

encoding = GaussianEncoding(sigma=encoding_sigma, input_size=num_inputs, encoded_size=encoding_size)
model_layers = [num_inputs] + mlp_hidden_layers + [1]  # [num_inputs] qui è un placeholder, l'encoding cambia la dim effettiva

# Passiamo None per hard_constraint qui, lo applicheremo dopo
model = SimpleSpatioTemporalFFN(
    spatial_sigmas=[1.0],  # From paper section 4.3
    temporal_sigmas=[1.0, 10.0],
    hidden_layers=[200] * 3,
    activation=nn.Tanh,
    hard_constraint_fn=hard_constraint2
)
print(f"Modello {type(model).__name__} istanziato.")
print(f"Architettura: Encoding({encoding_size * 2}) -> MLP({model_layers[1:]})")  # Dim input MLP è encoding_size*2

# --- Caricamento Pesi ---
print(f"Caricamento pesi da: {MODEL_WEIGHTS_PATH}")
if not os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"Errore: File dei pesi non trovato: {MODEL_WEIGHTS_PATH}")
    sys.exit(1)
try:
    checkpoint = torch.load(MODEL_WEIGHTS_PATH)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print("Pesi caricati con successo.")
except Exception as e:
    print(f"Errore durante il caricamento dei pesi da {MODEL_WEIGHTS_PATH}: {e}")
    sys.exit(1)

# Imposta il modello in modalità valutazione
model.eval()

# --- Generazione Mesh Grid ---
print(f"Generazione mesh grid con risoluzione ({N_X} x {N_Y} x {N_T})...")
x_lin = np.linspace(x_min, x_max, N_X)
y_lin = np.linspace(y_min, y_max, N_Y)
t_lin = np.linspace(0, t_f, N_T)  # Tempo fisico da 0 a t_f

# Crea la griglia di coordinate FISICHE
x_grid, y_grid, t_grid = np.meshgrid(x_lin, y_lin, t_lin, indexing='ij')

# Normalizza le coordinate per l'input del modello [0, 1]
X_grid_norm = (x_grid - x_min) / delta_x
Y_grid_norm = (y_grid - y_min) / delta_y
tau_grid_norm = t_grid / t_f  # tau = t / t_f

# Appiattisci e combina in un unico array di input normalizzato (N_total, 3)
grid_shape = x_grid.shape

N_total = N_X * N_Y * N_T
input_grid_flat_norm = np.stack([X_grid_norm.ravel(),
                                 Y_grid_norm.ravel(),
                                 tau_grid_norm.ravel()], axis=-1)

print(f"Griglia generata. Numero totale di punti: {N_total}")

# --- Predizione sulla Griglia (SENZA BATCH) ---
print("Inizio predizione sulla griglia (punto per punto)...")
u_pred_constrained_flat = np.zeros(N_total)  # Per salvare l'output fisico vincolato
u_pred_raw_flat = np.zeros(N_total)  # Per salvare l'output normalizzato raw

start_time = timer()
with torch.no_grad():  # Disattiva calcolo gradienti
    for i in range(N_total):
        # Prendi il singolo punto normalizzato
        input_point_norm_np = input_grid_flat_norm[i:i + 1, :]  # Shape (1, 3)
        input_point_norm = torch.tensor(input_point_norm_np, dtype=torch.float32)

        # Predizione RAW normalizzata dal modello
        U_pred_raw_norm = model(input_point_norm).cpu().numpy().item()  # Output è (1, 1) -> scalar
        u_pred_raw_flat[i] = U_pred_raw_norm

        # --- Applica HARD CONSTRAINT 2 (logica dal testo) ---
        # 1. Ottieni coordinate normalizzate e fisiche del punto corrente
        X_norm = input_point_norm_np[0, 0]
        Y_norm = input_point_norm_np[0, 1]
        tau_norm = input_point_norm_np[0, 2]

        x_phys = X_norm * delta_x + x_min
        y_phys = Y_norm * delta_y + y_min
        # t_phys = tau_norm * t_f # Già in t_grid

        # 2. Denormalizza l'output raw del modello
        u_pred_raw_phys = U_pred_raw_norm * delta_u + u_min

        # 3. Applica il vincolo moltiplicativo sui bordi spaziali
        # u = u * (x-x_max)*(x-x_min)*(y-y_max)*(y-y_min)
        boundary_term = (x_phys - x_max) * (x_phys - x_min) * \
                        (y_phys - y_max) * (y_phys - y_min)
        u_pred_constrained_phys = u_pred_raw_phys * boundary_term

        # Salva il risultato fisico vincolato
        u_pred_constrained_flat[i] = u_pred_constrained_phys

        if (i + 1) % 5000 == 0:
            print(f"  ...processati {i + 1}/{N_total} punti")

end_time = timer()
print(f"Predizione sulla griglia completata in {end_time - start_time:.2f} secondi.")

# --- Riorganizza i risultati nella forma della griglia ---
u_pred_constrained_grid = u_pred_constrained_flat.reshape(grid_shape)
u_pred_raw_norm_grid = u_pred_raw_flat.reshape(grid_shape)

print("\n--- Risultati della Griglia ---")
print(f"Shape della griglia predetta (u_constrained): {u_pred_constrained_grid.shape}")
# Puoi visualizzare/salvare u_pred_constrained_grid qui
# Esempio: valore al centro della griglia per t=0
center_x_idx, center_y_idx = N_X // 2, N_Y // 2
print(f"Valore predetto vincolato al centro (x={x_lin[center_x_idx]:.2f}, y={y_lin[center_y_idx]:.2f}, t={t_lin[0]:.2f}): {u_pred_constrained_grid[center_x_idx, center_y_idx, 0]:.6f}")
print(f"Valore raw normalizzato corrispondente: {u_pred_raw_norm_grid[center_x_idx, center_y_idx, 0]:.6f}")

# --- (Opzionale) Validazione con File CSV ---
if DATA_CSV_PATH and os.path.exists(DATA_CSV_PATH):
    print(f"\n--- Validazione con {DATA_CSV_PATH} ---")
    
    def convert_u(u_str):
            try:
                return float(u_str.strip('{}'))
            except ValueError:
                return np.nan  # Restituisci NaN se la conversione fallisce

    try:
        # Leggi il file CSV senza intestazione (header=None)
        df = pd.read_csv("C:/Users/saram/OneDrive/Documenti/GitHub/PINN-Membrane/membrane.csv", header=None)

        # Assegna manualmente i nomi delle colonne
        df.columns = ['x', 'y', 't', 'u']

        print("File CSV caricato e colonne rinominate con successo.")
        # ... continua con l'elaborazione dei dati ...
    except FileNotFoundError:
        print("Errore: File CSV non trovato.")
    except Exception as e:
        print(f"Errore durante la lettura del file CSV: {e}")
    
    if 'df' in locals():
        input_cols = ['x', 'y', 't']
        output_col = 'u'
        data_df = df
        if not all(col in data_df.columns for col in input_cols + [output_col]):
            print(f"Errore: Colonne richieste ({input_cols + [output_col]}) non trovate in {DATA_CSV_PATH}")
        else:
            # Prepara input e output dal CSV
            x_csv = data_df['x'].values
            y_csv = data_df['y'].values
            t_csv = data_df['t'].values
            u_true_csv = data_df[output_col].values

            # Normalizza input CSV per il modello
            X_csv_norm = (x_csv - x_min) / delta_x
            Y_csv_norm = (y_csv - y_min) / delta_y
            tau_csv_norm = t_csv / t_f
            input_csv_norm = np.stack([X_csv_norm, Y_csv_norm, tau_csv_norm], axis=-1)

            # Converti in tensori
            X_tensor_csv = torch.tensor(input_csv_norm, dtype=torch.float32)
            u_true_tensor_csv = torch.tensor(u_true_csv, dtype=torch.float32).unsqueeze(1)  # Shape (N, 1)

            # Predici sui dati CSV (qui possiamo usare batch se vogliamo, o punto per punto)
            print(f"Predizione sui {len(data_df)} punti del CSV...")
            with torch.no_grad():
                # Predizione raw normalizzata
                U_pred_raw_norm_csv = model(X_tensor_csv)  # Output (N, 1)

                # --- Applica hard_constraint2 anche qui per confronto diretto con 'u' ---
                # 1. Denormalizza output raw
                u_pred_raw_phys_csv = U_pred_raw_norm_csv * delta_u + u_min  # Ora è (N, 1)

                # 2. Applica vincolo moltiplicativo (richiede coordinate fisiche)
                x_phys_csv_t = torch.tensor(x_csv, dtype=torch.float32).unsqueeze(1)
                y_phys_csv_t = torch.tensor(y_csv, dtype=torch.float32).unsqueeze(1)
                boundary_term_csv = (x_phys_csv_t - x_max) * (x_phys_csv_t - x_min) * \
                                     (y_phys_csv_t - y_max) * (y_phys_csv_t - y_min)
                u_pred_constrained_phys_csv = u_pred_raw_phys_csv * boundary_term_csv  # Ora è (N, 1)

            # Calcola Errori (tra predizione VINCOLATA e u_true dal CSV)
            mse = torch.mean((u_pred_constrained_phys_csv - u_true_tensor_csv) ** 2)
            l2_diff_norm = torch.linalg.norm(u_pred_constrained_phys_csv - u_true_tensor_csv)
            l2_true_norm = torch.linalg.norm(u_true_tensor_csv)
            l2_rel_err = l2_diff_norm / l2_true_norm if l2_true_norm > 0 else torch.tensor(float('inf'))

            print("\nErrori calcolati rispetto al CSV (usando predizioni vincolate):")
            print(f"  Mean Squared Error (MSE): {mse.item():.8f}")
            print(f"  L2 Norm della Differenza: {l2_diff_norm.item():.8f}")
            print(f"  Errore Relativo L2:     {l2_rel_err.item():.8f} (ovvero {l2_rel_err.item() * 100:.4f} %)")
else:
    print("\nPercorso file CSV non specificato o file non trovato. Salto la validazione.")

print("\nScript terminato.")