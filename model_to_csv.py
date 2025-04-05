import torch
from main_membrane import *
import numpy as np
import pandas as pd
from pathinator import model_path, csv_simulation_path, model_test_path, csv_simulation_test_path

#model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
model = torch.load(model_test_path, map_location=torch.device('cpu'), weights_only=False)
model.eval()

'''input_data = torch.tensor([8.25, 0.7, 0.8])
output = model(input_data)*delta_u + u_min
print(output)'''

# Definisci la griglia di (x, y) e i valori di t
x_step, y_step = 0.1, 0.1
t_vals = np.linspace(0, 1, 201)
x_vals = np.arange(x_min, x_max + x_step, x_step)
y_vals = np.arange(y_min, y_max + y_step, y_step)

# Crea una lista per i risultati
results = []

def format_number(val):
    return f"{val:.2f}" if val != 0 else "0."

# Esegui il modello per ciascun valore di t, x, y
for x in x_vals:
    for y in y_vals:
        for t in t_vals:
            # Crea l'input del modello
            input_tensor = torch.tensor([x, y, t], dtype=torch.float32)
            
            # Calcola l'output del modello (z)
            with torch.no_grad():
                output = model(input_tensor).numpy()

            # Aggiungi i risultati alla lista
            z_norm = output[0]*delta_u + u_min  # Supponiamo che il modello restituisca un array di dimensione [1, 1]
            results.append([format_number(t*t_f), format_number(x), format_number(y), z_norm])
            #results.append([t, x, y, z_norm])

# Crea un DataFrame e salvalo come CSV
df = pd.DataFrame(results)
#df.to_csv(csv_simulation_path, index=False, header=False)
df.to_csv(csv_simulation_test_path, index=False, header=False)
print("File CSV creato con successo.")