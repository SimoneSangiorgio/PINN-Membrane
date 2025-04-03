import torch
from model_structure import model, hard_constraint2
import numpy as np
import pandas as pd


model = torch.load('C:\\Users\\simon\\OneDrive\\Desktop\\Progetti Ingegneria\\PINN Medical\\model_1400.pt', 
                        map_location=torch.device('cpu'), weights_only=False)

model.eval()

# Definisci la griglia di (x, y) e i valori di t
x_vals = np.linspace(0, 1, 11)  # 100 valori di x tra 0 e 1
y_vals = np.linspace(0, 1, 11)  # 100 valori di y tra 0 e 1
t_vals = np.linspace(0, 10, 201)   # 50 valori di t tra 0 e 1

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
            z = output[0]  # Supponiamo che il modello restituisca un array di dimensione [1, 1]
            results.append([format_number(t), format_number(x), format_number(y), format_number(z)])

# Crea un DataFrame e salvalo come CSV
df = pd.DataFrame(results)
df.to_csv('C:\\Users\\simon\\OneDrive\\Desktop\\Progetti Ingegneria\\PINN Medical\\membrane_simulation1.csv', index=False, header=False)

print("File CSV creato con successo.")