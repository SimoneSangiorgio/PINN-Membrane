import torch
# import torch.nn as nn # Non serve importarlo qui se l'intera struttura è nel .pth
import numpy as np
import pandas as pd
import os
import sys # Necessario per modificare sys.path
import matplotlib.pyplot as plt

# --- Funzione Dummy per soddisfare Pickle ---
# Definisci una funzione con lo stesso nome che esisteva nello script
# *al momento del salvataggio* del modello. Serve solo per il caricamento.
def hard_constraint2(x_in, y_out):
    """
    Funzione fittizia (dummy) per permettere a torch.load di trovare
    l'attributo 'hard_constraint2' che era presente nel modulo __main__
    al momento del salvataggio del modello completo.
    Il suo contenuto effettivo di solito non viene eseguito durante il caricamento.
    """
    # print("Warning: Chiamata la funzione dummy hard_constraint2 durante il caricamento!") # Uncomment for debugging
    # È sufficiente che la funzione esista. Possiamo semplicemente passare o ritornare l'output.
    # pass
    return y_out
# -------------------------------------------

# --- Configurazione ---
MODEL_PATH = 'C:/Users/saram/OneDrive/Documenti/GitHub/PINN-Membrane/result/model_10_epochs.pth'
CSV_PATH = 'C:/Users/saram/OneDrive/Documenti/GitHub/PINN-Membrane/membrane.csv'

# Determina la directory di output basata sulla posizione dello script
try:
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Gestisce casi in cui __file__ non è definito (es. ambienti interattivi)
    OUTPUT_DIR = '.' # Usa la directory di lavoro corrente
    print(f"Warning: Impossibile determinare automaticamente la directory dello script. Uso la directory corrente: {os.path.abspath(OUTPUT_DIR)}")

# --- Aggiungi la directory genitore al percorso di Python (Opzione B) ---
# Necessario se si esegue questo script da una sottodirectory (es. pinns_v2)
# e il modello salvato si aspetta di trovare moduli nella directory genitore.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"Aggiunta la directory genitore a sys.path: {parent_dir}")
except NameError:
     print("Warning: Impossibile determinare automaticamente la directory genitore da aggiungere a sys.path (__file__ non definito).")
# ---------------------------------------------------------

# --- Parametri Griglia ---
N_X, N_Y, N_T = 50, 50, 50

# --- Parametri Fisici ---
u_min, u_max = -0.21, 0.21
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
t_f = 10.0
delta_u = u_max - u_min # Differenza per denormalizzazione output

# --- Caricamento Modello ---
# Carica l'intero oggetto modello, assumendo salvato con torch.save(model, PATH)
print(f"Caricamento modello da: {MODEL_PATH}")
model = None # Inizializza a None
try:
    # Usa weights_only=False perché stiamo caricando l'intero oggetto modello (architettura + pesi)
    # La funzione dummy hard_constraint2 definita sopra dovrebbe risolvere l'AttributeError
    model = torch.load(MODEL_PATH, weights_only=False)
    # Assicura che il modello sia in modalità valutazione
    model.eval()
    print("Modello caricato con successo.")

except FileNotFoundError:
    print(f"Errore: File modello non trovato in {MODEL_PATH}")
    exit()
except AttributeError as e:
    print(f"Errore durante il caricamento del modello - AttributeError: {e}")
    print("Questo di solito accade se una funzione o classe definita nello script di salvataggio")
    print("non è presente (o ha un nome diverso) nello script di caricamento.")
    print("Assicurati che la funzione 'dummy' necessaria (es. hard_constraint2) sia definita prima di torch.load.")
    exit()
except ModuleNotFoundError as e:
     print(f"Errore durante il caricamento del modello - ModuleNotFoundError: {e}")
     print("Questo errore si verifica se il modulo Python contenente la definizione della classe del modello")
     print("(come salvato nel file .pth) non può essere trovato.")
     print("Verifica che la modifica a sys.path sia corretta o esegui lo script dalla directory genitore corretta.")
     exit()
except Exception as e: # Cattura altri errori generici
    print(f"Errore imprevisto durante il caricamento del modello: {e}")
    exit()

# Procedi solo se il modello è stato caricato con successo
if model is not None:

    # --- Generazione Griglia ---
    x = np.linspace(x_min, x_max, N_X)
    y = np.linspace(y_min, y_max, N_Y)
    t = np.linspace(0, t_f, N_T)
    # Crea una griglia di punti (x, y, t)
    grid = np.stack(np.meshgrid(x, y, t, indexing='ij'), axis=-1).reshape(-1, 3)
    print(f"Griglia generata con shape: {grid.shape}")

    # --- Predizione sulla Griglia ---
    # Normalizza gli input della griglia per il modello (assumendo min = 0)
    grid_norm = grid / [x_max, y_max, t_f]

    print("Inizio predizione sulla griglia...")
    with torch.no_grad(): # Disabilita il calcolo dei gradienti per l'inferenza
        input_tensor = torch.tensor(grid_norm, dtype=torch.float32)
        # Esegui il modello
        u_pred_norm = model(input_tensor).numpy().flatten() # Output normalizzato

    # Denormalizza le predizioni per ottenere i valori fisici
    u_pred = u_pred_norm * delta_u + u_min
    print("Predizione sulla griglia completata.")
    print(f"Shape predizioni griglia: {u_pred.reshape(N_X, N_Y, N_T).shape}")
    print(f"Min/Max predizioni griglia: {u_pred.min():.4f} / {u_pred.max():.4f}")


    # --- Validazione su file CSV (se esiste) ---
    if os.path.exists(CSV_PATH):
        print(f"\nInizio validazione dati da: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH, header=None, names=['x', 'y', 't', 'u'])

        # Normalizza gli input del CSV per il modello
        csv_inputs_norm = df[['x', 'y', 't']].values / [x_max, y_max, t_f]

        print("Inizio predizione sui dati CSV...")
        with torch.no_grad():
            inputs_tensor_csv = torch.tensor(csv_inputs_norm, dtype=torch.float32)
            # Esegui il modello sui dati CSV
            u_pred_norm_csv = model(inputs_tensor_csv).numpy().flatten() # Output normalizzato

        # Denormalizza le predizioni CSV
        u_pred_csv = u_pred_norm_csv * delta_u + u_min
        print("Predizione sui dati CSV completata.")

        # Calcola gli errori
        u_real_csv = df['u'].values
        mse = (u_pred_csv - u_real_csv) ** 2
        abs_error = np.abs(u_pred_csv - u_real_csv) # Errore Assoluto (L1)

        # Crea DataFrame con i risultati e salvalo
        results = pd.DataFrame({
            'u_real': u_real_csv,
            'u_pred': u_pred_csv,
            'MSE': mse,
            'Abs_Error': abs_error
        })

        # Calcola statistiche riassuntive sugli errori
        mse_mean = mse.mean()
        abs_error_mean = abs_error.mean()
        mse_std = mse.std()
        abs_error_std = abs_error.std()

        print(f"\nRisultati Validazione (CSV):")
        print(f"  Mean Squared Error (MSE): {mse_mean:.6f}")
        print(f"  Mean Absolute Error (MAE/L1): {abs_error_mean:.6f}")
        print(f"  Std Dev MSE: {mse_std:.6f}")
        print(f"  Std Dev MAE/L1: {abs_error_std:.6f}")

        output_csv_path = os.path.join(OUTPUT_DIR, 'evaluation_results_full_model.csv') # Nome file aggiornato
        results.to_csv(output_csv_path, index=False)
        print(f"\nRisultati CSV salvati in: '{output_csv_path}'")

        # --- Creazione Grafici ---
        print("Creazione grafici...")
        plt.figure(figsize=(12, 5)) # Leggermente più piccolo in altezza

        # Grafico 1: Valori Reali vs Predetti
        plt.subplot(1, 2, 1)
        plt.scatter(u_real_csv, u_pred_csv, alpha=0.5, s=10, label='Punti Dati') # Aggiunta etichetta
        # Linea ideale y=x per riferimento
        min_val = min(u_real_csv.min(), u_pred_csv.min()) * 1.1 # Aggiunge margine
        max_val = max(u_real_csv.max(), u_pred_csv.max()) * 1.1 # Aggiunge margine
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideale (y=x)')
        plt.xlabel('Valore Reale (u_real)')
        plt.ylabel('Valore Predetto (u_pred)')
        plt.title('Confronto Valori Reali vs Predetti (CSV)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # Assicura che gli assi abbiano la stessa scala

        # Grafico 2: Distribuzione degli Errori Assoluti
        plt.subplot(1, 2, 2)
        plt.hist(abs_error, bins=30, alpha=0.7, color='orange', label='Errore Assoluto')
        # plt.plot(mse, label=f'MSE (Media={mse_mean:.4f})', alpha=0.7) # Plot MSE meno informativo qui
        # plt.plot(abs_error, label=f'Errore Assoluto (Media={abs_error_mean:.4f})', alpha=0.7) # Plot punto per punto meno informativo dell'istogramma
        plt.xlabel('Errore Assoluto (|u_real - u_pred|)')
        plt.ylabel('Frequenza')
        plt.title(f'Distribuzione Errore Assoluto (Media={abs_error_mean:.4f})')
        plt.legend()
        plt.grid(True)
        # plt.yscale('log') # Log scale utile se ci sono pochi errori molto grandi

        plt.tight_layout() # Aggiusta spazi tra subplot
        output_plot_path = os.path.join(OUTPUT_DIR, 'evaluation_plots_full_model.png') # Nome file aggiornato
        plt.savefig(output_plot_path)
        print(f"Grafici salvati in: '{output_plot_path}'")
        plt.show()

    else:
        print(f"\nFile CSV non trovato in: {CSV_PATH}")
        print("Saltando la sezione di validazione e grafici.")

else:
    print("\nImpossibile procedere perché il modello non è stato caricato.")

print("\nScript terminato.")