import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Legge i file CSV
data = pd.read_csv("C:\\Users\\Simone\\Downloads\\Telegram Desktop\\membrane_simulation.csv", 
                   names=["t", "x", "y", "z"], header=None)
simulated_data = pd.read_csv("C:\\Users\\Simone\\Downloads\\Telegram Desktop\\membrane_simulation.csv", 
                             names=["t", "x", "y", "z"], header=None)

# Ottieni i valori unici per x, y e t
time_vals = np.sort(data['t'].unique())
x_vals = np.sort(data['x'].unique())
y_vals = np.sort(data['y'].unique())
X, Y = np.meshgrid(x_vals, y_vals)

# Imposta i limiti comuni per gli assi
x_min, x_max = data['x'].min(), data['x'].max()
y_min, y_max = data['y'].min(), data['y'].max()
z_min = min(data['z'].min(), simulated_data['z'].min())
z_max = max(data['z'].max(), simulated_data['z'].max())

# Crea una figura con due subplot 3D
fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

def update_all(frame):
    t = time_vals[frame]
    
    # Aggiorna il subplot per i dati reali
    ax1.cla()  # Pulisce l'asse
    frame_data = data[data['t'] == t]
    Z = np.zeros_like(X)
    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            z_val = frame_data[(frame_data['x'] == x) & (frame_data['y'] == y)]['z'].values
            Z[i, j] = z_val[0] if z_val.size > 0 else 0
    ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    #ax1.set_title(f"Dati Reali\n t = {t:.2f}")
    ax1.set_title(f"\n t = {t:.2f}")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    
    # DA IMPLEMENTARE
    '''ax2.cla()  # Pulisce l'asse
    frame_sim = simulated_data[simulated_data['t'] == t]
    Z_sim = np.zeros_like(X)
    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            z_val = frame_sim[(frame_sim['x'] == x) & (frame_sim['y'] == y)]['z'].values
            Z_sim[i, j] = z_val[0] if z_val.size > 0 else 0
    ax2.plot_surface(X, Y, Z_sim, cmap='viridis', edgecolor='none', alpha=0.8)
    ax2.set_title(f"Dati Simulati\n t = {t:.2f}")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)'''

# Crea l'animazione che aggiorna entrambi i subplot
ani = FuncAnimation(fig, update_all, frames=len(time_vals), interval=50, blit=False)
plt.show()
