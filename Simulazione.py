import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathinator import model_path, csv_simulation_path, csv_real_path

real_membrane = pd.read_csv(csv_real_path, names=["t", "x", "y", "z_real"], header=None)
real_membrane = real_membrane.apply(lambda col: col.round(2) if col.name in real_membrane.select_dtypes(include=np.number).columns[:3] else col)

simulated_membrane = pd.read_csv(csv_simulation_path, names=["t", "x", "y", "z_sim"], header=None)

error_z = real_membrane["z_real"]-simulated_membrane["z_sim"]
error_membrane = pd.DataFrame({"t": real_membrane["t"], "x": real_membrane["x"],"y": real_membrane["y"],"z_err": error_z})

merged_data = pd.merge(real_membrane, simulated_membrane, on=['t', 'x', 'y'], how='inner')
merged_data = pd.merge(merged_data, error_membrane, on=['t', 'x', 'y'], how='inner')


t_vals = np.sort(merged_data['t'].unique())
x_vals = np.sort(merged_data['x'].unique())
y_vals = np.sort(merged_data['y'].unique())
X, Y = np.meshgrid(x_vals, y_vals)


x_min, x_max = merged_data['x'].min(), merged_data['x'].max()
y_min, y_max = merged_data['y'].min(), merged_data['y'].max()
z_min = min(merged_data['z_real'].min(), merged_data['z_sim'].min(), merged_data['z_err'].min())
z_max = max(merged_data['z_real'].max(), merged_data['z_sim'].max(), merged_data['z_err'].max())


fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

min_z_real = merged_data['z_real'].min()
min_z_simul = merged_data['z_sim'].min()

def update_all(frame):
    t = t_vals[frame]

    def update_subplot(ax, data, z_col, title):
        ax.cla()
        frame_data = data[data['t'] == t]
        Z = np.zeros_like(X)
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                z_val = frame_data[(frame_data['x'] == x) & (frame_data['y'] == y)][z_col].values
                Z[i, j] = z_val[0] if z_val.size > 0 else np.nan

        if title == 'Error Membrane':
            ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', alpha=0.8)
            ax.set_title(f"{title}")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(merged_data['z_err'].min(), merged_data['z_err'].max())            
        else:
            ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
            ax.set_title(f"{title}")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

    update_subplot(ax1, merged_data, 'z_real', f'Real Membrane \n z min = {min_z_real:.5f}')
    update_subplot(ax2, merged_data, 'z_sim', f'Simulated Membrane \n z min = {min_z_simul:.5f}')
    update_subplot(ax3, merged_data, 'z_err', 'Error Membrane')


    fig.suptitle(f"time = {t:.2f}", fontsize=16)

ani = FuncAnimation(fig, update_all, frames=len(t_vals), interval=50, blit=False)
plt.show()
