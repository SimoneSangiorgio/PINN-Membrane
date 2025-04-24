# START OF video_plot_6.py (Simplified, Explicit FFmpeg Path)

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
import torch
import numpy as np
import json
from scipy.io import savemat
import time
import sys

print("--- Starting video_plot_6.py ---")
def hard_constraint(x, y_out):
    X = x[:, 0].reshape(-1, 1)
    Y = x[:, 1].reshape(-1, 1)
    tau = x[:, -1].reshape(-1, 1)

    x = X*delta_x + x_min
    y = Y*delta_y + y_min
    t = tau*t_f
    u = y_out*delta_u + u_min

    u = u*(x-x_max)*(x-x_min)*(y-y_max)*(y-y_min)

    U = (u-u_min)/delta_u
    return U
# --- Configuration ---
# !!! SET YOUR CORRECT EXPERIMENT NAME !!!
experiment_name = "membrane_6inputs_EFFN_Fmax_-1"
# !!! SET YOUR CORRECT MODEL FILENAME !!!
model_filename = 'model_200_4_tanh.pt'

# --- !!! IMPORTANT: FFmpeg Path !!! ---
# Option 1: Set the path directly here if FFmpeg is NOT reliably in your system PATH
# Replace None with the FULL path to ffmpeg.exe, using raw string (r"...") or double backslashes (\\)
# Example: ffmpeg_path_if_needed = r"C:\tools\ffmpeg\bin\ffmpeg.exe"
ffmpeg_path_if_needed = None

# Option 2: Provide path as a command-line argument when running the script
# Example: python video_plot_6.py "C:\tools\ffmpeg\bin\ffmpeg.exe"
if len(sys.argv) > 1:
    ffmpeg_path_if_needed = sys.argv[1]
    print(f"[*] Using FFmpeg path from command line: {ffmpeg_path_if_needed}")

# Set Matplotlib's ffmpeg path if explicitly provided and exists
if ffmpeg_path_if_needed and os.path.exists(ffmpeg_path_if_needed):
    try:
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path_if_needed
        print(f"[*] Successfully set Matplotlib ffmpeg path to: {ffmpeg_path_if_needed}")
    except Exception as e:
        print(f"[Warning] Could not set Matplotlib ffmpeg path: {e}. Will rely on system PATH.")
elif ffmpeg_path_if_needed:
    print(f"[Warning] Explicitly provided FFmpeg path not found: {ffmpeg_path_if_needed}. Will rely on system PATH.")
else:
    print("[*] No explicit FFmpeg path provided. Relying on system PATH.")
    print("[*] If saving fails, ensure ffmpeg.exe is in PATH or provide the path.")

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output", experiment_name)
model_path = os.path.join(output_dir, "model", model_filename)
params_path = os.path.join(output_dir, "params.json")
# --- Simplified Video Filename ---
video_filename = "model_200_4_tanh.mp4" 
video_path = os.path.join(output_dir, video_filename)
mat_path = os.path.join(output_dir, "data_all.mat") # Keep MAT filename consistent

print(f"[*] Using Model: {model_path}")
print(f"[*] Saving Video to: {video_path}") # Corrected print statement path
print(f"[*] Saving MAT: {mat_path}")

# --- Parameter Loading ---
with open(params_path, "r") as fp:
    params = json.load(fp)["additionalData"]
u_min = params["u_min"]; u_max = params["u_max"]
x_min = params["x_min"]; x_max = params["x_max"]
y_min = params["y_min"]; y_max = params["y_max"]
f_min = params["f_min"]; f_max = params["f_max"]
t_f = params["t_f"]
delta_u = u_max - u_min; delta_x = x_max - x_min
delta_f = f_max - f_min; delta_y = y_max - y_min

# --- Input Composition (No changes) ---
def compose_input(x_grid, y_grid, x_f_val, y_f_val, h_val, t_vec):
    num_t_steps = t_vec.shape[0]
    num_spatial_points = x_grid.size
    x_flat = x_grid.flatten().reshape(-1, 1)
    y_flat = y_grid.flatten().reshape(-1, 1)
    X_norm = (x_flat - x_min) / (delta_x + 1e-9)
    Y_norm = (y_flat - y_min) / (delta_y + 1e-9)
    X_f_norm = (x_f_val - x_min) / (delta_x + 1e-9)
    Y_f_norm = (y_f_val - y_min) / (delta_y + 1e-9)
    H_norm = (h_val - f_min) / (delta_f + 1e-9)
    T_norm = t_vec.reshape(-1, 1) / t_f
    X_ = np.tile(X_norm, (num_t_steps, 1))
    Y_ = np.tile(Y_norm, (num_t_steps, 1))
    X_f1 = np.full_like(X_, X_f_norm)
    Y_f1 = np.full_like(Y_, Y_f_norm)
    H_ = np.full_like(X_, H_norm)
    T_ = np.repeat(T_norm, num_spatial_points, axis=0)
    X_combined = np.hstack((X_, Y_, X_f1, Y_f1, H_, T_))
    return torch.Tensor(X_combined).to(torch.device("cpu"))

# --- Model Loading ---
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()

# --- Setup for Prediction ---
num_t_steps_vid = 201
num_x_steps_vid = 101
num_y_steps_vid = 101
tt = np.linspace(0, t_f, num=num_t_steps_vid, endpoint=True)
x_lin = np.linspace(x_min, x_max, num=num_x_steps_vid, endpoint=True)
y_lin = np.linspace(y_min, y_max, num=num_y_steps_vid, endpoint=True)
x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)
x_f1_fixed = 0.8; y_f1_fixed = 0.8; h_fixed = -3.0

# --- Generate Input & Predict ---
print("[*] Generating input and predicting...")
X_input_tensor = compose_input(x_mesh, y_mesh, x_f1_fixed, y_f1_fixed, h_fixed, tt)
preds_physical = np.zeros(X_input_tensor.shape[0])
batch_size = 10000
time_start_pred = time.time()
with torch.no_grad():
    for i in range(0, X_input_tensor.shape[0], batch_size):
        elem_norm_input = X_input_tensor[i:min(i + batch_size, X_input_tensor.shape[0])]
        pred_norm_output = model(elem_norm_input)
        pred_physical_batch = pred_norm_output.cpu().numpy() * delta_u + u_min
        preds_physical[i:min(i + batch_size, X_input_tensor.shape[0])] = pred_physical_batch.flatten()
time_end_pred = time.time()
print(f"[+] Prediction finished in {time_end_pred - time_start_pred:.2f} seconds.")

# --- Reshape ---
preds_reshaped = preds_physical.reshape(num_t_steps_vid, y_mesh.shape[0], x_mesh.shape[0])

# --- Animation ---
print("[*] Setting up animation...")
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

pred_min_actual = np.min(preds_reshaped)
pred_max_actual = np.max(preds_reshaped)
z_plot_min = max(u_min, pred_min_actual - abs(pred_min_actual)*0.05)
z_plot_max = min(u_max, pred_max_actual + abs(pred_max_actual)*0.05)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("u")

surf = ax.plot_surface(x_mesh, y_mesh, preds_reshaped[0, :, :],
                       cmap=cm.viridis, vmin=pred_min_actual, vmax=pred_max_actual,
                       linewidth=0, antialiased=False)
title_obj = ax.set_title(f"t = {tt[0]:.2f}")
fig.colorbar(surf, shrink=0.5, aspect=10, label="u")

def update(frame):
    ax.clear()
    ax.set_zlim(z_plot_min, z_plot_max)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("u")
    new_surf = ax.plot_surface(x_mesh, y_mesh, preds_reshaped[frame, :, :],
                               cmap=cm.viridis, vmin=pred_min_actual, vmax=pred_max_actual,
                               linewidth=0, antialiased=False)
    title_obj.set_text(f"t = {tt[frame]:.2f}")
    return new_surf, title_obj

frames = num_t_steps_vid
fps = 30
interval_ms = 1000 / fps

print(f"[*] Creating animation ({frames} frames, {fps} fps)...")
ani = FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)

print(f"[*] Saving animation using '{plt.rcParams['animation.writer']}' writer to: {video_path}")
try:
    # Force using the 'ffmpeg' writer IF the rcParam was set OR if it's the default
    ani.save(video_path, writer='ffmpeg', fps=fps, dpi=150)
    print(f"[+] Video saved successfully.")
except Exception as e:
     print(f"\n[ERROR] Failed to save animation: {e}")
     print("[ERROR] ---- IMPORTANT ----")
     print("[ERROR] 1. Ensure FFmpeg EXECUTABLE is installed (ffmpeg.org/download.html).")
     print("[ERROR] 2. Ensure the 'bin' directory containing ffmpeg.exe is in your system PATH.")
     print("[ERROR] 3. RESTART your terminal/IDE after changing PATH.")
     print("[ERROR] 4. Alternatively, edit 'ffmpeg_path_if_needed' in this script")
     print("[ERROR]    OR provide the full path to ffmpeg.exe as a command-line argument:")
     print(f"[ERROR]       python {os.path.basename(__file__)} \"C:\\path\\to\\ffmpeg\\bin\\ffmpeg.exe\"")
     print("[ERROR] -------------------")


plt.close(fig)

# --- Save Data to MAT file ---
print(f"[*] Saving data to: {mat_path}")
mdic = {
    "pinn_data": preds_reshaped.astype(np.float64),
    "X_pinn": x_mesh.astype(np.float64),
    "Y_pinn": y_mesh.astype(np.float64),
}
savemat(mat_path, mdic)
print("[+] Data saved successfully.")

print("--- video_plot_6.py finished ---")