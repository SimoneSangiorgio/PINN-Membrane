import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import os
import json
import math
from pinns_v2.model import SimpleSpatioTemporalFFN


# --- !!! DUMMY FUNCTION FOR LOADING ONLY !!! ---
# This function MUST exist with the exact name 'hard_constraint2' before torch.load
# It allows pickle to reconstruct the saved object, even though we won't USE this dummy.
def hard_constraint2(*args, **kwargs):
    """Dummy function to allow loading pickled model object."""
    # print("Warning: Dummy hard_constraint2 called during loading (expected).")
    # It doesn't matter what this returns, or if it does anything.
    # Returning None or zero is safest if it were ever actually called by mistake.
    if args:
        return args[1] # Try to return the second argument (like the original might have)
    return None
# --- END DUMMY FUNCTION ---

# --- CORRECT Hard Constraint Function (Local & Parameterized) ---
# This is the function we WILL ACTUALLY USE for calculations.
def hard_constraint2_local(x_in_norm, y_out_norm,
                           _delta_x, _x_min, _x_max,
                           _delta_y, _y_min, _y_max,
                           _delta_u, _u_min):
    """Applies boundary conditions u(x,y,t)=0 multiplicatively."""
    if _delta_u == 0 or _delta_x == 0 or _delta_y == 0:
        print("Warning in hard_constraint2_local: Delta value is zero.")
        return torch.zeros_like(y_out_norm)
    X_norm = x_in_norm[:, 0:1]; Y_norm = x_in_norm[:, 1:2]
    x_phys = X_norm * _delta_x + _x_min; y_phys = Y_norm * _delta_y + _y_min
    boundary_factor = (x_phys - _x_max) * (x_phys - _x_min) * (y_phys - _y_max) * (y_phys - _y_min)
    u_phys_raw = y_out_norm * _delta_u + _u_min
    u_phys_constrained = u_phys_raw * boundary_factor
    u_constrained_normalized = (u_phys_constrained - _u_min) / _delta_u
    return u_constrained_normalized
# --- End CORRECT Hard Constraint Function ---

# --- Configuration ---
output_name = "1400-update" # <<<--- CONFIRM NAME
base_output_dir = "./output"
run_output_dir = os.path.join(base_output_dir, output_name)
model_filename = "model_1400.pt"
model_path = os.path.join(run_output_dir, "model", model_filename)
params_path = os.path.join(run_output_dir, "params.json")
output_mat_file = f"data_{output_name}.mat"
n_spatial_steps = 51
num_t_steps = 101
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Parameters ---
print(f"Loading parameters from: {params_path}")
if not os.path.exists(params_path): print(f"Error: {params_path} not found."); exit()
try:
    with open(params_path, 'r') as f: params_json = json.load(f)
    config_params = params_json.get("additional_data", params_json)
    t_final = float(config_params.get("t_f", 10.0)); u_min = float(config_params.get("u_min", -1.0))
    u_max = float(config_params.get("u_max", 1.0)); x_min = float(config_params.get("x_min", 0.0))
    x_max = float(config_params.get("x_max", 1.0)); y_min = float(config_params.get("y_min", 0.0))
    y_max = float(config_params.get("y_max", 1.0))
    delta_u = u_max - u_min; delta_x = x_max - x_min; delta_y = y_max - y_min
    print(f"Loaded: t_f={t_final}, u_min={u_min}, u_max={u_max}, delta_u={delta_u}, x_range=[{x_min},{x_max}], y_range=[{y_min},{y_max}]")
    if delta_u == 0 or delta_x == 0 or delta_y == 0: print("Error: Zero range detected."); exit()
except Exception as e: print(f"Error loading params: {e}"); exit()

# --- Define CORRECT Model Architecture ---
# This is the clean architecture we will load the weights into.
print("Instantiating clean model architecture...")
try:
    model_clean = SimpleSpatioTemporalFFN( # Use a different variable name
        spatial_sigmas=[1.0], temporal_sigmas=[1.0, 10.0],
        hidden_layers=[200]*3, activation=nn.Tanh,
        hard_constraint_fn=None # Instantiate WITHOUT constraint ref
    )
    model_clean.to(device) # Move the clean model structure to device
    print("Clean model architecture defined.")
except Exception as e: print(f"Error defining architecture: {e}"); exit()

# --- Load Weights from Saved File ---
print(f"Loading model file (may contain full object): {model_path}")
if not os.path.exists(model_path): print(f"Error: {model_path} not found."); exit()

try:
    # Load the potentially problematic full object. The dummy hard_constraint2 allows this.
    # Make sure the DUMMY function exists before this line!
    loaded_full_object = torch.load(model_path, map_location=device)
    print("Successfully loaded the saved object (might be full model).")

    # Extract the state_dict from the loaded object
    if hasattr(loaded_full_object, 'state_dict'):
        state_dict = loaded_full_object.state_dict()
        print("Extracted state_dict from loaded object.")
        # Optional: Delete the loaded object now we have the weights
        del loaded_full_object
    elif isinstance(loaded_full_object, dict):
         # This case shouldn't happen based on the error, but handle defensively
         state_dict = loaded_full_object
         print("Loaded file was already a state_dict.")
    else:
        raise TypeError("Loaded object is not a model or state_dict.")

    # Load the extracted weights into our CLEAN architecture
    model_clean.load_state_dict(state_dict)
    model_clean.eval() # Set the CLEAN model to evaluation mode
    print("Weights loaded into clean architecture. Model ready for inference.")

    # Assign the clean model to the variable name we use later
    model = model_clean # Use the clean model from now on

except Exception as e:
    print(f"An error occurred during loading or state_dict extraction: {e}")
    print("Ensure the DUMMY 'hard_constraint2' function exists before torch.load.")
    exit()

# --- Generate Evaluation Grid ---
print(f"Generating evaluation grid...")
x_coords_norm_1d = np.linspace(0, 1, n_spatial_steps, dtype=np.float32)
y_coords_norm_1d = np.linspace(0, 1, n_spatial_steps, dtype=np.float32)
X_grid_norm, Y_grid_norm = np.meshgrid(x_coords_norm_1d, y_coords_norm_1d, indexing='xy')
x_pinn_norm_flat = X_grid_norm.flatten(); y_pinn_norm_flat = Y_grid_norm.flatten()
spatial_points_norm_np = np.stack([x_pinn_norm_flat, y_pinn_norm_flat], axis=-1)
num_spatial_points = spatial_points_norm_np.shape[0]
t_coords_phys = np.linspace(0, t_final, num_t_steps, dtype=np.float32)
print(f"Grid generated with {num_spatial_points} spatial points.")

# --- Run Inference using the CLEAN model ---
print(f"Running inference...")
all_predictions_physical = []
with torch.no_grad():
    for i, t_phys in enumerate(t_coords_phys):
        if (i + 1) % 10 == 0 or i == 0: print(f"  Time step {i+1}/{num_t_steps}")
        tau_norm_val = t_phys / t_final if t_final != 0 else 0.0
        t_col_norm = torch.full((num_spatial_points, 1), tau_norm_val, dtype=torch.float32, device=device)
        spatial_points_norm_tensor = torch.from_numpy(spatial_points_norm_np).to(device)
        input_tensor_normalized = torch.cat([spatial_points_norm_tensor, t_col_norm], dim=1)

        # USE THE CLEAN MODEL (which has the loaded weights)
        output_raw_normalized = model(input_tensor_normalized)

        # Apply the CORRECT LOCAL constraint function
        output_constrained_normalized = hard_constraint2_local(
            input_tensor_normalized, output_raw_normalized,
            delta_x, x_min, x_max, delta_y, y_min, y_max, delta_u, u_min
        )
        output_physical = output_constrained_normalized * delta_u + u_min
        all_predictions_physical.append(output_physical.squeeze().cpu().numpy())

pinn_data_matlab_format = np.stack(all_predictions_physical, axis=0)
print(f"Inference complete. Output shape: {pinn_data_matlab_format.shape}")

# --- Prepare Data for MATLAB ---
print("Preparing data dictionary for MATLAB...")
# Generate the PHYSICAL coordinates corresponding to the flattened data points
# These are useful for plotting or analysis directly in MATLAB
x_coords_phys_1d = x_coords_norm_1d * delta_x + x_min
y_coords_phys_1d = y_coords_norm_1d * delta_y + y_min
X_grid_phys, Y_grid_phys = np.meshgrid(x_coords_phys_1d, y_coords_phys_1d, indexing='xy')
x_phys_flat = X_grid_phys.flatten() # Shape: (num_spatial_points,)
y_phys_flat = Y_grid_phys.flatten() # Shape: (num_spatial_points,)

# <<< --- START MODIFICATION --- >>>
# Ensure data types are float64 (double) for MATLAB compatibility
print("Converting data to float64 for MATLAB...")
pinn_data_matlab_format_double = pinn_data_matlab_format.astype(np.float64)
x_phys_flat_double = x_phys_flat.astype(np.float64)
y_phys_flat_double = y_phys_flat.astype(np.float64)
# Also convert tlist for consistency, although it might not be strictly necessary for griddata itself
t_coords_phys_double = t_coords_phys.astype(np.float64)
# <<< --- END MODIFICATION --- >>>


# Create the dictionary to be saved using the CONVERTED arrays
matlab_dict = {
    'pinn_data': pinn_data_matlab_format_double,   # Use double version
    'X_pinn': x_phys_flat_double,                  # Use double version
    'Y_pinn': y_phys_flat_double,                  # Use double version
    'tlist': t_coords_phys_double,                 # Use double version
    # Optional: Include parameters for convenience in MATLAB
    'params': {
        't_f': t_final, 'u_min': u_min, 'u_max': u_max, 'delta_u': delta_u,
        'x_min': x_min, 'x_max': x_max, 'delta_x': delta_x,
        'y_min': y_min, 'y_max': y_max, 'delta_y': delta_y,
        'n_spatial_steps': n_spatial_steps, 'num_t_steps': num_t_steps,
        'source_script': os.path.basename(__file__),
        'model_path': model_path
    }
}

# --- Save .mat file Locally ---
# (The rest of the saving code remains the same)
print(f"Saving data to MATLAB file: {output_mat_file}")
try:
    # Save in the current directory where the script is run
    sio.savemat(output_mat_file, matlab_dict, do_compression=True) # Compression is good for large files
    print(f"Data saved successfully to {os.path.abspath(output_mat_file)}")
except Exception as e:
    print(f"Error saving .mat file: {e}")

print("Script finished.")