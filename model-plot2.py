# --- START OF FILE model_plot_data.py ---

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import os
import json
import math
# Assuming pinns_v2 package is correctly installed or in PYTHONPATH
# If not, you might need to adjust the import path based on your project structure
try:
    from pinns_v2.model import SimpleSpatioTemporalFFN
except ImportError:
    print("Error: Could not import SimpleSpatioTemporalFFN from pinns_v2.model.")
    print("Ensure the 'pinns_v2' package is installed or accessible in your PYTHONPATH.")
    exit()


# --- !!! DUMMY FUNCTION FOR LOADING ONLY !!! ---
# This function MUST exist with the exact name 'hard_constraint2' before torch.load
# It allows pickle to reconstruct the saved object, even though we won't USE this dummy.
# THIS VERSION MATCHES THE SIGNATURE FROM main_2_inputs.py (takes 2 args)
def hard_constraint2(x_in, y_out):
    """Dummy function to allow loading pickled model object."""
    # print("Warning: Dummy hard_constraint2 called during loading (expected).")
    # It doesn't matter what this returns. Returning the output is safe.
    return y_out
# --- END DUMMY FUNCTION ---

# --- CORRECT Hard Constraint Function (Local & Parameterized) ---
# This is the function we WILL ACTUALLY USE for calculations.
# It mirrors the logic from main_2_inputs.py's hard_constraint2
# but takes parameters explicitly. Assumes x_in_norm has shape (N, 3)
def hard_constraint2_local(x_in_norm, y_out_norm,
                           _t_f, # Add t_f parameter
                           _delta_x, _x_min, _x_max,
                           _delta_y, _y_min, _y_max,
                           _delta_u, _u_min):
    """Applies boundary conditions u(x,y,t)=0 multiplicatively. (Matches main_2_inputs.py logic)"""
    # Add safety checks for division by zero
    if _delta_u == 0:
        print("Warning in hard_constraint2_local: delta_u is zero.")
        return torch.zeros_like(y_out_norm)
    if _delta_x == 0:
        print("Warning in hard_constraint2_local: delta_x is zero.")
        # If delta_x is zero, the term (x_phys - x_max) * (x_phys - x_min) will be zero.
        return torch.zeros_like(y_out_norm)
    if _delta_y == 0:
        print("Warning in hard_constraint2_local: delta_y is zero.")
        # If delta_y is zero, the term (y_phys - y_max) * (y_phys - y_min) will be zero.
        return torch.zeros_like(y_out_norm)

    # Extract normalized coordinates
    X_norm = x_in_norm[:, 0:1] # Shape (N, 1)
    Y_norm = x_in_norm[:, 1:2] # Shape (N, 1)
    # tau_norm = x_in_norm[:, 2:3] # Not needed for this specific constraint factor

    # Convert to physical coordinates
    x_phys = X_norm * _delta_x + _x_min
    y_phys = Y_norm * _delta_y + _y_min
    # t_phys = tau_norm * _t_f # Not needed for this specific constraint factor

    # Convert raw normalized output to physical u
    # Note: The hard constraint in main_2_inputs.py applies the factor *after*
    # converting y_out_norm to u_phys. We replicate that here.
    u_phys_raw = y_out_norm * _delta_u + _u_min

    # Calculate the boundary factor (spatial part only for hard_constraint2)
    boundary_factor = (x_phys - _x_max) * (x_phys - _x_min) * (y_phys - _y_max) * (y_phys - _y_min)

    # Apply the constraint multiplicatively to the physical u
    u_phys_constrained = u_phys_raw * boundary_factor

    # Convert the constrained physical u back to normalized u
    u_constrained_normalized = (u_phys_constrained - _u_min) / _delta_u

    return u_constrained_normalized
# --- End CORRECT Hard Constraint Function ---

# --- Configuration ---
output_name = "1400-update" # <<<--- CONFIRM NAME (Matches the name in the MATLAB script)
base_output_dir = "./output" # Assuming output is in a subdir relative to script
run_output_dir = os.path.join(base_output_dir, output_name)
model_filename = "model_1400.pt" # <<<--- CONFIRM FILENAME
model_path = os.path.join(run_output_dir, "model", model_filename)
params_path = os.path.join(run_output_dir, "params.json")
output_mat_file = f"data_{output_name}.mat" # Output file name matches MATLAB script
n_spatial_steps = 51 # Number of points along x and y axes
num_t_steps = 101    # Number of time steps (Matches MATLAB tlist linspace)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Parameters ---
print(f"Loading parameters from: {params_path}")
if not os.path.exists(params_path):
    print(f"Error: Parameter file not found at {params_path}")
    exit()

try:
    with open(params_path, 'r') as f:
        params_json = json.load(f)
    # Handle potential nesting of parameters under "additional_data" or "additionalData"
    config_params = params_json.get("additional_data", params_json.get("additionalData", params_json))

    # Load parameters matching main_2_inputs.py and needed by hard_constraint2_local
    t_f = float(config_params['t_f'])
    u_min = float(config_params['u_min'])
    u_max = float(config_params['u_max'])
    x_min = float(config_params['x_min'])
    x_max = float(config_params['x_max'])
    y_min = float(config_params['y_min'])
    y_max = float(config_params['y_max'])
    # f_min = float(config_params['f_min']) # Not directly needed for plotting u
    # f_max = float(config_params['f_max']) # Not directly needed for plotting u

    delta_u = u_max - u_min
    delta_x = x_max - x_min
    delta_y = y_max - y_min
    # delta_f = f_max - f_min # Not directly needed

    print(f"Loaded Parameters:")
    print(f"  t_f    = {t_f}")
    print(f"  u_min  = {u_min}, u_max = {u_max}, delta_u = {delta_u}")
    print(f"  x_min  = {x_min}, x_max = {x_max}, delta_x = {delta_x}")
    print(f"  y_min  = {y_min}, y_max = {y_max}, delta_y = {delta_y}")

    if delta_u == 0: print("Error: delta_u is zero."); exit()
    if delta_x == 0: print("Error: delta_x is zero."); exit()
    if delta_y == 0: print("Error: delta_y is zero."); exit()

except KeyError as e:
    print(f"Error: Missing parameter '{e}' in {params_path}")
    exit()
except Exception as e:
    print(f"Error loading or parsing parameters from {params_path}: {e}")
    exit()

# --- Define CORRECT Model Architecture (Matches main_2_inputs.py) ---
# This is the clean architecture we will load the weights into.
print("Instantiating clean model architecture (SimpleSpatioTemporalFFN)...")
try:
    # These parameters MUST match the ones used during training in main_2_inputs.py
    model_clean = SimpleSpatioTemporalFFN(
        spatial_sigmas=[1.0],
        temporal_sigmas=[1.0, 10.0],
        hidden_layers=[200]*3,
        activation=nn.Tanh, # Ensure activation matches training script (Tanh used in example)
        hard_constraint_fn=None # Instantiate WITHOUT constraint ref - IMPORTANT!
    )
    model_clean.to(device) # Move the clean model structure to device
    print("Clean model architecture defined.")
except Exception as e:
    print(f"Error defining model architecture: {e}")
    exit()

# --- Load Weights from Saved File ---
print(f"Loading model weights from: {model_path}")
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

try:
    # Load the potentially problematic full object. The dummy hard_constraint2 allows this.
    # Make sure the DUMMY function exists before this line!
    # Ensure map_location sends tensors to the desired device during loading
    loaded_full_object = torch.load(model_path, map_location=device)
    print("Successfully loaded the saved object (might be full model).")

    # Extract the state_dict from the loaded object
    if hasattr(loaded_full_object, 'state_dict') and callable(loaded_full_object.state_dict):
        # If it's a model object (common case)
        state_dict = loaded_full_object.state_dict()
        print("Extracted state_dict from loaded model object.")
        # Optional: Delete the loaded object now we have the weights
        del loaded_full_object
    elif isinstance(loaded_full_object, dict):
         # If the saved file was already just a state_dict
         state_dict = loaded_full_object
         print("Loaded file was already a state_dict.")
    else:
        raise TypeError("Loaded object is not a model instance or a state_dict.")

    # Load the extracted weights into our CLEAN architecture
    model_clean.load_state_dict(state_dict)
    model_clean.eval() # Set the CLEAN model to evaluation mode
    print("Weights loaded into clean architecture. Model ready for inference.")

    # Assign the clean model to the variable name we use later
    model = model_clean # Use the clean model from now on

except FileNotFoundError:
     print(f"Error: Model file not found at {model_path}")
     exit()
except Exception as e:
    print(f"An error occurred during model loading or state_dict processing: {e}")
    print("Ensure:")
    print("  - The DUMMY 'hard_constraint2' function exists before torch.load.")
    print(f"  - The model architecture defined here matches the saved model '{model_filename}'.")
    print(f"  - The file '{model_path}' is a valid PyTorch model or state_dict.")
    exit()

# --- Generate Evaluation Grid ---
print(f"Generating evaluation grid ({n_spatial_steps}x{n_spatial_steps} spatial, {num_t_steps} temporal)...")
# Normalized coordinates (0 to 1)
x_coords_norm_1d = np.linspace(0, 1, n_spatial_steps, dtype=np.float32)
y_coords_norm_1d = np.linspace(0, 1, n_spatial_steps, dtype=np.float32)
X_grid_norm, Y_grid_norm = np.meshgrid(x_coords_norm_1d, y_coords_norm_1d, indexing='xy') # Match MATLAB's meshgrid default

# Flatten spatial grid for batch processing
x_pinn_norm_flat = X_grid_norm.flatten() # Shape: (n_spatial_steps*n_spatial_steps,)
y_pinn_norm_flat = Y_grid_norm.flatten() # Shape: (n_spatial_steps*n_spatial_steps,)
spatial_points_norm_np = np.stack([x_pinn_norm_flat, y_pinn_norm_flat], axis=-1) # Shape: (N_spatial, 2)
num_spatial_points = spatial_points_norm_np.shape[0]

# Physical time coordinates (matches MATLAB tlist)
# The MATLAB script generates tlist from 0 to 5 with n steps. Let's use t_f from params.
# t_final_matlab = 5.0 # As seen in MATLAB linspace(0, 5, n)
# print(f"Generating time steps from 0 to {t_final_matlab} based on MATLAB script.")
# t_coords_phys = np.linspace(0, t_final_matlab, num_t_steps, dtype=np.float32)
# ---- Let's stick to t_f from params.json for consistency with training ----
print(f"Generating time steps from 0 to {t_f} based on loaded params.")
t_coords_phys = np.linspace(0, t_f, num_t_steps, dtype=np.float32)


print(f"Grid generated with {num_spatial_points} spatial points per time step.")

# --- Run Inference using the CLEAN model ---
print(f"Running inference...")
all_predictions_physical = [] # Store results in physical units

# Prepare spatial points tensor once
spatial_points_norm_tensor = torch.from_numpy(spatial_points_norm_np).to(device)

with torch.no_grad(): # Disable gradient calculations for inference
    for i, t_phys in enumerate(t_coords_phys):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Time step {i+1}/{num_t_steps} (t={t_phys:.4f})")

        # Normalize current time step
        tau_norm_val = t_phys / t_f if t_f != 0 else 0.0
        # Create time column tensor matching spatial points batch size
        t_col_norm = torch.full((num_spatial_points, 1), tau_norm_val, dtype=torch.float32, device=device)

        # Combine spatial and temporal coordinates for model input
        # Input shape: (num_spatial_points, 3) -> (x_norm, y_norm, t_norm)
        input_tensor_normalized = torch.cat([spatial_points_norm_tensor, t_col_norm], dim=1)

        # 1. Get RAW output from the network (normalized)
        # Ensure model is on the correct device (done during loading)
        output_raw_normalized = model(input_tensor_normalized) # Shape: (N_spatial, 1)

        # 2. Apply the CORRECT LOCAL hard constraint function
        # This function takes normalized input/output and parameters, returns normalized constrained output
        output_constrained_normalized = hard_constraint2_local(
            input_tensor_normalized, output_raw_normalized,
            t_f, delta_x, x_min, x_max, delta_y, y_min, y_max, delta_u, u_min
        ) # Shape: (N_spatial, 1)

        # 3. Denormalize the constrained output to get PHYSICAL values
        output_physical = output_constrained_normalized * delta_u + u_min # Shape: (N_spatial, 1)

        # Store the physical predictions (flattened for this time step)
        # Squeeze to remove the last dimension -> (N_spatial,)
        all_predictions_physical.append(output_physical.squeeze().cpu().numpy())

# Stack predictions along the time axis
# Resulting shape: (num_t_steps, num_spatial_points)
pinn_data_matlab_format = np.stack(all_predictions_physical, axis=0)
print(f"Inference complete. Output shape: {pinn_data_matlab_format.shape}")

# --- Prepare Data for MATLAB ---
print("Preparing data dictionary for MATLAB...")

# Generate the PHYSICAL coordinates corresponding to the *flattened* data points
# These are needed for griddata in MATLAB
x_coords_phys_1d = x_coords_norm_1d * delta_x + x_min
y_coords_phys_1d = y_coords_norm_1d * delta_y + y_min
# Regenerate meshgrid for physical coords IF needed elsewhere, but flatten directly
X_grid_phys, Y_grid_phys = np.meshgrid(x_coords_phys_1d, y_coords_phys_1d, indexing='xy')
x_phys_flat = X_grid_phys.flatten() # Shape: (num_spatial_points,)
y_phys_flat = Y_grid_phys.flatten() # Shape: (num_spatial_points,)

# <<< --- Convert data types to float64 (double) for MATLAB --- >>>
print("Converting data to float64 for MATLAB compatibility...")
pinn_data_matlab_format_double = pinn_data_matlab_format.astype(np.float64)
x_phys_flat_double = x_phys_flat.astype(np.float64)
y_phys_flat_double = y_phys_flat.astype(np.float64)
# Also convert tlist (t_coords_phys) for consistency
t_coords_phys_double = t_coords_phys.astype(np.float64)
# <<< --- End Conversion --- >>>


# Create the dictionary to be saved using the CONVERTED float64 arrays
# Match the variable names expected by the MATLAB script:
# 'pinn_data', 'X_pinn', 'Y_pinn', 'tlist'
matlab_dict = {
    'pinn_data': pinn_data_matlab_format_double, # Shape: (num_t_steps, num_spatial_points)
    'X_pinn': x_phys_flat_double,               # Shape: (num_spatial_points,) - Flattened physical X
    'Y_pinn': y_phys_flat_double,               # Shape: (num_spatial_points,) - Flattened physical Y
    'tlist': t_coords_phys_double,              # Shape: (num_t_steps,) - Physical time steps
    # Optional: Include parameters for convenience in MATLAB
    'params': {
        't_f': t_f, 'u_min': u_min, 'u_max': u_max, 'delta_u': delta_u,
        'x_min': x_min, 'x_max': x_max, 'delta_x': delta_x,
        'y_min': y_min, 'y_max': y_max, 'delta_y': delta_y,
        'n_spatial_steps': n_spatial_steps, 'num_t_steps': num_t_steps,
        'source_script': os.path.basename(__file__),
        'model_path': model_path,
        'run_output_dir': run_output_dir
    }
}

# --- Save .mat file Locally ---
# Save in the current directory where the script is run, or specify a full path
output_mat_path = os.path.join(".", output_mat_file) # Save in script's directory
print(f"Saving data to MATLAB file: {output_mat_path}")
try:
    sio.savemat(output_mat_path, matlab_dict, do_compression=True, oned_as='column') # 'column' is often preferred in MATLAB
    print(f"Data saved successfully to {os.path.abspath(output_mat_path)}")
except Exception as e:
    print(f"Error saving .mat file: {e}")

print("Script finished.")
# --- END OF FILE model_plot_data.py ---