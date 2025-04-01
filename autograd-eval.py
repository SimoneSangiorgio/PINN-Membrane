import torch

# --- Define Input Tensors ---
# Choose specific values for evaluation
a_val = 2.0
b_val = 3.0

# Create tensors. Crucially, set requires_grad=True to track operations
a = torch.tensor([a_val], requires_grad=True, dtype=torch.float32)
b = torch.tensor([b_val], requires_grad=True, dtype=torch.float32)

# --- Define the Function ---
# Use PyTorch operations to build the computation graph
z = (a**2 * b) + (3 * b**3)

print(f"Inputs:")
print(f"  a = {a.item()}")
print(f"  b = {b.item()}")
print(f"Output:")
print(f"  z = {z.item()}") # z = (2^2 * 3) + (3 * 3^3) = (4*3) + (3*27) = 12 + 81 = 93

# --- Compute Gradients using autograd.grad ---
# torch.autograd.grad(outputs, inputs, ...)
# - outputs: The tensor(s) to differentiate (z in this case). Must be scalar or have grad_outputs specified.
# - inputs: A tuple or list of tensors w.r.t. which the gradient is computed (a, b).
# - create_graph: Set to True if you need to compute higher-order derivatives later.
# - allow_unused: Set to True if some inputs might not affect the output (not needed here).

# Since z is a scalar, grad_outputs defaults to torch.tensor(1.0)
gradients = torch.autograd.grad(outputs=z, inputs=(a, b))

# The result 'gradients' is a tuple containing the gradient for each input tensor
grad_a = gradients[0] # Corresponds to ∂z/∂a
grad_b = gradients[1] # Corresponds to ∂z/∂b

print("\n--- Gradients calculated by torch.autograd.grad ---")
print(f"∂z/∂a at (a={a_val}, b={b_val}): {grad_a.item()}")
print(f"∂z/∂b at (a={a_val}, b={b_val}): {grad_b.item()}")

# --- Verification ---
print("\n--- Analytical (Manual) Calculation ---")
analytical_grad_a = 2 * a_val * b_val
analytical_grad_b = a_val**2 + 9 * b_val**2
print(f"Analytical ∂z/∂a at (a={a_val}, b={b_val}): {analytical_grad_a}")
print(f"Analytical ∂z/∂b at (a={a_val}, b={b_val}): {analytical_grad_b}")

# --- Comparison ---
print("\n--- Comparison ---")
print(f"Match for ∂z/∂a? {torch.isclose(grad_a, torch.tensor(analytical_grad_a))}")
print(f"Match for ∂z/∂b? {torch.isclose(grad_b, torch.tensor(analytical_grad_b))}")