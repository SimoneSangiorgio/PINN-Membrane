from pinns_v2.loss import ResidualLoss, TimeCausalityLoss, SupervisedDomainLoss, ICLoss
from pinns_v2.common import Component
import torch
import numpy as np

class ComponentManager(Component):
    def __init__(self) -> None:
        super().__init__("ComponentManager")
        self._component_list_train = []
        self._component_list_valid = []

    def add_train_component(self, component:Component):
        self._component_list_train.append(component)
    
    def add_validation_component(self, component:Component):
        self._component_list_valid.append(component)

    def get_params(self):
        p = []
        for component in self._component_list_train:
            p.append(component.get_params())
        q = []
        for component in self._component_list_valid:
            q.append(component.get_params())
        return {"Training Components": p, "Validation Components": q}

    def apply(self, model, train = True):
        loss = 0
        if train:
            for elem in self._component_list_train:
                loss += elem.apply(model)
        else:
            for elem in self._component_list_valid:
                loss += elem.apply(model)
        return loss

    def search(self, name, like = False, train = True):
        if train:
            for elem in self._component_list_train:
                if like :
                    if name in elem.name:
                        return elem
                else:
                    if elem.name == name:
                        return elem
            return None
        else:
            for elem in self._component_list_valid:
                if like :
                    if name in elem.name:
                        return elem
                else:
                    if elem.name == name:
                        return elem
            return None
            
    def number_of_iterations(self, train = True):
        residual = self.search("NTKAdaptiveWave", train)
        if residual == None:
            residual = self.search("NTKAdaptiveWave", like=True, train = True)
        return len(residual.dataset)
    

class ResidualComponent(Component):
    def __init__(self, pde_fn, dataset, device = None) -> None:
        super().__init__("Residual")
        self.pde_fn = pde_fn
        self.dataset = dataset
        self.loss = ResidualLoss(self.pde_fn)
        self.iterator = iter(dataset)
        self.device = device if device != None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def apply(self, model):
        x_in = next(self.iterator)
        x_in = torch.Tensor(x_in).to(self.device)
        loss = self.loss.compute_loss(model, x_in)
        return loss

    def get_params(self):
        return {self.name: self.loss.get_params()}

class ResidualTimeCausalityComponent(Component):
    def __init__(self, pde_fn, dataset, eps_time, number_of_buckets=10, device = None) -> None:
        super().__init__("ResidualTimeCausality")
        self.pde_fn = pde_fn
        self.dataset = dataset
        self.loss = TimeCausalityLoss(self.pde_fn, eps_time, number_of_buckets)
        self.iterator = iter(dataset)
        self.device = device if device != None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def apply(self, model):
        x_in = next(self.iterator)
        x_in = torch.Tensor(x_in).to(self.device)
        loss = self.loss.compute_loss(model, x_in)
        return loss

    def get_params(self):
        return {self.name: self.loss.get_params()}
    
class ICComponent(Component):
    def __init__(self, ic_fns, dataset, device=None) -> None:
        super().__init__("IC")
        self.ic_fns = ic_fns
        self.dataset = dataset
        self.loss = []
        for fn in ic_fns:
            self.loss.append(ICLoss(fn))
        self.iterator = iter(dataset)
        self.device = device if device != None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def apply(self, model):
        x_in = next(self.iterator)
        x_in = torch.Tensor(x_in).to(self.device)
        loss = 0
        for l in self.loss:
            loss += l.compute_loss(model, x_in)
        return loss

    def get_params(self):
        p = []
        for el in self.loss:
            p.append(el.get_params())
        return {self.name: p}
    
class SupervisedComponent(Component):
    def __init__(self, dataset, device = None) -> None:
        super().__init__("Supervised")
        self.dataset = dataset
        self.loss = SupervisedDomainLoss()
        self.iterator = iter(dataset)
        self.device = device if device != None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def apply(self, model):
        x_in = next(self.iterator)
        x_in = torch.Tensor(x_in).to(self.device)
        loss = self.loss.compute_loss(model, x_in)
        return loss

    def get_params(self):
        return {self.name: self.loss.get_params()}

class NTKAdaptiveWaveComponent(Component):
    def __init__(self, pde_fn, ic_fns, dataset, ic_dataset,
                 update_freq=1000, device=None,
                 min_lambda=0.01, max_lambda=100.0): # Added clipping parameters
        super().__init__("NTKAdaptiveWave")
        self.pde_fn = pde_fn
        # Ensure ic_fns is a list or tuple [ic_fn_u, ic_fn_vel]
        if not isinstance(ic_fns, (list, tuple)) or len(ic_fns) != 2:
            raise ValueError("ic_fns must be a list or tuple containing two functions: [ic_fn_u, ic_fn_vel]")
        self.ic_fns = ic_fns
        self.dataset = dataset
        self.ic_dataset = ic_dataset
        self.update_freq = update_freq
        self.step_count = 0
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.residual_iterator = iter(dataset)
        self.ic_iterator = iter(ic_dataset)

        # Loss calculation objects
        self.ic_u_loss = ICLoss(ic_fns[0])  # Position IC loss calculator
        self.ic_ut_loss = ICLoss(ic_fns[1]) # Velocity IC loss calculator
        self.pde_loss = ResidualLoss(pde_fn)   # Residual loss calculator

        # Adaptive weights
        self.lambda_u = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self.lambda_ut = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self.lambda_r = torch.tensor(1.0, device=self.device, dtype=torch.float32)

        # Clipping parameters
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        # Loggers (optional)
        self.lambda_u_log = []
        self.lambda_ut_log = []
        self.lambda_r_log = []

    def _compute_ntk_trace_approximation(self, model, loss_compute_fn, x_in):
        """
        Computes an approximation of the NTK trace using sum of squared gradients
        of the provided loss computation function.
        """
        model.zero_grad() # Zero gradients before computation

        # Ensure input requires grad for gradient calculation w.r.t parameters
        # Need to detach and require grad if x_in comes from iterator directly
        x_in_grad = x_in.detach().requires_grad_(True)

        # Compute the loss using the provided function (e.g., self.ic_u_loss.compute_loss)
        loss = loss_compute_fn(model, x_in_grad)

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
             print(f"Warning: Invalid loss ({loss}) encountered during trace computation. Returning zero trace.")
             return torch.tensor(0.0, device=self.device)

        # Compute gradients of the loss w.r.t. model parameters
        try:
            grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
        except RuntimeError as e:
            print(f"Warning: Error during gradient computation for trace: {e}. Returning zero trace.")
            return torch.tensor(0.0, device=self.device)


        # Calculate the sum of squared gradients (approximates trace)
        trace = torch.tensor(0.0, device=self.device)
        count = 0
        for g in grads:
            if g is not None:
                trace += torch.sum(g**2)
                count += 1
        
        if count == 0:
             print("Warning: No gradients computed for trace calculation. Returning zero trace.")
             return torch.tensor(0.0, device=self.device)
             
        return trace

    def update_weights(self, model):
        """Update lambda weights using NTK trace approximations and apply clipping."""
        model.eval() # Use eval mode for consistent trace calculation
        with torch.enable_grad(): # Still need gradients for the trace calc
            # Get a batch of data for each loss component for trace calculation
            # It's often okay to reuse the main iterators if batch sizes are consistent
            # Or use separate iterators if needed
            try:
                x_res = next(self.residual_iterator)
                x_res = torch.tensor(x_res, device=self.device, dtype=torch.float32)
            except StopIteration:
                print("Warning: Residual iterator stopped during weight update. Reinitializing.")
                self.residual_iterator = iter(self.dataset)
                x_res = torch.tensor(next(self.residual_iterator), device=self.device, dtype=torch.float32)
            try:
                x_ic = next(self.ic_iterator)
                x_ic = torch.tensor(x_ic, device=self.device, dtype=torch.float32)
            except StopIteration:
                print("Warning: IC iterator stopped during weight update. Reinitializing.")
                self.ic_iterator = iter(self.ic_dataset)
                x_ic = torch.tensor(next(self.ic_iterator), device=self.device, dtype=torch.float32)


            # Compute NTK trace approximations
            trace_u = self._compute_ntk_trace_approximation(model, self.ic_u_loss.compute_loss, x_ic)
            trace_ut = self._compute_ntk_trace_approximation(model, self.ic_ut_loss.compute_loss, x_ic)
            trace_r = self._compute_ntk_trace_approximation(model, self.pde_loss.compute_loss, x_res)


            # --- Weight Update Logic ---
            # Avoid division by zero or very small traces which cause explosion
            eps = 1e-8
            if trace_u.abs() < eps or trace_ut.abs() < eps or trace_r.abs() < eps or torch.isnan(trace_u) or torch.isnan(trace_ut) or torch.isnan(trace_r):
                print(f"Warning: Very small or invalid trace encountered (u:{trace_u:.2e}, ut:{trace_ut:.2e}, r:{trace_r:.2e}). Skipping weight update.")
                model.train() # Set back to training mode
                return # Skip update

            total_trace = trace_u + trace_ut + trace_r

            # Calculate raw new lambdas
            new_lambda_u = total_trace / (trace_u + eps) # Add eps for safety
            new_lambda_ut = total_trace / (trace_ut + eps)
            new_lambda_r = total_trace / (trace_r + eps)

            # <<< Apply Clipping >>>
            self.lambda_u = torch.clamp(new_lambda_u, min=self.min_lambda, max=self.max_lambda).detach()
            self.lambda_ut = torch.clamp(new_lambda_ut, min=self.min_lambda, max=self.max_lambda).detach()
            self.lambda_r = torch.clamp(new_lambda_r, min=self.min_lambda, max=self.max_lambda).detach()

            # Log if needed
            self.lambda_u_log.append(self.lambda_u.item())
            self.lambda_ut_log.append(self.lambda_ut.item())
            self.lambda_r_log.append(self.lambda_r.item())

            print(f"Trace approx: u={trace_u:.2e}, ut={trace_ut:.2e}, r={trace_r:.2e}")
            print(f"Weights updated: lu={self.lambda_u:.2e}, lut={self.lambda_ut:.2e}, lr={self.lambda_r:.2e} (clipped [{self.min_lambda}, {self.max_lambda}])")

        model.train() # Set model back to training mode


    def apply(self, model):
        """Computes the weighted loss for training."""
        # Get training batches (use main iterators)
        try:
            x_res = next(self.residual_iterator)
        except StopIteration:
            self.residual_iterator = iter(self.dataset)
            x_res = next(self.residual_iterator)
        try:
            x_ic = next(self.ic_iterator)
        except StopIteration:
            self.ic_iterator = iter(self.ic_dataset)
            x_ic = next(self.ic_iterator)

        x_res = torch.tensor(x_res, device=self.device, dtype=torch.float32)
        x_ic = torch.tensor(x_ic, device=self.device, dtype=torch.float32)

        # Update weights periodically
        # It's important this happens *before* computing the loss for the current step's backprop
        if self.update_freq > 0 and self.step_count > 0 and self.step_count % self.update_freq == 0:
            print(f"\n--- Updating weights at step {self.step_count} ---")
            self.update_weights(model) # This now includes clipping
            print(f"--- Weight update complete ---\n")

        # Compute individual losses using the loss objects and training data
        model.train() # Ensure model is in train mode
        loss_u = self.ic_u_loss.compute_loss(model, x_ic)
        loss_ut = self.ic_ut_loss.compute_loss(model, x_ic)
        loss_r = self.pde_loss.compute_loss(model, x_res)

        # Check for invalid loss values before weighting
        if loss_u is None or torch.isnan(loss_u) or torch.isinf(loss_u) or \
           loss_ut is None or torch.isnan(loss_ut) or torch.isinf(loss_ut) or \
           loss_r is None or torch.isnan(loss_r) or torch.isinf(loss_r):
           print(f"Warning: Invalid loss component detected (u:{loss_u}, ut:{loss_ut}, r:{loss_r}). Returning 0 loss for this step.")
           self.step_count += 1
           return torch.tensor(0.0, device=self.device, requires_grad=True) # Return zero loss that requires grad


        # Weighted total loss (Eq. 6.2 in paper)
        # Use the possibly clipped, detached lambda values
        total_loss = (
            self.lambda_u * loss_u +
            self.lambda_ut * loss_ut +
            self.lambda_r * loss_r
        )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
             print(f"Warning: NaN or Inf total loss detected after weighting. Returning 0 loss.")
             total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)


        self.step_count += 1
        return total_loss

    # Add this method to get component parameters if needed for logging/saving
    def get_params(self):
         # Start with component-specific parameters
         component_params = {
             "update_freq": self.update_freq,
             "min_lambda": self.min_lambda,
             "max_lambda": self.max_lambda,
             "current_lambda_u": self.lambda_u.item(),
             "current_lambda_ut": self.lambda_ut.item(),
             "current_lambda_r": self.lambda_r.item(),
             # Add parameters/history from internal loss functions
             # These will call the get_params we defined in loss.py
             "Internal_IC_U_Loss": self.ic_u_loss.get_params(),
             "Internal_IC_UT_Loss": self.ic_ut_loss.get_params(),
             "Internal_PDE_Loss": self.pde_loss.get_params()
             # Note: This nests the {LossName: {params}} structure
             # inside the NTK component's parameters.
         }
         # Return in the standard {ComponentName: {params...}} format
         return {self.get_name(): component_params} # Use get_name() from base class
# class NTKAdaptiveWaveComponent(Component):
#     def __init__(self, pde_fn, ic_fns, dataset, ic_dataset, update_freq=1000, device=None):
#         super().__init__("NTKAdaptiveWave")
#         self.pde_fn = pde_fn
#         self.ic_fns = ic_fns  # Now contains BOTH [ic_fn_u, ic_fn_vel]
#         self.dataset = dataset
#         self.ic_dataset = ic_dataset
#         self.update_freq = update_freq
#         self.step_count = 0
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.residual_iterator = iter(dataset)
#         self.ic_iterator = iter(ic_dataset)
        
#         # Separate losses for position and velocity ICs
#         self.ic_u_loss = ICLoss(ic_fns[0])  # Position
#         self.ic_ut_loss = ICLoss(ic_fns[1])  # Velocity
#         self.pde_loss = ResidualLoss(pde_fn)
        
#         # Adaptive weights for ALL THREE terms
#         self.lambda_u = torch.tensor(1.0, device=self.device)
#         self.lambda_ut = torch.tensor(1.0, device=self.device)
#         self.lambda_r = torch.tensor(1.0, device=self.device)

#     def compute_ntk_trace(self, model, loss_fn, dataset_type='residual'):
#         """Compute NTK trace for a specific loss component"""
        
#         x_in = next(self.residual_iterator if dataset_type == 'residual' else self.ic_iterator)
        
#         x_in = torch.tensor(x_in, device=self.device, dtype=torch.float32).requires_grad_(True)
#         loss = loss_fn.compute_loss(model, x_in)
#         grads = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
#         trace = sum(torch.sum(g**2) for g in grads if g is not None)
#         return trace

#     def update_weights(self, model):
#         """Update λ weights using NTK traces (Algorithm 1 in paper)"""
#         with torch.enable_grad():
#             print("Updating weights")
#             trace_u = self.compute_ntk_trace(model, self.ic_u_loss, 'ic')
#             trace_ut = self.compute_ntk_trace(model, self.ic_ut_loss, 'ic')
#             trace_r = self.compute_ntk_trace(model, self.pde_loss, 'residual')
            
#             total_trace =  trace_ut + trace_r + trace_u
            
#             # Stabilize division
#             self.lambda_u = (total_trace / (trace_u + 1e-8)).detach()
#             self.lambda_ut = (total_trace / (trace_ut + 1e-8)).detach()
#             self.lambda_r = (total_trace / (trace_r + 1e-8)).detach()

#     def apply(self, model):
#         # Fetch batches
#         x_res = torch.Tensor(next(self.residual_iterator)).to(self.device)
#         x_ic = torch.Tensor(next(self.ic_iterator)).to(self.device)

#         # Update weights periodically
#         if self.update_freq > 0 and self.step_count > 0 and self.step_count % self.update_freq == 0:
#             self.update_weights(model)

#         # Compute all three losses
#         loss_u = self.ic_u_loss.compute_loss(model, x_ic)  # Position
#         print("loss_u", loss_u)
#         loss_ut = self.ic_ut_loss.compute_loss(model, x_ic)  # Velocity
#         print("loss_ut", loss_ut)
#         loss_r = self.pde_loss.compute_loss(model, x_res)    # Residual
#         print("loss_r", loss_r)
        
#         print("lambda_u", self.lambda_u)
#         print("lambda_ut", self.lambda_ut) 
#         print("lambda_r", self.lambda_r)
#         # Weighted total loss (Eq. 6.2 in paper)
#         total_loss = (
#             self.lambda_u.detach() * loss_u +
#             self.lambda_ut.detach() * loss_ut +
#             self.lambda_r.detach() * loss_r
#         )
        
#         self.step_count += 1
#         return total_loss
# class NTKAdaptiveWaveComponent(Component):
#     def __init__(self, pde_fn, ic_fn, dataset, update_freq=1000, device=None):
#         super().__init__("NTKAdaptiveWave")
#         self.pde_fn = pde_fn
#         self.ic_fn = ic_fn
#         self.dataset = dataset  # Keep original dataset object
#         self.update_freq = update_freq
#         self.step_count = 0
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.iterator = iter(dataset)  # Use dataset iterator like other components
#         self.pde_loss = ResidualLoss(self.pde_fn)
#         # Initialize adaptive weights
#         self.lambda_u = torch.tensor(1.0, device=self.device)
#         self.lambda_ut = torch.tensor(1.0, device=self.device)
#         self.lambda_r = torch.tensor(1.0, device=self.device)

#     def compute_ntk_trace(self, model, loss_fn):
#         """Batch-wise NTK trace computation"""
        
#         try:
#             x_in = next(self.iterator)
#         except StopIteration:
#             self.iterator = iter(self.dataset)
#             x_in = next(self.iterator)
            
#         x_in = torch.tensor(x_in, device=self.device, dtype=torch.float32).requires_grad_(True)
#         loss = loss_fn(model, x_in)
#         grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
#         return sum(torch.sum(g**2) for g in grads)

#     def update_weights(self, model):
#         """Update weights using batch-wise NTK traces"""
#         with torch.enable_grad():
#             trace_u = self.compute_ntk_trace(model, self.ic_loss)
#             trace_ut = self.compute_ntk_trace(model, self.ic_velocity_loss)
#             trace_r = self.compute_ntk_trace(model, self.pde_loss)
            
#             total_trace = trace_u + trace_ut + trace_r
            
#             # Update weights with stability epsilon
#             self.lambda_u = (total_trace / (trace_u + 1e-8)).detach()
#             self.lambda_ut = (total_trace / (trace_ut + 1e-8)).detach()
#             self.lambda_r = (total_trace / (trace_r + 1e-8)).detach()

#     def ic_loss(self, model, inputs):
#         u_pred = model(inputs)
#         u_true = torch.zeros_like(u_pred)
#         return torch.mean((u_pred - u_true)**2)

#     def ic_velocity_loss(self, model, inputs):
#         inputs.requires_grad_(True)
#         u_pred = model(inputs)
#         dt = torch.autograd.grad(u_pred, inputs, grad_outputs=torch.ones_like(u_pred),
#                                create_graph=True)[0][:, -1:]
#         return torch.mean(dt**2)
    


#     def apply(self, model):
#         # Get batch from iterator like ResidualComponent
#         x_in = next(self.iterator)
#         x_in = torch.Tensor(x_in).to(self.device)
#         print(x_in.shape)

#         # Update weights if needed
#         if self.update_freq>0 and self.step_count % self.update_freq == 0 and self.step_count > 0:
#             print("Updating weights")
#             self.update_weights(model)

#         # Compute losses
#         loss_u = self.ic_loss(model, x_in)
#         loss_ut = self.ic_velocity_loss(model, x_in)
#         loss_r = self.pde_loss.compute_loss(model, x_in)
#         total_loss = (self.lambda_u * loss_u +
#                       self.lambda_ut * loss_ut +
#                       self.lambda_r * loss_r)
        
#         self.step_count += 1
#         return total_loss
