from pinns_v2.loss import ResidualLoss, TimeCausalityLoss, SupervisedDomainLoss, ICLoss
from pinns_v2.common import Component
import torch


class ComponentManager(Component):
    def __init__(self) -> None:
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
        residual = self.search("Residual", train)
        if residual == None:
            residual = self.search("Residual", like=True, train = True)
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
    def __init__(self, pde_fn, ic_fn, dataset, update_freq=1000, ntk_batch_size = 100, device=None):
        super().__init__("NTKAdaptiveWave")
        self.pde_fn = pde_fn
        self.ic_fn = ic_fn
        self.dataset = dataset
        self.update_freq = update_freq
        self.step_count = 0
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iterator = iter(dataset)
        # Initialize adaptive weights
        self.lambda_u = torch.tensor(1.0, device=self.device)
        self.lambda_ut = torch.tensor(1.0, device=self.device)
        self.lambda_r = torch.tensor(1.0, device=self.device)
        self.ntk_batch_size = ntk_batch_size

    def compute_ntk_trace(self, model, loss_fn, inputs):
        print(f"\n[DEBUG NTK] Input shape: {inputs.shape}")  # <-- ADD THIS
        print(f"[DEBUG NTK] Model parameters: {sum(p.numel() for p in model.parameters())}")  # <-- ADD THIS
        
        params = list(model.parameters())
        idx = torch.randperm(inputs.shape[0])[:self.ntk_batch_size]
        inputs = inputs[idx]
        print(f"[DEBUG NTK] Using subset shape: {inputs.shape}")  # <-- ADD THIS
        """Compute NTK trace using Hutchinson's estimator with memory optimization"""
        params = list(model.parameters())
        
        # Use a small random subset for NTK computation
        idx = torch.randperm(inputs.shape[0])[:self.ntk_batch_size]
        inputs = inputs[idx].detach().requires_grad_(True)
        
        # Compute loss for the small batch
        with torch.no_grad():  # Don't need gradients for the loss computation itself
            loss = loss_fn(model, inputs)
        
        # Compute gradients per parameter to save memory
        traces = []
        for param in params:
            grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)[0]
            traces.append(torch.sum(grad**2))
        
        return torch.mean(torch.stack(traces))

    def update_weights(self, model, x_u, x_ut, x_r):
        """Update weights based on NTK traces with memory optimization"""
        with torch.enable_grad():
            # Compute traces in no_grad context to prevent memory buildup
            with torch.no_grad():
                trace_u = self.compute_ntk_trace(model, self.ic_loss, x_u)
                trace_ut = self.compute_ntk_trace(model, self.ic_velocity_loss, x_ut)
                trace_r = self.compute_ntk_trace(model, self.pde_loss, x_r)
                
                total_trace = trace_u + trace_ut + trace_r
                
                # Update weights with stability epsilon
                self.lambda_u = (total_trace / (trace_u + 1e-8)).detach()
                self.lambda_ut = (total_trace / (trace_ut + 1e-8)).detach()
                self.lambda_r = (total_trace / (trace_r + 1e-8)).detach()

    def ic_loss(self, model, x):
        """Initial condition loss (NO requires_grad_() calls)"""
        u_pred = model(x)
        u_true = torch.zeros_like(u_pred)
        return torch.mean((u_pred - u_true)**2)

    def ic_velocity_loss(self, model, x):
        """Initial velocity loss with proper gradient tracking"""
        x.requires_grad_(True)  # Ensure gradients are enabled
        u_pred = model(x)
        
        # Compute time derivative
        dt = torch.autograd.grad(
            outputs=u_pred,
            inputs=x,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True
        )[0][:, -1:]  # Take the time derivative (last dimension)
        
        return torch.mean(dt**2)
    
    def pde_loss(self, model, x):
        """Wave equation residual loss"""
        residual = self.pde_fn(model, x)
        return torch.mean(residual**2)

    def apply(self, model):
        # Get the next batch from the iterator
        x_in = next(self.iterator)
        
        # Convert numpy array to torch tensor if needed and set requires_grad
        if isinstance(x_in, np.ndarray):
            x_in = torch.tensor(x_in, dtype=torch.float32, requires_grad=True)
        else:
            x_in = x_in.clone().detach().requires_grad_(True)
        
        # Move to device
        x_in = x_in.to(self.device)
        
        # For NTK computation - use a subset
        ntk_batch = x_in[:self.ntk_batch_size]
        
        # Update weights if needed
        if self.step_count % self.update_freq == 0 and self.step_count > 0:
            self.update_weights(model, ntk_batch, ntk_batch, ntk_batch)
        
        # Compute losses
        loss_u = self.ic_loss(model, x_in)
        loss_ut = self.ic_velocity_loss(model, x_in)
        loss_r = self.pde_loss(model, x_in)

        total_loss = (
            self.lambda_u * loss_u +
            self.lambda_ut * loss_ut +
            self.lambda_r * loss_r
        )
        
        self.step_count += 1
        return total_loss
