from pinns_v2.loss import ResidualLoss, TimeCausalityLoss, SupervisedDomainLoss, ICLoss
from pinns_v2.common import Component
import torch
import numpy as np

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
    def __init__(self, pde_fn, ic_fns, dataset, ic_dataset, update_freq=1000, device=None):
        super().__init__("NTKAdaptiveWave")
        self.pde_fn = pde_fn
        self.ic_fns = ic_fns  # Now contains BOTH [ic_fn_u, ic_fn_vel]
        self.dataset = dataset
        self.ic_dataset = ic_dataset
        self.update_freq = update_freq
        self.step_count = 0
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.residual_iterator = iter(dataset)
        self.ic_iterator = iter(ic_dataset)
        
        # Separate losses for position and velocity ICs
        self.ic_u_loss = ICLoss(ic_fns[0])  # Position
        self.ic_ut_loss = ICLoss(ic_fns[1])  # Velocity
        self.pde_loss = ResidualLoss(pde_fn)
        
        # Adaptive weights for ALL THREE terms
        self.lambda_u = torch.tensor(1.0, device=self.device)
        self.lambda_ut = torch.tensor(1.0, device=self.device)
        self.lambda_r = torch.tensor(1.0, device=self.device)

    def compute_ntk_trace(self, model, loss_fn, dataset_type='residual'):
        """Compute NTK trace for a specific loss component"""
        try:
            x_in = next(self.residual_iterator if dataset_type == 'residual' else self.ic_iterator)
        except StopIteration:
            # Reset iterator
            self.residual_iterator = iter(self.dataset) if dataset_type == 'residual' else iter(self.ic_dataset)
            x_in = next(self.residual_iterator)
        x_in = torch.tensor(x_in, device=self.device, dtype=torch.float32).requires_grad_(True)
        loss = loss_fn.compute_loss(model, x_in)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        return sum(torch.sum(g**2) for g in grads)

    def update_weights(self, model):
        """Update λ weights using NTK traces (Algorithm 1 in paper)"""
        with torch.enable_grad():
            print("Updating weights")
            trace_u = self.compute_ntk_trace(model, self.ic_u_loss, 'ic')
            trace_ut = self.compute_ntk_trace(model, self.ic_ut_loss, 'ic')
            trace_r = self.compute_ntk_trace(model, self.pde_loss, 'residual')
            
            total_trace =  trace_ut + trace_r + trace_u
            
            # Stabilize division
            self.lambda_u = (total_trace / (trace_u + 1e-8)).detach()
            self.lambda_ut = (total_trace / (trace_ut + 1e-8)).detach()
            self.lambda_r = (total_trace / (trace_r + 1e-8)).detach()

    def apply(self, model):
        # Fetch batches
        x_res = torch.Tensor(next(self.residual_iterator)).to(self.device)
        x_ic = torch.Tensor(next(self.ic_iterator)).to(self.device)

        # Update weights periodically
        if self.update_freq > 0 and self.step_count > 0 and self.step_count % self.update_freq == 0:
            self.update_weights(model)

        # Compute all three losses
        loss_u = self.ic_u_loss.compute_loss(model, x_ic)  # Position
        print("loss_u", loss_u)
        loss_ut = self.ic_ut_loss.compute_loss(model, x_ic)  # Velocity
        print("loss_ut", loss_ut)
        loss_r = self.pde_loss.compute_loss(model, x_res)    # Residual
        print("loss_r", loss_r)
        
        print("lambda_u", self.lambda_u)
        print("lambda_ut", self.lambda_ut) 
        print("lambda_r", self.lambda_r)
        # Weighted total loss (Eq. 6.2 in paper)
        total_loss = (
            self.lambda_u.detach() * loss_u +
            self.lambda_ut.detach() * loss_ut +
            self.lambda_r.detach() * loss_r
        )
        
        self.step_count += 1
        return total_loss
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
