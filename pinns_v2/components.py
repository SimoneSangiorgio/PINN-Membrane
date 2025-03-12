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
    

    