class Component:
    def __init__(self, name) -> None:
        self.name = name
    
    def get_name(self):
        return self.name

    def get_params(self):
        return {self.name:{}}
    
    def __str__(self):
        return f"{self.name}: {self.get_params()}"
    

class LossComponent(Component):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.history = []
    
    def compute_loss(self, model, x_in):
        return None