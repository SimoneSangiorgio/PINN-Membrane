import torch
import torch.nn as nn
import numpy as np
from torch.func import vmap
from functools import partial

from pinns_v2.common import LossComponent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualLoss(LossComponent):
    def __init__(self, pde_fn) -> None:
        super().__init__("ResidualLoss")
        self.pde_fn = pde_fn
    
    def _residual_loss(self, model, pde_fn, x_in):
        r = pde_fn(model, x_in)
        pde_loss = (r**2)
        return pde_loss

    def _compute_loss_r(self, model, pde_fn, x_in):
        r_pred = vmap(partial(self._residual_loss, model, pde_fn), (0), randomness="different")(x_in)
        pde_loss = torch.mean(r_pred)  
        return pde_loss

    def compute_loss(self, model, x_in):
        pde_loss = self._compute_loss_r(model, self.pde_fn, x_in)
        self.history.append(pde_loss.item())
        return pde_loss

  
class ICLoss(LossComponent):
    def __init__(self, ic_fn) -> None:
        super().__init__("ICLoss")
        self.ic_fn = ic_fn
    
    def _ic_loss(self, model, ic_fn, x_in):
        u, true = ic_fn(model, x_in)
        loss_ic = ((u.flatten() - true.flatten())**2)
        return loss_ic

    def _compute_loss_ic(self, model, ic_fn, x_in):
        r_pred = vmap(partial(self._ic_loss, model, ic_fn), (0), randomness="different")(x_in)
        pde_loss = torch.mean(r_pred)
        return pde_loss

    def compute_loss(self, model, x_in):
        ic_loss = self._compute_loss_ic(model, self.ic_fn, x_in)
        self.history.append(ic_loss.item())
        return ic_loss


class TimeCausalityLoss(LossComponent):
    def __init__(self, pde_fn, eps_time, bucket_size) -> None:
        super().__init__("TimeCausality")
        self.eps_time = eps_time
        self.bucket_size = bucket_size
        self.pde_fn = pde_fn
    
    def get_params(self):
        return {"eps_time": self.eps_time, "bucket_size": self.bucket_size}
    
    def _residual_loss(self, model, pde_fn, x_in):
        r = pde_fn(model, x_in)
        pde_loss = torch.mean(r**2)
        return pde_loss

    def _compute_loss_r_time_causality(self, model, pde_fn, bucket_size, eps_time, x_in):
        r_pred = vmap(partial(self._residual_loss, model, pde_fn), (0), randomness="different")(x_in)
        r_pred = r_pred.reshape(bucket_size, -1)
        pde_loss_t = torch.mean(r_pred, axis = 1)
        with torch.no_grad():
            M = np.triu(np.ones((bucket_size, bucket_size)), k=1).T
            M = torch.Tensor(M).to(device)
            W = torch.exp(- eps_time * (M @ pde_loss_t))
        return W, pde_loss_t

    def compute_loss(self, model, x_in):
        W, pde_loss_t = self._compute_loss_r_time_causality(model, self.pde_fn, self.bucket_size, self.eps_time, x_in)
        loss = torch.mean(W*pde_loss_t)
        self.history.append(loss.item())
        with torch.no_grad():
            index = torch.max(torch.nonzero(W)).cpu()
            print(f"Current index: {index} \t Value: {W[index]}")
        return loss


class SupervisedDomainLoss(LossComponent):
    def __init__(self) -> None:
        super().__init__("SupervisedDomainLoss")
    
    def _dsd_loss(self, model, u_true, x_in):
        u = model(x_in)
        loss_dsd = torch.mean((u.flatten() - u_true.flatten())**2)
        return loss_dsd
    
    def _compute_loss_dsd(self, model, x_in):
        splitted_dataset = torch.hsplit(x_in, [x_in.shape[1] - 1])
        x_in = splitted_dataset[0]
        u_true = splitted_dataset[1]
        loss_dsd = vmap(partial(self._dsd_loss, model, u_true), (0), randomness="different")(x_in)
        loss_dsd = torch.mean(loss_dsd)
        return loss_dsd

    def compute_loss(self, model, x_in):
        dsd_loss = self._compute_loss_dsd(model, x_in)
        self.history.append(dsd_loss.item())
        return dsd_loss
