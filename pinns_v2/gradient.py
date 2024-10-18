from torch.func import jacrev, vmap, hessian, vjp, jvp
import torch

def _jacobian(model, input, i=None, j=None):
    if i is None and j is None:
        jac_fn = jacrev(model)
        jac = jac_fn(input)
        if jac.dim() == 1:
            jac = jac.unsqueeze(0)
        return jac, jac_fn
    elif j is None:
        output, vjp_fn = vjp(model, input)
        g = torch.zeros_like(output)
        g[..., i] = 1
        jac = vjp_fn(g)[0]
        if jac.dim() == 1:
            jac = jac.unsqueeze(0)
        return jac, vjp_fn
    else:
        g = torch.zeros_like(input)
        g[..., j] = 1
        output, d = jvp(model, (input,), (g,))
        if i is None:
            if d.dim() == 1:
                d = d.unsqueeze(0)
            return d, None
        else:
            return d[..., i].unsqueeze(0), None
        




def _hessian(model, input, i = None, j = None):
    
    h = hessian(model)
    hes = vmap(h, in_dims = (0, ))
    if i==None and j==None:
        return hes(input)[..., i, j]
    




