from torch.func import jacrev, vmap, hessian, vjp, jvp
import torch

def jacobian(output, input, i=None, j=None, create_graph=True):
    # dimension 0 is the batchsize, so we have to work with all the other dimensions rather
    # than the first one
    # If i == None and j == None, return the full jacobian
    # If, instead i == None and j != None return the j-th colum of the jacobian
    # If i != None and j == None return the i-th row of the jacobian
    # Else return the element (i,j) of the jacobian
    if i == None:
        # compute the full gradient and output it
        grads = []
        for k in range(output.shape[-1]):
            g = torch.zeros_like(output)
            g[..., k] = 1
            # compute the k_th row of the Jacobian
            d_k = torch.autograd.grad(output, input, grad_outputs=g, create_graph=create_graph)[0]
            grads.append(d_k)
        d = torch.stack(grads, dim = 1)
        if j == None:
            return d
        else:
            return d[..., j]
    else:
        g = torch.zeros_like(output)
        g[..., i] = 1
        d = torch.autograd.grad(output, input, grad_outputs=g, create_graph=create_graph)[0]
        if j == None:
            return d
        else:
            return d[..., j]
        

def _jacobian(model, input, i=None, j=None):
    # dimension 0 is the batchsize, so we have to work with all the other dimensions rather
    # than the first one
    # If i == None and j == None, return the full jacobian
    # If, instead i == None and j != None return the j-th colum of the jacobian
    # If i != None and j == None return the i-th row of the jacobian
    # Else return the element (i,j) of the jacobian
    if i == None and j == None:
        # compute the full gradient and output it
        jac_fn = jacrev(model)
        #jac = vmap(jac_fn, in_dims = (0, ))
        return jac_fn(input), jac_fn
    elif j == None:
        output, vjp_fn = vjp(model, input)
        g = torch.zeros_like(output)
        g[..., i] = 1
        return vjp_fn(g), vjp_fn
    else:
        g = torch.zeros_like(input)
        g[..., j] = 1
        output, d = jvp(model, (input, ), (g, ))
        if i == None:
            return d, None
        else:
            return d[...,i], None


def _hessian(model, input, i = None, j = None):
    
    h = hessian(model)
    hes = vmap(h, in_dims = (0, ), randomness="different")
    if i==None and j==None:
        return hes(input)[..., i, j]
    



