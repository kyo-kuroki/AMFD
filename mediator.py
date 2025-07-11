import torch
import inspect
import itertools
from collections import defaultdict
import torch




def get_qubo(f, arg_shapes, device='cuda:0'):
    device = device if torch.cuda.is_available() else 'cpu'

    # 引数名の取得
    if isinstance(arg_shapes, dict):
        arg_names = list(inspect.signature(f).parameters)
    elif isinstance(arg_shapes, list):
        arg_names = list(range(len(arg_shapes)))
    else:
        raise ValueError('arg_shapes requires type of list or dictionary')

    vars = []
    split_sizes = []
    index_map = {}
    flat_index = 0
    for name in arg_names:
        shape = arg_shapes[name]
        t = torch.zeros(shape, requires_grad=True, device=device)
        vars.append(t)
        split_sizes.append(t.numel())
        for idx in itertools.product(*[range(s) for s in shape]):
            index_map[flat_index] = (name, idx)
            flat_index += 1

    output = f(*vars)
    const = output.item()
    grads = torch.autograd.grad(output, vars, create_graph=True)
    h = torch.cat([g.reshape(-1) for g in grads])

    try:
        Q = torch.func.jacrev(torch.func.grad(f))(*vars)
        n = int(Q.numel() ** 0.5)
        Q = torch.reshape(Q, (n, n)).detach().clone() 
    except Exception as e:
        print(f"making hessian serially due to : {e}")
        def rowwise_hesse(h):
            Q_rows = []
            for i in range(h.numel()):
                grad_i = torch.autograd.grad(h[i], vars, retain_graph=True)
                Q_rows.append(torch.cat([g.reshape(-1) for g in grad_i]).detach())
            return torch.stack(Q_rows).detach() 
        Q = rowwise_hesse(h).requires_grad_(False)

    h = h + torch.diagonal(0.5 * Q)
    Q.fill_diagonal_(0)

    return {
        'const': const,
        'h': h.detach(),
        'Q': Q.detach()
    }, {
        'arg_names': arg_names,
        'split_sizes': split_sizes,
        'arg_shapes': arg_shapes,
        'index_map': index_map
    }



def restore_variables(flat_tensor: torch.Tensor, index_map: dict):
    """
    restore original shape variable
    
    Args:
        flat_tensor (torch.Tensor): 
        index_map (dict): {flat_index: (var_name, index_tuple)} 
    
    Returns:
        dict: {var_name: tensor of original shape}
    """
    var_shapes = defaultdict(lambda: set())
    for _, (var_name, idx_tuple) in index_map.items():
        var_shapes[var_name].add(idx_tuple)

    var_tensors = {}
    for var, indices in var_shapes.items():
        shape = tuple(max(id[i] for id in indices) + 1 for i in range(len(next(iter(indices)))))
        var_tensors[var] = torch.zeros(shape, dtype=flat_tensor.dtype)

    for flat_i, (var_name, idx) in index_map.items():
        var_tensors[var_name][idx] = flat_tensor[flat_i]

    return var_tensors




