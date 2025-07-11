import torch
import tqdm
from typing import Callable, List, Tuple
import copy
from torch.func import hessian, vmap
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import itertools
import contextlib



def squared_norm_and_diag_hessians(f: Callable, *inputs: torch.Tensor, m: int = 128):
    inputs = [torch.zeros_like(x, requires_grad=True) for x in inputs]

    # Forward and 1st-order gradients
    output = f(*inputs)
    if output.numel() != 1:
        raise ValueError("f must return a scalar output.")
    grads = torch.autograd.grad(output, inputs, create_graph=True)

    diag_hessians = []

    for i, x in enumerate(inputs):
        flat_x = x.view(-1)
        n = flat_x.numel()
        diag = torch.empty(n, device=x.device)

        # scalar function: partial over x (the i-th input)
        def scalar_fn(x_partial):
            x_reconstructed = x_partial.view_as(x)
            new_inputs = list(inputs)
            new_inputs[i] = x_reconstructed
            return f(*new_inputs)

        # compute Hessian diagonals m elements at a time
        for j in range(0, n, m):
            idx = slice(j, min(j + m, n))
            x_chunk = flat_x[idx].detach().requires_grad_(True)

            def chunk_fn(chunk):
                x_new = flat_x.clone()
                x_new[idx] = chunk
                return scalar_fn(x_new)

            hess_chunk = hessian(chunk_fn)(x_chunk)  # (m, m)
            diag_chunk = hess_chunk.diagonal(dim1=0, dim2=1)
            diag[idx] = diag_chunk.detach()

        diag_hessians.append(diag.view_as(x))


    # compute Hessian
    try:
        Q_squared_norm = ((hessian(f)(*inputs))**2).sum()
    except Exception as e:
        print(f"Direct computing Hessian was failed: {e}. Computing row-wise.")
        h = torch.cat([g.reshape(-1) for g in grads])
        def row_hesse(h_i):
            grad_i = torch.autograd.grad(h_i, inputs, retain_graph=True)
            return (torch.cat([g.reshape(-1) for g in grad_i])**2).sum().detach().requires_grad_(False)
        # Hessianの各行を1個ずつ計算（メモリ効率よい）
        Q_squared_norm = 0
        for i in range(h.numel()):
            Q_squared_norm += row_hesse(h[i])

    h_squared_norm = sum(((grad + 0.5*diag_hessian)**2).sum() - (diag_hessian**2).sum() for grad, diag_hessian in zip(grads, diag_hessians))

    return h_squared_norm + Q_squared_norm, diag_hessians




def pre_compile(f, shapes, device='cuda:0'):
    auto_grid_amfd(f, shapes, zeta_vals=[0], eta_vals=[0.1], t_st=0.1, t_en=0, num_rep=1, Nstep=2, squared_norm=torch.tensor(1,device=device), diag_hessians=[torch.zeros(shape, device=device) for shape in shapes], device=device, show_progress=False)




def auto_grid_amfd(
    f: Callable,
    shapes: List[torch.Size],
    eta_vals: List[float],
    zeta_vals: List[float],
    num_rep: int,
    t_st=0.4,
    t_en=0.001,
    Nstep=None,
    squared_norm=None,
    diag_hessians=None,
    device='cuda:0',
    show_progress=True
):
    """
    Performs batched AMFD optimization over eta, zeta grid with repetitions.

    Args:
        f: Callable returning scalar value given list of tensors.
        shapes: List of shapes for input tensors.
        eta_vals: List of eta values to grid search.
        zeta_vals: List of zeta values to grid search.
        num_rep: Number of repetitions per eta/zeta pair.
        t_st, t_en: Temperature schedule.
        Nstep: Number of iterations.
        device: CUDA device.

    Returns:
        - List of binary solutions (batched).
        - Corresponding scalar function values.
    """
    torch._dynamo.reset()
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()

    grid = list(itertools.product(eta_vals, zeta_vals))
    B = len(grid) * num_rep

    # Shapes of inputs
    num_tensors = len(shapes)
    in_dims = (0,) * num_tensors
    sizes = [torch.prod(torch.tensor(s)).item() for s in shapes]
    total_vars = sum(sizes)

    if Nstep is None:
        Nstep = max(2000, total_vars)

    # Initialize variables
    before = [torch.rand((B, *shape), device=device) for shape in shapes]
    eta_tensor = torch.tensor([e for (e, _) in grid for _ in range(num_rep)], device=device).float()
    zeta_tensor = torch.tensor([z for (_, z) in grid for _ in range(num_rep)], device=device).float()

    after = [b + eta_tensor.view(B, *[1]*len(b.shape[1:])) * (0.5 - b) for b in before]

    delta_t = torch.tensor((t_st - t_en) / (Nstep - 1), device=device)
    now_temp = torch.tensor(t_st, device=device).float()
    eta = eta_tensor
    zeta = zeta_tensor

    # Compute coeffs (outside batch)
    if squared_norm is None or diag_hessians is None:
        squared_norm, diag_hessians = squared_norm_and_diag_hessians(f, *[x[0] for x in after])
    coeff = torch.sqrt(total_vars / squared_norm)

    # Gradient function
    df = vmap(torch.func.grad(f, argnums=tuple(range(num_tensors))), in_dims=in_dims)
    @torch.compile(mode='reduce-overhead')
    def iteration_step(x_b_list, x_a_list):
        x_f_list = [(xa + zeta.view(B, *[1]*len(xa.shape[1:])) * (xa - xb)).detach()
                    for xb, xa in zip(x_b_list, x_a_list)]
        
        grads = df(*x_f_list)
        if not isinstance(grads, (list, tuple)):
            grads = [grads]

        x_next_list = []
        
        for xb, xa, xf, g, dh in zip(x_b_list, x_a_list, x_f_list, grads, diag_hessians):
            mask = (xa == 0) | (xa == 1)
            # diag_hessiansの次元合わせ
            grad = (g + dh * (0.5 - xf)).masked_fill(mask, 0.0)

            update = 2 * xa - xb - eta.view(B, *[1]*len(xa.shape[1:])) * (
                coeff * grad + now_temp * (xa - 0.5))
            x_next = torch.clamp(update, 0.0, 1.0).detach()
            x_next_list.append(x_next)
        return x_a_list, x_next_list

    for _ in tqdm.trange(Nstep, mininterval=5.0, disable=not show_progress):
        before, after = iteration_step(before, after)
        before = [x.detach().clone() for x in before]
        after = [x.detach().clone() for x in after]
        now_temp -= delta_t

    # Final thresholding
    rounded = [torch.where(x > 0.5, 1.0, 0.0) for x in after]
    results = vmap(f, in_dims=in_dims)(*rounded)

    return rounded, results, eta_tensor, zeta_tensor



def auto_amfd(
    f: Callable,
    shapes: List[torch.Size],
    eta_vals: List[float],
    zeta_vals: List[float],
    num_rep: int,
    t_st=0.4,
    t_en=0.001,
    Nstep=None,
    squared_norm=None,
    diag_hessians=None,
    device='cuda:0',
    show_progress=True
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs batched AMFD optimization over eta, zeta grid with repetitions.

    Args:
        f: Callable returning scalar value given list of tensors.
        shapes: List of shapes for input tensors.
        eta_vals: List of eta values to grid search.
        zeta_vals: List of zeta values to grid search.
        num_rep: Number of repetitions per eta/zeta pair.
        t_st, t_en: Temperature schedule.
        Nstep: Number of iterations.
        device: CUDA device.

    Returns:
        - List of binary solutions (each tensor: [B, ...])
        - Corresponding scalar function values: shape [B]
        - eta tensor used: shape [B]
        - zeta tensor used: shape [B]
    """
    torch._dynamo.reset()
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()

    # 全ての(eta, zeta)の組に対しnum_rep回繰り返し
    B = len(eta_vals)*num_rep
    eta_tensor = torch.as_tensor(eta_vals, device=device).float().repeat(num_rep)
    zeta_tensor = torch.as_tensor(zeta_vals, device=device).float().repeat(num_rep)


    # 形状などの準備
    num_tensors = len(shapes)
    in_dims = (0,) * num_tensors
    sizes = [torch.prod(torch.tensor(s)).item() for s in shapes]
    total_vars = sum(sizes)

    if Nstep is None:
        Nstep = max(2000, 2 * total_vars)

    # 変数初期化
    before = [torch.rand((B, *shape), device=device) for shape in shapes]
    after = [b + eta_tensor.view(B, *[1]*len(b.shape[1:])) * (0.5 - b) for b in before]

    delta_t = torch.tensor((t_st - t_en) / (Nstep - 1), device=device)
    now_temp = torch.tensor(t_st, device=device).float()
    eta = eta_tensor
    zeta = zeta_tensor

    # 係数の計算
    if squared_norm is None or diag_hessians is None:
        squared_norm, diag_hessians = squared_norm_and_diag_hessians(f, *[x[0] for x in after])
    coeff = torch.sqrt(total_vars / squared_norm)

    # 勾配関数
    df = vmap(torch.func.grad(f, argnums=tuple(range(num_tensors))), in_dims=in_dims)

    @torch.compile(mode='reduce-overhead')
    def iteration_step(x_b_list, x_a_list):
        x_f_list = [(xa + zeta.view(B, *[1]*len(xa.shape[1:])) * (xa - xb)).detach()
                    for xb, xa in zip(x_b_list, x_a_list)]
        
        grads = df(*x_f_list)
        if not isinstance(grads, (list, tuple)):
            grads = [grads]

        x_next_list = []
        
        for xb, xa, xf, g, dh in zip(x_b_list, x_a_list, x_f_list, grads, diag_hessians):
            mask = (xa == 0) | (xa == 1)
            grad = (g + dh * (0.5 - xf)).masked_fill(mask, 0.0)

            update = 2 * xa - xb - eta.view(B, *[1]*len(xa.shape[1:])) * (
                coeff * grad + now_temp * (xa - 0.5))
            x_next = torch.clamp(update, 0.0, 1.0).detach()
            x_next_list.append(x_next)
        return x_a_list, x_next_list

    for _ in tqdm.trange(Nstep, mininterval=5.0, disable=not show_progress):
        before, after = iteration_step(before, after)
        before = [x.detach().clone() for x in before]
        after = [x.detach().clone() for x in after]
        now_temp -= delta_t

    # 0.5 thresholdingしてバイナリ解に
    rounded = [torch.where(x > 0.5, 1.0, 0.0) for x in after]
    results = vmap(f, in_dims=in_dims)(*rounded)

    return rounded, results, eta_tensor, zeta_tensor



@torch.no_grad()
def amfd(const, h, Q, t_st=0.4, t_en=0.01, Nstep=None, rep_num=100, zeta=0.05, eta=0.03, device='cuda:0'):
    torch.set_float32_matmul_precision('medium')
    # Initial Parameter Setting
    num_spin = Q.shape[0] 
    if Nstep is None:
        Nstep = 2 * num_spin 


    coeff = torch.sqrt(num_spin/(torch.sum(h**2)+torch.sum((Q**2))))

    # GPUへ転送
    Q_norm = (coeff * Q).to(device)
    h_norm = (coeff * h).to(device)
    delta_t = torch.tensor((t_st - t_en)/(Nstep - 1), device=device)
    nowtemp = torch.tensor(t_st, device=device)
    eta = torch.tensor(eta, device = device)
    zeta = torch.tensor(zeta, device=device)

    # Initial Spin State 
    spin_before = torch.rand((rep_num, num_spin), dtype=torch.float32, device=device)
    spin_after = spin_before + eta * (0.5-spin_before)

    @torch.compile(mode='reduce-overhead')
    # @torch.compile(mode='max-autotune')
    def iteration(spin_before, spin_after, now_t):
        # forward
        spin = spin_after + zeta * (spin_after - spin_before)
        FEi = torch.where((spin_after==1) | (spin_after==0), now_t * (spin_after - 0.5), h_norm + spin @ Q_norm + now_t * (spin_after - 0.5))
        return spin_after.detach(), torch.clamp(2*spin_after - spin_before - eta * FEi, 0.0, 1.0).detach()
    
    for i in tqdm.trange(Nstep, mininterval=1.0):
        s_b, s_a = iteration(spin_before=spin_before, spin_after=spin_after, now_t=nowtemp)
        spin_before, spin_after = s_b.clone(), s_a.clone()
        nowtemp -= delta_t

    # 最終スピン状態を決定
    spin_after.copy_(torch.where(spin_after > 0.5, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)))

    # エネルギーを計算
    energy = torch.sum(spin_after @ Q.to(device) * spin_after, dim=1) / 2 + torch.sum(spin_after * h.to(device), dim=1) + const
    spin_best = spin_after[torch.argmin(energy)]

    return spin_best.detach(), energy.min().detach()




@torch.no_grad()
def grid_amfd(const, h, Q, t_st=0.4, t_en=0.01, Nstep=None, rep_num=4,
              zeta=[0, 1, 2, 5, 10, 20, 50],
              eta=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
              device='cuda:0', use_float16=False):

    dtype = torch.float16 if use_float16 else torch.float32
    torch.set_float32_matmul_precision('medium')  # float32時の最適化設定

    # 準備
    num_spin = Q.shape[0]
    if Nstep is None:
        Nstep = 2 * num_spin

    coeff = torch.sqrt(num_spin / (torch.sum(h**2) + torch.sum(Q**2)))
    Q = (coeff * Q).to(device=device, dtype=dtype)
    h = (coeff * h).to(device=device, dtype=dtype)

    # パラメータ展開
    eta_tensor = torch.tensor(eta, device=device, dtype=dtype)
    zeta_tensor = torch.tensor(zeta, device=device, dtype=dtype)
    param_grid = torch.cartesian_prod(eta_tensor, zeta_tensor)  # (n_combo, 2)

    n_combo = param_grid.shape[0]
    total_rep = rep_num * n_combo

    eta_expanded = (param_grid[:, 0].repeat_interleave(rep_num))[:, None]
    zeta_expanded = (param_grid[:, 1].repeat_interleave(rep_num))[:, None]

    delta_t = torch.tensor((t_st - t_en)/(Nstep - 1), device=device, dtype=dtype)
    nowtemp = torch.tensor(t_st, device=device, dtype=dtype)

    spin_before = torch.rand((total_rep, num_spin), dtype=dtype, device=device)
    spin_after = spin_before + eta_expanded * (0.5 - spin_before)

    @torch.compile(mode='default')  
    def iteration(spin_before, spin_after, now_t):
        spin = spin_after + zeta_expanded * (spin_after - spin_before)
        FEi = torch.where(
            (spin_after == 1) | (spin_after == 0),
            now_t * (spin_after - 0.5),
            h + spin @ Q + now_t * (spin_after - 0.5)
        )
        return spin_after.detach(), torch.clamp(
            2 * spin_after - spin_before - eta_expanded * FEi,
            0.0, 1.0
        ).detach()

    for _ in tqdm.trange(Nstep, mininterval=10):
        s_b, s_a = iteration(spin_before=spin_before, spin_after=spin_after, now_t=nowtemp)
        spin_before, spin_after = s_b, s_a
        nowtemp -= delta_t

    spin_after.copy_(torch.where(spin_after > 0.5, 1.0, 0.0))

    # エネルギー計算は float32 に変換して精度を保つ
    spin_after = spin_after.to(torch.float32)
    Q = (Q/coeff).to(device=device, dtype=torch.float32)
    h = (h/coeff).to(device=device, dtype=torch.float32)
    energy = torch.sum(spin_after @ Q * spin_after, dim=1) / 2 \
           + torch.sum(spin_after * h, dim=1) + const

    best_idx = torch.argmin(energy)
    spin_best = spin_after[best_idx]
    return spin_best.detach(), energy[best_idx].detach()




def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def multi_gpu_processing(rank, world_size, return_dict,
                         const, h, Q, t_st=0.4, t_en=0.01, Nstep=None, rep_num=4,
                         zeta=[0, 1, 2, 5, 10, 20, 50],
                         eta=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]):
    setup(rank, world_size)

    device = torch.device(f'cuda:{rank}')
    dtype = torch.float32
    torch.set_float32_matmul_precision('medium')

    num_spin = Q.shape[0]
    if Nstep is None:
        Nstep = 2 * num_spin

    # 初期は CPU 上
    Q = Q.cpu()
    h = h.cpu()

    coeff = torch.sqrt(num_spin / (torch.sum(h**2) + torch.sum(Q**2)))
    Q = (coeff * Q).to(dtype=dtype)
    h = (coeff * h).to(dtype=dtype)

    # Q を行で分割（修正済み）
    chunk_size = (num_spin + world_size - 1) // world_size
    start = rank * chunk_size
    end = min((rank + 1) * chunk_size, num_spin)

    Q_local = Q[start:end].contiguous().to(device)
    h_full = h.to(device)

    # パラメータ展開
    eta_tensor = torch.tensor(eta, dtype=dtype, device=device)
    zeta_tensor = torch.tensor(zeta, dtype=dtype, device=device)
    param_grid = torch.cartesian_prod(eta_tensor, zeta_tensor)

    n_combo = param_grid.shape[0]
    total_rep = rep_num * n_combo

    eta_expanded = (param_grid[:, 0].repeat_interleave(rep_num))[:, None].to(device)
    zeta_expanded = (param_grid[:, 1].repeat_interleave(rep_num))[:, None].to(device)

    delta_t = torch.tensor((t_st - t_en) / (Nstep - 1), device=device, dtype=dtype)
    nowtemp = torch.tensor(t_st, device=device, dtype=dtype)

    spin_before = torch.rand((total_rep, num_spin), dtype=dtype, device=device)
    spin_after = spin_before + eta_expanded * (0.5 - spin_before)

    def iteration(spin_before, spin_after, now_t):
        spin = spin_after + zeta_expanded * (spin_after - spin_before)

        # 分散行列積: spin @ Q_local.T
        local_Q_spin = spin @ Q_local.T  # shape: [B, local_chunk_size]

        # gather用テンソルはプロセスごとにサイズ調整（ここがポイント）
        gathered = [torch.empty((total_rep,
                                 min(chunk_size, num_spin - i * chunk_size)),
                                dtype=dtype, device=device)
                    for i in range(world_size)]
        dist.all_gather(gathered, local_Q_spin)
        Q_spin_full = torch.cat(gathered, dim=1)[:, :num_spin]  # 正しい列数に切り詰め

        FEi = torch.where(
            (spin_after == 1) | (spin_after == 0),
            now_t * (spin_after - 0.5),
            h_full + Q_spin_full + now_t * (spin_after - 0.5)
        )

        return spin_after.detach(), torch.clamp(
            2 * spin_after - spin_before - eta_expanded * FEi,
            0.0, 1.0
        ).detach()


    for _ in tqdm.trange(Nstep, mininterval=10):
        spin_before, spin_after = iteration(spin_before, spin_after, nowtemp)
        nowtemp -= delta_t

    spin_after = torch.where(spin_after > 0.5, 1.0, 0.0)


    # 各ランクの行範囲
    start = rank * chunk_size
    end = min((rank + 1) * chunk_size, num_spin)
    Q_local = Q[:, start:end].to(device)
    h_local = h[start:end].to(device)

    # 分散内積項
    Qx = spin_after @ Q_local  # shape: [rep, chunk]
    partial_energy = torch.sum(spin_after[:, start:end] * Qx, dim=1)  # shape: [rep]
    dist.all_reduce(partial_energy, op=dist.ReduceOp.SUM)

    # h項
    h_partial = torch.sum(spin_after[:, start:end] * h_local, dim=1)
    dist.all_reduce(h_partial, op=dist.ReduceOp.SUM)

    # rank 0 でのみ最終値を保持
    if rank == 0:
        energy = 0.5 * partial_energy + h_partial + const
        best_idx = torch.argmin(energy)
        return_dict["spin"] = spin_after[best_idx].detach().cpu()
        return_dict["energy"] = energy[best_idx].detach().cpu()

    cleanup()


def amfd_multi_gpu(const, h, Q, t_st=0.4, t_en=0.01, Nstep=None, rep_num=4,
                        zeta=[0, 1, 2, 5, 10, 20, 50],
                        eta=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]):
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(multi_gpu_processing,args=(world_size, return_dict, const, h, Q, t_st, t_en, Nstep, rep_num, zeta, eta),
             nprocs=world_size,
             join=True)

    return return_dict.get("spin"), return_dict.get("energy")




