
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import read_file as rf
import optimizer as op
import generator as gn
import torch
import math
import time
from pathlib import Path
import pandas as pd


def get_top_k(solutions, results, eta, zeta, k=5):
    topk_values, topk_indices = torch.topk(results, k=k, largest=False)

    # get top-k solutions
    topk_solutions = [x[topk_indices].cpu() for x in solutions]

    # parameters
    eta_flat = eta.view(-1)
    zeta_flat = zeta.view(-1)
    topk_eta = eta_flat[topk_indices]   # shape: [k]
    topk_zeta = zeta_flat[topk_indices] # shape: [k]

    return topk_solutions, topk_values.cpu(), topk_eta.cpu(), topk_zeta.cpu()

def crossover_parameters(eta_tensor: torch.Tensor, zeta_tensor: torch.Tensor):
    """
    Given eta_tensor and zeta_tensor, return extended tensors by adding
    pairwise averages (0.5 * (a + b)) for all combinations.

    Args:
        eta_tensor: Tensor of shape [N]
        zeta_tensor: Tensor of shape [N]

    Returns:
        new_eta_tensor: shape [N + N_C2]
        new_zeta_tensor: shape [N + N_C2]
    """
    # 元のパラメータ
    eta_tensor = eta_tensor.flatten()
    zeta_tensor = zeta_tensor.flatten()
    N = eta_tensor.shape[0]

    # 全ての2要素の組み合わせを取得（順序なし、重複なし）
    eta_pairs = torch.combinations(eta_tensor, r=2)  # shape: [N_C2, 2]
    zeta_pairs = torch.combinations(zeta_tensor, r=2)

    # 平均をとって交配
    eta_crossed = 0.5 * eta_pairs.sum(dim=1)  # shape: [N_C2]
    zeta_crossed = 0.5 * zeta_pairs.sum(dim=1)

    # 元の値と結合
    new_eta = torch.cat([eta_tensor, eta_crossed], dim=0)
    new_zeta = torch.cat([zeta_tensor, zeta_crossed], dim=0)

    return new_eta, new_zeta

# def check_double_onehot_constraint(sol):
#     is_valid = torch.allclose(sol.sum(dim=-2), torch.ones_like(sol.sum(dim=-2)), atol=1e-5) and \
#         torch.allclose(sol.sum(dim=-1), torch.ones_like(sol.sum(dim=-1)), atol=1e-5)
#     return is_valid

# def check_double_onehot_constraint(sol):
#     row_sum = sol.sum(dim=-2)
#     col_sum = sol.sum(dim=-1)
#     row_valid = torch.all(torch.isclose(row_sum, torch.ones_like(row_sum), atol=1e-5), dim=-1)
#     col_valid = torch.all(torch.isclose(col_sum, torch.ones_like(col_sum), atol=1e-5), dim=-1)
#     return row_valid & col_valid  # shape: (B,)

def check_double_onehot_constraint(sol):
    row_sum = sol.sum(dim=-2)
    col_sum = sol.sum(dim=-1)
    row_valid = torch.all(torch.isclose(row_sum, torch.ones_like(row_sum), atol=1e-5), dim=-1)
    col_valid = torch.all(torch.isclose(col_sum, torch.ones_like(col_sum), atol=1e-5), dim=-1)
    is_valid = row_valid & col_valid  # shape: (B,)

    if torch.any(is_valid):
        first_valid_index = torch.nonzero(is_valid, as_tuple=True)[0][0].item()
    else:
        first_valid_index = None

    return torch.any(is_valid).item(), first_valid_index


def eval_tsp(instance, k=4, genetic=True, step_scale=10, tuning_step_scale=1, device='cuda:0', min_step=0):

    dists = torch.from_numpy(rf.TSP().read_file(instance)).float()
    num_city = dists.shape[0]

    # TSPの定式化を呼び出し(coeffは制約係数: defaultは平均最大距離の1倍)
    coeff = 1
    sample = gn.TSP(dists, coeff1=coeff, coeff2=coeff, device=device)
    shapes = [torch.Size([num_city-1, num_city-1])]
    # pre compile
    op.pre_compile(sample.generator, shapes, device=device)

    start_time = time.time()
    while True:
        squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo)

        sols, vals, etas, zetas = op.auto_grid_amfd(sample.generator, shapes, zeta_vals=[0, 1, 2, 5, 10, 20, 50], eta_vals=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], t_st=0.35, t_en=0.001, num_rep=1, Nstep=max(min_step,tuning_step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device)
        tuning_end_time = time.time()

        topk_sols, topk_vals, topk_etas, topk_zetas = get_top_k(sols, vals, etas, zetas, k=k)
        is_valid, valid_idx = check_double_onehot_constraint(topk_sols[0])
        if is_valid:
            break
        else: 
            coeff *= 1.1
            sample = gn.TSP(dists, coeff1=coeff, coeff2=coeff, device=device)
            squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo)


    del sols, vals, etas, zetas # free memory

    if genetic:
        # 遺伝的アルゴリズムを使用して最適化
        cross_etas, cross_zetas = crossover_parameters(topk_etas, topk_zetas)
        tuned_sols, tuned_vals, tuned_etas, tuned_zetas = op.auto_amfd(sample.generator, shapes, zeta_vals=cross_zetas, eta_vals=cross_etas, t_st=0.35, t_en=0.001, num_rep=10, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device, show_progress=True)

    else:
        tuned_sols, tuned_vals, tuned_etas, tuned_zetas = op.auto_amfd(sample.generator, shapes, zeta_vals=topk_zetas, eta_vals=topk_etas, t_st=0.35, t_en=0.001, num_rep=25, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device, show_progress=True)

    end_time = time.time()

    # 結果の確認
    best_sol, best_val, best_eta, best_zeta = get_top_k(tuned_sols, tuned_vals, tuned_etas, tuned_zetas, k=tuned_vals.numel())
    is_valid, _ = check_double_onehot_constraint(best_sol[0])
    if  not is_valid or best_val[_].item() > topk_vals[valid_idx].item():
        best_sol, best_val, best_eta, best_zeta = topk_sols, topk_vals, topk_etas, topk_zetas
        print("Tuning did not improve the solution, using the best from tuning phase.")


    is_valid, valid_idx = check_double_onehot_constraint(topk_sols[0])
    tuning_result = {'instance': Path(instance).stem, 'process':'tuning', 'step_scale':tuning_step_scale, 'time':round(tuning_end_time-start_time,5), 'value': round(topk_vals[valid_idx].item(),5), 'eta':round(topk_etas[valid_idx].item(),5), 'zeta':round(topk_zetas[valid_idx].item(),5), 'constraint satisfaction': is_valid, 'constraint coeff':coeff}
    is_valid, valid_idx = check_double_onehot_constraint(best_sol[0])
    tuned_result = {'instance': Path(instance).stem, 'process':'tuned', 'step_scale':step_scale, 'time':round(end_time-start_time,5), 'value': round(best_val[valid_idx].item(),5), 'eta':round(best_eta[valid_idx].item(),5), 'zeta':round(best_zeta[valid_idx].item(),5), 'constraint satisfaction': is_valid, 'constraint coeff':coeff}

    return tuning_result , tuned_result




def eval_qap(instance, k=4, genetic=True, step_scale=10, tuning_step_scale=1, device='cuda:0', min_step=0):

    flows, dists = rf.QAP().read_file(instance)
    flows, dists = torch.from_numpy(flows).float(), torch.from_numpy(dists).float()
    num_city = dists.shape[0]

    # QAPの定式化を呼び出し
    coeff = 1
    sample = gn.QAP(flows, dists, coeff1=coeff, coeff2=coeff, device=device)
    shapes = [torch.Size([num_city, num_city])]
    # pre compile
    op.pre_compile(sample.generator, shapes, device=device)

    start_time = time.time()
    while True:
        squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo)

        sols, vals, etas, zetas = op.auto_grid_amfd(sample.generator, shapes, zeta_vals=[0, 1, 2, 5, 10, 20, 50], eta_vals=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], t_st=0.35, t_en=0.001, num_rep=1, Nstep=max(min_step,tuning_step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device)
        tuning_end_time = time.time()

        topk_sols, topk_vals, topk_etas, topk_zetas = get_top_k(sols, vals, etas, zetas, k=k)
        is_valid, valid_idx = check_double_onehot_constraint(topk_sols[0])
        if is_valid:
            break
        else:
            coeff *= 1.1
            sample = gn.QAP(flows, dists, coeff1=coeff, coeff2=coeff, device=device)
            squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo)

    del sols, vals, etas, zetas # free memory

    if genetic:
        # 遺伝的アルゴリズムを使用して最適化
        cross_etas, cross_zetas = crossover_parameters(topk_etas, topk_zetas)
        tuned_sols, tuned_vals, tuned_etas, tuned_zetas = op.auto_amfd(sample.generator, shapes, zeta_vals=cross_zetas, eta_vals=cross_etas, t_st=0.35, t_en=0.001, num_rep=10, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device, show_progress=True)

    else:
        tuned_sols, tuned_vals, tuned_etas, tuned_zetas = op.auto_amfd(sample.generator, shapes, zeta_vals=topk_zetas, eta_vals=topk_etas, t_st=0.35, t_en=0.001, num_rep=25, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device, show_progress=True)

    end_time = time.time()

    # 結果の確認
    best_sol, best_val, best_eta, best_zeta = get_top_k(tuned_sols, tuned_vals, tuned_etas, tuned_zetas, k=tuned_vals.numel())
    is_valid, _ = check_double_onehot_constraint(best_sol[0])
    if not is_valid or best_val[_].item() > topk_vals[valid_idx].item():
        best_sol, best_val, best_eta, best_zeta = topk_sols, topk_vals, topk_etas, topk_zetas
        print("Tuning did not improve the solution, using the best from tuning phase.")


    is_valid, valid_idx = check_double_onehot_constraint(topk_sols[0])
    tuning_result = {'instance': Path(instance).stem, 'process':'tuning', 'step_scale':tuning_step_scale, 'time':round(tuning_end_time-start_time,5), 'value': round(topk_vals[valid_idx].item(),5), 'eta':round(topk_etas[valid_idx].item(),5), 'zeta':round(topk_zetas[valid_idx].item(),5), 'constraint satisfaction': is_valid, 'constraint coeff':coeff}
    is_valid, valid_idx = check_double_onehot_constraint(best_sol[0])
    tuned_result = {'instance': Path(instance).stem, 'process':'tuned', 'step_scale':step_scale, 'time':round(end_time-start_time,5), 'value': round(best_val[valid_idx].item(),5), 'eta':round(best_eta[valid_idx].item(),5), 'zeta':round(best_zeta[valid_idx].item(),5), 'constraint satisfaction': is_valid, 'constraint coeff':coeff}

    return tuning_result , tuned_result


def check_misp_constraint(sols, graph, atol=1e-5):
    """
    Parameters:
        sols: Tensor of shape (B, N)
        graph: Tensor of shape (N, N)
        atol: float

    Returns:
        mask: BoolTensor of shape (B,), 各バッチで条件を満たすかどうか
        first_valid_index: int or None, 条件を満たす最初のバッチのインデックス（なければ None）
    """
    # (B, N) @ (N, N) → (B, N)
    prod = sols @ graph
    elementwise = sols * prod  # (B, N)
    sums = elementwise.sum(dim=1)  # (B,)
    
    mask = torch.isclose(sums, torch.zeros_like(sums), atol=atol)

    # 最初にTrueになるインデックスを取得（なければ None）
    first_valid_index = mask.nonzero(as_tuple=True)[0]
    first_valid_index = first_valid_index[0].item() if first_valid_index.numel() > 0 else None

    return torch.any(mask).item(), first_valid_index


def eval_misp(instance, k=4, genetic=True, step_scale=10, tuning_step_scale=1, device='cuda:0', min_step=0):

    graph = torch.from_numpy(rf.MISP().read_file(instance)).float()
    num_nodes = graph.shape[0]

    # MISPの定式化を呼び出し(coeffは制約係数: defaultは平均最大距離の1倍)
    sample = gn.MISP(graph, coeff1=1, device=device)
    shapes = [torch.Size([num_nodes])]
    # pre compile
    op.pre_compile(sample.generator, shapes, device=device)

    start_time = time.time()

    squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo)

    sols, vals, etas, zetas = op.auto_grid_amfd(sample.generator, shapes, zeta_vals=[0, 1, 2, 5, 10, 20, 50], eta_vals=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], t_st=0.35, t_en=0.001, num_rep=1, Nstep=max(min_step,tuning_step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device)
    tuning_end_time = time.time()

    topk_sols, topk_vals, topk_etas, topk_zetas = get_top_k(sols, vals, etas, zetas, k=k)

    del sols, vals, etas, zetas # free memory

    if genetic:
        # 遺伝的アルゴリズムを使用して最適化
        cross_etas, cross_zetas = crossover_parameters(topk_etas, topk_zetas)
        tuned_sols, tuned_vals, tuned_etas, tuned_zetas = op.auto_amfd(sample.generator, shapes, zeta_vals=cross_zetas, eta_vals=cross_etas, t_st=0.35, t_en=0.001, num_rep=10, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device, show_progress=True)

    else:
        tuned_sols, tuned_vals, tuned_etas, tuned_zetas = op.auto_amfd(sample.generator, shapes, zeta_vals=topk_zetas, eta_vals=topk_etas, t_st=0.35, t_en=0.001, num_rep=25, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device, show_progress=True)

    end_time = time.time()

    # 結果の確認
    best_sol, best_val, best_eta, best_zeta = get_top_k(tuned_sols, tuned_vals, tuned_etas, tuned_zetas, k=tuned_vals.numel())
    tuning_valid, tuning_idx = check_misp_constraint(topk_sols, graph)
    tuned_valid, tuned_idx = check_misp_constraint(best_sol, graph)
    if not tuned_valid or best_val[tuned_idx].item() > topk_vals[tuning_idx].item():
        best_sol, best_val, best_eta, best_zeta = topk_sols, topk_vals, topk_etas, topk_zetas
        print("Tuning did not improve the solution, using the best from tuning phase.")


    tuning_result = {'instance': Path(instance).stem, 'process':'tuning', 'step_scale':tuning_step_scale, 'time':round(tuning_end_time-start_time,5), 'value': round(topk_vals[tuning_idx].item(),5), 'eta':round(topk_etas[tuning_idx].item(),5), 'zeta':round(topk_zetas[tuning_idx].item(),5), 'constraint satisfaction': tuning_valid, 'constraint coeff':1}
    tuned_result = {'instance': Path(instance).stem, 'process':'tuned', 'step_scale':step_scale, 'time':round(end_time-start_time,5), 'value': round(best_val[tuned_idx].item(),5), 'eta':round(best_eta[tuned_idx].item(),5), 'zeta':round(best_zeta[tuned_idx].item(),5), 'constraint satisfaction': tuned_valid, 'constraint coeff':1}

    return tuning_result , tuned_result

def eval_mcp(instance, k=4, genetic=True, step_scale=10, tuning_step_scale=1, device='cuda:0', min_step=0):

    graph = torch.from_numpy(rf.MCP().read_file(instance)).float()
    num_nodes = graph.shape[0]

    # MISPの定式化を呼び出し(coeffは制約係数: defaultは平均最大距離の1倍)
    sample = gn.MCP(graph, device=device)
    shapes = [torch.Size([num_nodes])]
    # pre compile
    op.pre_compile(sample.generator, shapes, device=device)

    start_time = time.time()

    squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo)

    sols, vals, etas, zetas = op.auto_grid_amfd(sample.generator, shapes, zeta_vals=[0, 1, 2, 5, 10, 20, 50], eta_vals=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], t_st=0.35, t_en=0.001, num_rep=1, Nstep=max(min_step,tuning_step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device)
    tuning_end_time = time.time()

    topk_sols, topk_vals, topk_etas, topk_zetas = get_top_k(sols, vals, etas, zetas, k=k)

    del sols, vals, etas, zetas # free memory

    if genetic:
        # 遺伝的アルゴリズムを使用して最適化
        cross_etas, cross_zetas = crossover_parameters(topk_etas, topk_zetas)
        tuned_sols, tuned_vals, tuned_etas, tuned_zetas = op.auto_amfd(sample.generator, shapes, zeta_vals=cross_zetas, eta_vals=cross_etas, t_st=0.35, t_en=0.001, num_rep=10, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device, show_progress=True)

    else:
        tuned_sols, tuned_vals, tuned_etas, tuned_zetas = op.auto_amfd(sample.generator, shapes, zeta_vals=topk_zetas, eta_vals=topk_etas, t_st=0.35, t_en=0.001, num_rep=25, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device, show_progress=True)

    end_time = time.time()

    # 結果の確認
    best_sol, best_val, best_eta, best_zeta = get_top_k(tuned_sols, tuned_vals, tuned_etas, tuned_zetas, k=1)
    if best_val[0].item() > topk_vals[0].item():
        best_sol, best_val, best_eta, best_zeta = [topk_sols[0]], [topk_vals[0]], [topk_etas[0]], [topk_zetas[0]]
        print("Tuning did not improve the solution, using the best from tuning phase.")


    tuning_result = {'instance': Path(instance).stem, 'process':'tuning', 'step_scale':tuning_step_scale, 'time':round(tuning_end_time-start_time,5), 'value': round(topk_vals[0].item(),5), 'eta':round(topk_etas[0].item(),5), 'zeta':round(topk_zetas[0].item(),5), 'constraint satisfaction': True, 'constraint coeff':None}

    tuned_result = {'instance': Path(instance).stem, 'process':'tuned', 'step_scale':step_scale, 'time':round(end_time-start_time,5), 'value': round(best_val[0].item(),5), 'eta':round(best_eta[0].item(),5), 'zeta':round(best_zeta[0].item(),5), 'constraint satisfaction': True, 'constraint coeff':None}

    return tuning_result , tuned_result

def check_gcp_constraint(sols, graph):
    zero = torch.tensor(0.0, device=sols.device)
    graph_expanded = graph.unsqueeze(0)  # shape: [1, N, N]

    # 条件1: xᵀ @ graph @ x ≈ 0
    cond1_vals = torch.isclose(
        (sols * (graph_expanded @ sols)).sum(dim=(1, 2)),
        zero,
        atol=1e-5
    )

    # 条件2: 各行の和が1に近い (ペナルティが小さい ≈ 0)
    cond2_vals = torch.isclose(
        ((1 - sols.sum(dim=2))**2).sum(dim=1),
        zero,
        atol=1e-5
    )

    # 両方の条件を満たすバッチインデックスを取得
    valid_mask = cond1_vals & cond2_vals

    is_valid = valid_mask.any().item()

    if is_valid:
        # 各バッチに対して、有効な行の数を数える（列和が1e-3より大きい行数）
        row_counts = (sols.sum(dim=-2) > 1e-3).sum(dim=1)  # shape: [B]
        
        # 無効なバッチは無限大でマスク（ランキングから除外）
        row_counts[~valid_mask] = sols.shape[-2]  # 無効なバッチは最大行数でマスク

        # 最小行数を持つバッチのインデックスを取得
        first_valid_index = torch.argmin(row_counts).item()
    else:
        first_valid_index = -1  # または -1 など
    return is_valid, first_valid_index

def eval_gcp(instance, k=4, genetic=True, step_scale=10, tuning_step_scale=1, device='cuda:0', min_step=0):

    graph = torch.from_numpy(rf.GCP().read_file(instance)).float()
    num_nodes = graph.shape[0]

    # GCPの定式化を呼び出し(coeffは制約係数: defaultは平均最大距離の1倍)
    sample = gn.GCP(graph, coeff1=1, coeff2=1, coeff3=1, num_color=None, device=device)
    shapes = [torch.Size([num_nodes, sample.num_color]), torch.Size([sample.num_color])]
    # pre compile
    op.pre_compile(sample.generator, shapes, device=device)

    start_time = time.time()

    squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo)
    tuning_is_valid = False
    while True:
        sols, vals, etas, zetas = op.auto_grid_amfd(sample.generator, shapes, zeta_vals=[0, 1, 2, 5, 10, 20, 50], eta_vals=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], t_st=0.35, t_en=0.001, num_rep=1, Nstep=max(min_step,tuning_step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device)
        topk_sols, topk_vals, topk_etas, topk_zetas = get_top_k(sols, vals, etas, zetas, k=k)
        is_valid, valid_index = check_gcp_constraint(sols=topk_sols[0], graph=graph)
        if is_valid:
            del diag_hessians, squared_norm
            num_color = int(round((topk_sols[0][valid_index].sum(dim=0) > 1e-3).sum().item(), 5))
            sample = gn.GCP(graph, coeff1=1, coeff2=1, coeff3=1, num_color=int(num_color-1), device=device)
            shapes = [torch.Size([num_nodes, sample.num_color]), torch.Size([sample.num_color])]
            squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo)
            tuning_is_valid = True
        else: break

    tuning_end_time = time.time()


    if genetic:
        # 遺伝的アルゴリズムを使用して最適化
        cross_etas, cross_zetas = crossover_parameters(topk_etas, topk_zetas)
        tuned_sols, tuned_vals, tuned_etas, tuned_zetas = op.auto_amfd(sample.generator, shapes, zeta_vals=cross_zetas, eta_vals=cross_etas, t_st=0.35, t_en=0.001, num_rep=10, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), diag_hessians=diag_hessians, squared_norm=squared_norm, device=device, show_progress=True)

    else:
        tuned_sols, tuned_vals, tuned_etas, tuned_zetas = op.auto_amfd(sample.generator, shapes, zeta_vals=topk_zetas, eta_vals=topk_etas, t_st=0.35, t_en=0.001, num_rep=25, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), diag_hessians=diag_hessians, squared_norm=squared_norm, device=device, show_progress=True)

    end_time = time.time()

    # 結果の確認
    tuning_color = round((topk_sols[0][valid_index].sum(dim=0) > 1e-3).sum().item(), 5)
    tuning_result = {'instance': Path(instance).stem, 'process':'tuning', 'step_scale':tuning_step_scale, 'time':round(tuning_end_time-start_time,5), 'value': tuning_color, 'eta':round(topk_etas[valid_index].item(),5), 'zeta':round(topk_zetas[valid_index].item(),5), 'constraint satisfaction': tuning_is_valid, 'constraint coeff':1}

    best_sol, best_val, best_eta, best_zeta = get_top_k(tuned_sols, tuned_vals, tuned_etas, tuned_zetas, k=tuned_sols[0].shape[0])
    post_is_valid, valid_index = check_gcp_constraint(sols=best_sol[0], graph=graph)
    if post_is_valid:
        tuned_color = round((best_sol[0][valid_index].sum(dim=0) > 1e-3).sum().item(),5)
    else: tuned_color = tuning_color

    if tuned_color >= tuning_color:
        best_sol, best_val, best_eta, best_zeta = [topk_sols[0], topk_sols[1]], [topk_vals[0]], [topk_etas[0]], [topk_zetas[0]]
        post_is_valid = tuning_is_valid
        print("Tuning did not improve the solution, using the best from tuning phase.")

    tuned_result = {'instance': Path(instance).stem, 'process':'tuned', 'step_scale':step_scale, 'time':round(end_time-start_time,5), 'value': tuned_color, 'eta':round(best_eta[valid_index].item(),5), 'zeta':round(best_zeta[valid_index].item(),5), 'constraint satisfaction': post_is_valid, 'constraint coeff':1}

    return tuning_result , tuned_result


def eval_all_tsp(k=4, genetic=True, tuning_step_scale=2, step_scales=[1, 2, 10, 20], device='cuda:0', seed=0, dir=os.path.join(parent_dir, 'datasets/tsp')):
    """
    Evaluate all TSP instances in the specified directory.
    
    Args:
        tsp_dir (str): Directory containing TSP files.
        k (int): Number of top solutions to consider.
        genetic (bool): Whether to use genetic algorithm for optimization.
        step_scale (int): Step scale for optimization.
        tuning_step_scale (int): Step scale for tuning phase.
        device (str): Device to run the evaluation on.
    """

    # ディレクトリ内の.tspファイルをすべて取得
    tsp_files = [f for f in os.listdir(dir) if f.endswith('.tsp')]

    results = []
    for tsp_file in sorted(tsp_files):
        instance = os.path.join(dir, tsp_file)
        for step_scale in step_scales:
            torch.manual_seed(seed)
            # 評価実行
            tuning_result, tuned_result = eval_tsp(
                instance=instance,
                k=k,
                genetic=genetic,
                step_scale=step_scale,
                tuning_step_scale=tuning_step_scale,
                device=device
            )
            results += [tuning_result, tuned_result]
            print("Tuning Result:", tuning_result)
            print("Tuned Result:", tuned_result)
            print("-" * 60)
            df = pd.DataFrame(results)
            target_dir = os.path.join(os.path.dirname(__file__), f'results_k{k}')
            os.makedirs(target_dir, exist_ok=True)
            df.to_csv(os.path.join(target_dir, 'tsp_results.csv'), index=False)

def eval_all_qap(k=4, genetic=True, tuning_step_scale=2, step_scales=[1, 2, 10, 20], device='cuda:0', seed=0, dir=os.path.join(parent_dir, 'datasets/qap')):
    """
    Evaluate all QAP instances in the specified directory.
    
    Args:
        qap_dir (str): Directory containing QAP files.
        k (int): Number of top solutions to consider.
        genetic (bool): Whether to use genetic algorithm for optimization.
        step_scale (int): Step scale for optimization.
        tuning_step_scale (int): Step scale for tuning phase.
        device (str): Device to run the evaluation on.
    """

    # ディレクトリ内の.qapファイルをすべて取得
    files = [f for f in os.listdir(dir) if f.endswith('.qap')]

    results = []
    for file in sorted(files):

        instance = os.path.join(dir, file)
        for step_scale in step_scales:
            torch.manual_seed(seed)
            # 評価実行
            tuning_result, tuned_result = eval_qap(
                instance=instance,
                k=k,
                genetic=genetic,
                step_scale=step_scale,
                tuning_step_scale=tuning_step_scale,
                device=device
            )
            results += [tuning_result, tuned_result]
            print("Tuning Result:", tuning_result)
            print("Tuned Result:", tuned_result)
            print("-" * 60)
            df = pd.DataFrame(results)
            target_dir = os.path.join(os.path.dirname(__file__), f'results_k{k}')
            os.makedirs(target_dir, exist_ok=True)
            df.to_csv(os.path.join(target_dir, 'qap_results.csv'), index=False)

def eval_all_misp(k=4, genetic=True, tuning_step_scale=2, step_scales=[1, 2, 10, 20], device='cuda:0', seed=0, dir=os.path.join(parent_dir, 'datasets/misp')):
    """
    Evaluate all MISP instances in the specified directory.
    
    Args:
        dir (str): Directory containing MISP files.
        k (int): Number of top solutions to consider.
        genetic (bool): Whether to use genetic algorithm for optimization.
        step_scale (int): Step scale for optimization.
        tuning_step_scale (int): Step scale for tuning phase.
        device (str): Device to run the evaluation on.
    """

    # ディレクトリ内の.clqファイルをすべて取得
    files = [f for f in os.listdir(dir) if f.endswith('.clq')]

    results = []
    for file in sorted(files):
        instance = os.path.join(dir, file)
        for step_scale in step_scales:
            print(f"Evaluating {file} ...")
            torch.manual_seed(seed)
            # 評価実行
            tuning_result, tuned_result = eval_misp(
                instance=instance,
                k=k,
                genetic=genetic,
                step_scale=step_scale,
                tuning_step_scale=tuning_step_scale,
                device=device
            )
            results += [tuning_result, tuned_result]
            print("Tuning Result:", tuning_result)
            print("Tuned Result:", tuned_result)
            print("-" * 60)
            df = pd.DataFrame(results)
            target_dir = os.path.join(os.path.dirname(__file__), f'results_k{k}')
            os.makedirs(target_dir, exist_ok=True)
            df.to_csv(os.path.join(target_dir, 'misp_results.csv'), index=False)

def eval_all_mcp(k=4, genetic=True, tuning_step_scale=2, step_scales=[1, 2, 10, 20], device='cuda:0', seed=0, dir=os.path.join(parent_dir, 'datasets/mcp')):
    """
    Evaluate all MISP instances in the specified directory.
    
    Args:
        dir (str): Directory containing MISP files.
        k (int): Number of top solutions to consider.
        genetic (bool): Whether to use genetic algorithm for optimization.
        step_scale (int): Step scale for optimization.
        tuning_step_scale (int): Step scale for tuning phase.
        device (str): Device to run the evaluation on.
    """

    # ディレクトリ内の.mcpファイルをすべて取得
    files = [f for f in os.listdir(dir) if f.endswith('.mcp')]

    results = []
    for file in sorted(files):
        instance = os.path.join(dir, file)
        for step_scale in step_scales:
            print(f"Evaluating {file} ...")
            torch.manual_seed(seed)
            # 評価実行
            tuning_result, tuned_result = eval_mcp(
                instance=instance,
                k=k,
                genetic=genetic,
                step_scale=step_scale,
                tuning_step_scale=tuning_step_scale,
                device=device
            )
            results += [tuning_result, tuned_result]
            print("Tuning Result:", tuning_result)
            print("Tuned Result:", tuned_result)
            print("-" * 60)
            df = pd.DataFrame(results)
            target_dir = os.path.join(os.path.dirname(__file__), f'results_k{k}')
            os.makedirs(target_dir, exist_ok=True)
            df.to_csv(os.path.join(target_dir, f'mcp_results.csv'), index=False)




def eval_all_gcp(k=4, genetic=True, tuning_step_scale=2, step_scales=[1, 2, 10, 20], device='cuda:0', seed=0, dir=os.path.join(parent_dir, 'datasets/gcp')):
    """
    Evaluate all MCP instances in the specified directory.
    
    Args:
        dir (str): Directory containing MCP files.
        k (int): Number of top solutions to consider.
        genetic (bool): Whether to use genetic algorithm for optimization.
        step_scale (int): Step scale for optimization.
        tuning_step_scale (int): Step scale for tuning phase.
        device (str): Device to run the evaluation on.
    """

    # ディレクトリ内の.mcpファイルをすべて取得
    files = [f for f in os.listdir(dir) if f.endswith('.col')]

    results = []
    for file in sorted(files):
        instance = os.path.join(dir, file)
        for step_scale in step_scales:
            print(f"Evaluating {file} ...")
            torch.manual_seed(seed)
            # 評価実行
            tuning_result, tuned_result = eval_gcp(
                instance=instance,
                k=k,
                genetic=genetic,
                step_scale=step_scale,
                tuning_step_scale=tuning_step_scale,
                device=device
            )
            results += [tuning_result, tuned_result]
            print("Tuning Result:", tuning_result)
            print("Tuned Result:", tuned_result)
            print("-" * 60)
            df = pd.DataFrame(results)
            target_dir = os.path.join(os.path.dirname(__file__), f'results_k{k}')
            os.makedirs(target_dir, exist_ok=True)
            df.to_csv(os.path.join(target_dir, 'gcp_results.csv'), index=False)




if __name__ == "__main__":
    # 再現性のためのシード固定
    seed = 0
    torch.manual_seed(seed)
    device = 'cuda:0'

    # eval_all_tsp(
    #     k=5,
    #     genetic=True,
    #     tuning_step_scale=2,
    #     step_scales=[2, 5, 10, 20, 50],
    #     device=device,
    #     seed=seed
    # )

    # eval_all_qap(
    #     k=5,
    #     genetic=True,
    #     tuning_step_scale=2,
    #     step_scales=[2, 5, 10, 20, 50],
    #     device=device,
    #     seed=seed
    # )

    # eval_all_misp(
    #     k=5,
    #     genetic=True,
    #     tuning_step_scale=2,
    #     step_scales=[2, 5, 10, 20, 50],
    #     device=device,
    #     seed=seed
    # )

    # eval_all_mcp(
    #     k=5,
    #     genetic=True,
    #     tuning_step_scale=2,
    #     step_scales=[2, 5, 10, 20, 50],
    #     device=device,
    #     seed=seed
    # )   

    eval_all_gcp(
        k=5,
        genetic=True,
        tuning_step_scale=2,
        step_scales=[2, 5, 10, 20, 50],
        device=device,
        seed=seed
    )


