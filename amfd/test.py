
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import read_file as rf
import amfd_optimizer as op
import generator as gn
import torch
import math
import time
from pathlib import Path
import pandas as pd
import copy


def eval_mcp(instance, step_scale=10, num_rep=100, device='cuda:0', min_step=0):

    graph = torch.from_numpy(rf.MCP().read_file(instance)).float()
    num_nodes = graph.shape[0]

    sample = gn.MCP(graph, device=device)
    shapes = [torch.Size([num_nodes])]

    squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo)

    sols, vals, etas, zetas = op.auto_grid_amfd(sample.generator, shapes, zeta_vals=[0, 1, 2, 5, 10, 20, 50], eta_vals=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], t_st=0.35, t_en=0.001, num_rep=num_rep, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device)

    return sols, vals, etas, zetas


def eval_misp(instance, step_scale=10, num_rep=100, device='cuda:0', min_step=0):

    graph = torch.from_numpy(rf.MISP().read_file(instance)).float()
    num_nodes = graph.shape[0]

    sample = gn.MISP(graph, coeff1=1, device=device)
    shapes = [torch.Size([num_nodes])]

    squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo)

    sols, vals, etas, zetas = op.auto_grid_amfd(sample.generator, shapes, zeta_vals=[0, 1, 2, 5, 10, 20, 50], eta_vals=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], t_st=0.35, t_en=0.001, num_rep=num_rep, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device)

    return sols, vals, etas, zetas

def eval_tsp(instance, step_scale=10, num_rep=100, device='cuda:0', min_step=0):

    dists = torch.from_numpy(rf.TSP().read_file(instance)).float()
    num_city = dists.shape[0]

    coeff = 1
    sample = gn.TSP(dists, coeff1=coeff, coeff2=coeff, device=device)
    shapes = [torch.Size([num_city-1, num_city-1])]

    squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo)

    sols, vals, etas, zetas = op.auto_grid_amfd(sample.generator, shapes, zeta_vals=[0, 1, 2, 5, 10, 20, 50], eta_vals=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], t_st=0.35, t_en=0.001, num_rep=num_rep, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device)

    return sols, vals, etas, zetas


def eval_qap(instance, step_scale=10, num_rep=100, device='cuda:0', min_step=0):

    flows, dists = rf.QAP().read_file(instance)
    flows, dists = torch.from_numpy(flows).float(), torch.from_numpy(dists).float()
    num_city = dists.shape[0]

    coeff = 1
    sample = gn.QAP(flows, dists, coeff1=coeff, coeff2=coeff, device=device)
    shapes = [torch.Size([num_city, num_city])]

    squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo)

    sols, vals, etas, zetas = op.auto_grid_amfd(sample.generator, shapes, zeta_vals=[0, 1, 2, 5, 10, 20, 50], eta_vals=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], t_st=0.35, t_en=0.001, num_rep=num_rep, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device)

    return sols, vals, etas, zetas


def eval_gcp(instance, step_scale=10, num_rep=100, device='cuda:0', min_step=0):

    graph = torch.from_numpy(rf.GCP().read_file(instance)).float()
    num_nodes = graph.shape[0]

    sample = gn.GCP(graph, coeff1=1, coeff2=1, coeff3=1, num_color=None, device=device)
    shapes = [torch.Size([num_nodes, sample.num_color]), torch.Size([sample.num_color])]

    squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=sample.build_qubo) 
    sols, vals, etas, zetas = op.auto_grid_amfd(sample.generator, shapes, zeta_vals=[0, 1, 2, 5, 10, 20, 50], eta_vals=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], t_st=0.35, t_en=0.001, num_rep=num_rep, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device)

    return sols, vals, etas, zetas

def eval_bqp(instance, step_scale=10, num_rep=100, device='cuda:0', min_step=0):

    graph = torch.from_numpy(rf.BQP().read_file(instance)).float()
    num_nodes = graph.shape[0]

    sample = gn.BQP(graph, device=device)
    shapes = [torch.Size([num_nodes])]

    squared_norm, diag_hessians = op.squared_norm_and_diag_hessians(sample.generator, shapes, device=device, generate_function=None)

    sols, vals, etas, zetas = op.auto_grid_amfd(sample.generator, shapes, zeta_vals=[0, 1, 2, 5, 10, 20, 50], eta_vals=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], t_st=0.35, t_en=0.001, num_rep=num_rep, Nstep=max(min_step, step_scale*sum(math.prod(shape) for shape in shapes)), squared_norm=squared_norm, diag_hessians=diag_hessians, device=device)

    return sols, vals, etas, zetas


def make_table(vals, etas, zetas):
    etas = etas.cpu().numpy()
    zetas = zetas.cpu().numpy()
    vals = vals.cpu().numpy()

    # DataFrame化
    df = pd.DataFrame({
        'eta': etas,
        'zeta': zetas,
        'val': vals
    })

    # eta, zetaごとに平均を計算し、行方向=eta、列方向=zetaでpivot
    pivot_df = df.groupby(['eta', 'zeta'])['val'].min().unstack()

    return pivot_df


if __name__ == '__main__':
    dataset_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/datasets'

    instance = os.path.join(dataset_dir, 'mcp/G22.mcp')
    sols, vals, etas, zetas = eval_mcp(instance, step_scale=10, num_rep=100, device='cuda:0', min_step=0)
    df = make_table(vals, etas, zetas)
    df.to_csv('mcp_sensitivity.csv')

    instance = os.path.join(dataset_dir, 'misp/C2000.5.clq')
    sols, vals, etas, zetas = eval_misp(instance, step_scale=10, num_rep=100, device='cuda:0', min_step=0)
    df = make_table(vals, etas, zetas)
    df.to_csv('misp_sensitivity.csv')

    instance = os.path.join(dataset_dir, 'tsp/eil51.tsp')
    sols, vals, etas, zetas = eval_tsp(instance, step_scale=10, num_rep=100, device='cuda:0', min_step=0)
    df = make_table(vals, etas, zetas)
    df.to_csv('tsp_sensitivity.csv')

    instance = os.path.join(dataset_dir, 'qap/tai50a.qap')
    sols, vals, etas, zetas = eval_qap(instance, step_scale=10, num_rep=100, device='cuda:0', min_step=0)
    df = make_table(vals, etas, zetas)
    df.to_csv('qap_sensitivity.csv')

    instance = os.path.join(dataset_dir, 'gcp/jean.col')
    sols, vals, etas, zetas = eval_gcp(instance, step_scale=10, num_rep=100, device='cuda:0', min_step=0)
    df = make_table(vals, etas, zetas)
    df.to_csv('gcp_sensitivity.csv')

    instance = os.path.join(dataset_dir, 'orlib/bqp2500_01.txt')
    sols, vals, etas, zetas = eval_bqp(instance, step_scale=10, num_rep=10, device='cuda:0', min_step=0)
    df = make_table(vals, etas, zetas)
    df.to_csv('bqp_sensitivity.csv')
