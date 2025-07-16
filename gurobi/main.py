
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
import mediator as md
import read_file as rf
import generator as gn
import torch
from pathlib import Path
import pandas as pd
import gurobi_optimizer as go



def get_amfd_result(csv_file, instance_name):

    # CSVファイルの読み込み
    df = pd.read_csv(csv_file)  # ← ファイル名を適宜変更してください

    # 条件を満たす行を抽出
    filtered = df[(df['instance'] == instance_name)]

    # time列とbest known solution列を取り出す
    result = filtered[['time', 'value', 'best known solution']]

    return list(result.itertuples(index=False, name=None))

def check_double_onehot_constraint(x):
    is_valid = torch.allclose(x.sum(dim=0), torch.ones_like(x.sum(dim=0)), atol=1e-5) and \
        torch.allclose(x.sum(dim=1), torch.ones_like(x.sum(dim=1)), atol=1e-5)
    return is_valid

def check_misp_constraint(x, graph):
    return torch.allclose((x.float() * (x.float() @ graph.float())).sum(), torch.zeros(1, device=graph.device), atol=1e-5) 


def eval_tsp(instance, time_limit=60, target_obj=None, time_points=None, thread_num=8):

    dists = (rf.TSP().read_file(instance))

    # MILP solver
    results = []
    best_sol_1, best_obj_1, runtime_1, obj_log_1 = go.TSP(dists).gurobi_optimize_MILP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=[])
    for t, obj, sol in obj_log_1:
        result = {'instance': Path(instance).stem, 'process':'MILP', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MILP', 'time':round(runtime_1, 5), 'value': round(best_obj_1, 2), 'constraint satisfaction': True, 'best known solution':target_obj})

    # MIQP solver
    best_sol_2, best_obj_2, runtime_2, obj_log_2 = go.TSP(dists).gurobi_optimize_MIQP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=[])  
    for t, obj, sol in obj_log_2:
        result = {'instance': Path(instance).stem, 'process':'MIQP', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MIQP', 'time':round(runtime_2,5), 'value': round(best_obj_2,2), 'constraint satisfaction': True, 'best known solution':target_obj})

    # QUBO solver
    qubo, meta = md.get_qubo(gn.TSP(torch.from_numpy(dists).float()).generator, {'x': torch.Size([dists.shape[0]-1, dists.shape[0]-1])}, device='cuda:0')
    best_sol_3, best_obj_3, runtime_3, obj_log_3 = go.QUBO(qubo['Q'], qubo['h'], qubo['const']).gurobi_optimize_QUBO(time_limit=time_limit, time_points=time_points, thread_num=thread_num, target_obj=target_obj, obj_log=[])
    for t, obj, sol in obj_log_3:
        sol = md.restore_variables(torch.tensor(sol),meta['index_map'])
        result = {'instance': Path(instance).stem, 'process':'QUBO', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': check_double_onehot_constraint(sol['x']), 'best known solution':target_obj}
        results.append(result)
    best_sol_3 = md.restore_variables(torch.tensor(best_sol_3),meta['index_map'])
    results.append({'instance': Path(instance).stem, 'process':'QUBO', 'time':round(runtime_3,5), 'value': round(best_obj_3,2), 'constraint satisfaction': check_double_onehot_constraint(best_sol_3['x']), 'best known solution':target_obj})
    return results

def eval_qap(instance, time_limit=60, target_obj=None, time_points=None, thread_num=8):

    flows, dists = (rf.QAP().read_file(instance))

    # MILP solver
    results = []
    best_sol, best_obj, runtime, obj_log = go.QAP(flows, dists).gurobi_optimize_MILP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=[])
    for t, obj, sol in obj_log:
        result = {'instance': Path(instance).stem, 'process':'MILP', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MILP', 'time':round(runtime,5), 'value': round(best_obj,2), 'constraint satisfaction': True, 'best known solution':target_obj})

    # MIQP solver
    best_sol, best_obj, runtime, obj_log = go.QAP(flows, dists).gurobi_optimize_MIQP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=[])  
    for t, obj, sol in obj_log:
        result = {'instance': Path(instance).stem, 'process':'MIQP', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MIQP', 'time':round(runtime,5), 'value': round(best_obj,2), 'constraint satisfaction': True, 'best known solution':target_obj})

    # QUBO solver
    qubo, meta = md.get_qubo(gn.QAP(torch.from_numpy(flows).float(), torch.from_numpy(dists).float(), coeff1=1, coeff2=1).generator, {'x': torch.Size([dists.shape[0], dists.shape[0]])}, device='cuda:0')
    best_sol, best_obj, runtime, obj_log = go.QUBO(qubo['Q'], qubo['h'], qubo['const']).gurobi_optimize_QUBO(time_limit=time_limit, time_points=time_points, thread_num=thread_num, target_obj=target_obj, obj_log=[])
    for t, obj, sol in obj_log:
        sol = md.restore_variables(torch.tensor(sol),meta['index_map'])
        result = {'instance': Path(instance).stem, 'process':'QUBO', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': check_double_onehot_constraint(sol['x']), 'best known solution':target_obj}
        results.append(result)
    best_sol = md.restore_variables(torch.tensor(best_sol),meta['index_map'])
    results.append({'instance': Path(instance).stem, 'process':'QUBO', 'time':round(runtime,5), 'value': round(best_obj,2), 'constraint satisfaction': check_double_onehot_constraint(best_sol['x']), 'best known solution':target_obj})
    return results





def eval_misp(instance, time_limit=60, target_obj=None, time_points=None, thread_num=8):

    E = (rf.MISP().read_file(instance))

    # MILP solver
    results = []
    best_sol, best_obj, runtime, obj_log = go.MISP(E).gurobi_optimize_MILP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=[])
    for t, obj, sol in obj_log:
        result = {'instance': Path(instance).stem, 'process':'MILP', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MILP', 'time':round(runtime,5), 'value': round(best_obj,2), 'constraint satisfaction': True, 'best known solution':target_obj})

    # MIQP solver
    best_sol, best_obj, runtime, obj_log = go.MISP(E).gurobi_optimize_MIQP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=[])  
    for t, obj, sol in obj_log:
        result = {'instance': Path(instance).stem, 'process':'MIQP', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MIQP', 'time':round(runtime,5), 'value': round(best_obj,2), 'constraint satisfaction': True, 'best known solution':target_obj})

    # QUBO solver
    qubo, meta = md.get_qubo(gn.MISP(torch.from_numpy(E).float(), coeff1=1).generator, {'x': torch.Size([E.shape[0]])}, device='cuda:0')
    best_sol, best_obj, runtime, obj_log = go.QUBO(qubo['Q'], qubo['h'], qubo['const']).gurobi_optimize_QUBO(time_limit=time_limit, time_points=time_points, thread_num=thread_num, target_obj=target_obj, obj_log=[])
    for t, obj, sol in obj_log:
        sol = md.restore_variables(torch.tensor(sol),meta['index_map'])
        result = {'instance': Path(instance).stem, 'process':'QUBO', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': check_misp_constraint(sol['x'], torch.from_numpy(E).float()), 'best known solution':target_obj}
        results.append(result)
    best_sol = md.restore_variables(torch.tensor(best_sol),meta['index_map'])
    results.append({'instance': Path(instance).stem, 'process':'QUBO', 'time':round(runtime,5), 'value': round(best_obj,2), 'constraint satisfaction': check_misp_constraint(best_sol['x'], torch.from_numpy(E).float()), 'best known solution':target_obj})
    return results

def eval_mcp(instance, time_limit=60, target_obj=None, time_points=None, thread_num=8):

    E = (rf.MCP().read_file(instance))

    # MILP solver
    results = []
    best_sol, best_obj, runtime, obj_log = go.MCP(E).gurobi_optimize_MILP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=[])
    for t, obj, sol in obj_log:
        result = {'instance': Path(instance).stem, 'process':'MILP', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MILP', 'time':round(runtime,5), 'value': round(best_obj,2), 'constraint satisfaction': True, 'best known solution':target_obj})

    # MIQP solver
    best_sol, best_obj, runtime, obj_log = go.MCP(E).gurobi_optimize_MIQP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=[])  
    for t, obj, sol in obj_log:
        result = {'instance': Path(instance).stem, 'process':'MIQP', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MIQP', 'time':round(runtime,5), 'value': round(best_obj,2), 'constraint satisfaction': True, 'best known solution':target_obj})

    # QUBO solver
    qubo, meta = md.get_qubo(gn.MCP(torch.from_numpy(E).float()).generator, {'x': torch.Size([E.shape[0]])}, device='cuda:0')
    best_sol, best_obj, runtime, obj_log = go.QUBO(qubo['Q'], qubo['h'], qubo['const']).gurobi_optimize_QUBO(time_limit=time_limit, time_points=time_points, thread_num=thread_num, target_obj=target_obj, obj_log=[])
    for t, obj, sol in obj_log:
        sol = md.restore_variables(torch.tensor(sol),meta['index_map'])
        result = {'instance': Path(instance).stem, 'process':'QUBO', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    best_sol = md.restore_variables(torch.tensor(best_sol),meta['index_map'])
    results.append({'instance': Path(instance).stem, 'process':'QUBO', 'time':round(runtime,5), 'value': round(best_obj,2), 'constraint satisfaction': True, 'best known solution':target_obj})
    return results

def check_gcp_constraint(sols, graph):
    if sols.ndim == 2:
        sols.unsqueeze(0)
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

    return is_valid

def eval_gcp(instance, time_limit=60, target_obj=None, time_points=None, thread_num=8):

    E = (rf.GCP().read_file(instance))

    # MILP solver
    results = []
    best_sol, best_obj, runtime, obj_log = go.GCP(E).gurobi_optimize_MILP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=[])
    for t, obj, sol in obj_log:
        result = {'instance': Path(instance).stem, 'process':'MILP', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MILP', 'time':round(runtime,5), 'value': round(best_obj,2), 'constraint satisfaction': True, 'best known solution':target_obj})

    # MIQP solver
    best_sol, best_obj, runtime, obj_log = go.GCP(E).gurobi_optimize_MIQP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=[])  
    for t, obj, sol in obj_log:
        result = {'instance': Path(instance).stem, 'process':'MIQP', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MIQP', 'time':round(runtime,5), 'value': round(best_obj,2), 'constraint satisfaction': True, 'best known solution':target_obj})

    # QUBO solver
    sample = gn.GCP(torch.from_numpy(E).float(), coeff1=1, coeff2=1, coeff3=1)
    qubo, meta = md.get_qubo(sample.generator, {'x': torch.Size([E.shape[0], sample.num_color]), 'y':torch.Size([sample.num_color])}, device='cuda:0')
    best_sol, best_obj, runtime, obj_log = go.QUBO(qubo['Q'], qubo['h'], qubo['const']).gurobi_optimize_QUBO(time_limit=time_limit, time_points=time_points, thread_num=thread_num, target_obj=target_obj, obj_log=[])
    for t, obj, sol in obj_log:
        sol = md.restore_variables(torch.tensor(sol),meta['index_map'])
        result = {'instance': Path(instance).stem, 'process':'QUBO', 'time':round(t,5), 'value': round(obj,2), 'constraint satisfaction': check_gcp_constraint(sol['x'], torch.from_numpy(E).float()), 'best known solution':target_obj}
        results.append(result)
    best_sol = md.restore_variables(torch.tensor(best_sol),meta['index_map'])
    results.append({'instance': Path(instance).stem, 'process':'QUBO', 'time':round(runtime,5), 'value': round(best_obj,2), 'constraint satisfaction': check_gcp_constraint(sol['x'], torch.from_numpy(E).float()), 'best known solution':target_obj})
    return results


def eval_all_tsp(datasets_dir, amfd_results_dir, thread_num):
    # ディレクトリ内の.tspファイルをすべて取得
    tsp_files = [f for f in os.listdir(datasets_dir) if f.endswith('.tsp')]

    results = []
    for tsp_file in sorted(tsp_files):
        instance = os.path.join(datasets_dir, tsp_file)
        amfd_res = get_amfd_result(csv_file=os.path.join(amfd_results_dir, 'tsp_results.csv'), instance_name=Path(instance).stem)
        time_points = [t for t, v, best in amfd_res]
        best_known = amfd_res[-1][-1]
        results += eval_tsp(instance, time_limit=2*time_points[-1], target_obj=best_known, time_points=time_points, thread_num=thread_num)
        os.makedirs(os.path.dirname(__file__)+'/results', exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.dirname(__file__)+'/results/tsp_results.csv')
        print("-" * 60)

def eval_all_qap(datasets_dir, amfd_results_dir, thread_num):
    # ディレクトリ内の.tspファイルをすべて取得
    tsp_files = [f for f in os.listdir(datasets_dir) if f.endswith('.qap')]

    results = []
    for tsp_file in sorted(tsp_files):
        instance = os.path.join(datasets_dir, tsp_file)
        amfd_res = get_amfd_result(csv_file=os.path.join(amfd_results_dir, 'qap_results.csv'), instance_name=Path(instance).stem)
        time_points = [t for t, v, best in amfd_res]
        best_known = amfd_res[-1][-1]
        # 評価実行
        results += eval_qap(instance, time_limit=2*time_points[-1], target_obj=best_known, time_points=time_points, thread_num=thread_num)
        os.makedirs(os.path.dirname(__file__)+'/results', exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.dirname(__file__)+'/results/qap_results.csv')
        print("-" * 60)

def eval_all_misp(datasets_dir, amfd_results_dir, thread_num):
    # ディレクトリ内の.tspファイルをすべて取得
    tsp_files = [f for f in os.listdir(datasets_dir) if f.endswith('.clq')]

    results = []
    for tsp_file in sorted(tsp_files):
        instance = os.path.join(datasets_dir, tsp_file)
        amfd_res = get_amfd_result(csv_file=os.path.join(amfd_results_dir, 'misp_results.csv'), instance_name=Path(instance).stem)
        time_points = [t for t, v, best in amfd_res]
        best_known = -amfd_res[-1][-1]
        # 評価実行
        results += eval_misp(instance, time_limit=2*time_points[-1], target_obj=best_known, time_points=time_points, thread_num=thread_num)
        os.makedirs(os.path.dirname(__file__)+'/results', exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.dirname(__file__)+'/results/misp_results.csv')
        print("-" * 60)

def eval_all_mcp(datasets_dir, amfd_results_dir, thread_num):
    # ディレクトリ内の.tspファイルをすべて取得
    tsp_files = [f for f in os.listdir(datasets_dir) if f.endswith('.mcp')]

    results = []
    for tsp_file in sorted(tsp_files):
        instance = os.path.join(datasets_dir, tsp_file)
        amfd_res = get_amfd_result(csv_file=os.path.join(amfd_results_dir, 'mcp_results.csv'), instance_name=Path(instance).stem)
        time_points = [t for t, v, best in amfd_res]
        best_known = -amfd_res[-1][-1]
        # 評価実行
        results += eval_mcp(instance, time_limit=2*time_points[-1], target_obj=best_known, time_points=time_points, thread_num=thread_num)
        os.makedirs(os.path.dirname(__file__)+'/results', exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.dirname(__file__)+'/results/mcp_results.csv')
        print("-" * 60)

def eval_all_gcp(datasets_dir, amfd_results_dir, thread_num):
    # ディレクトリ内の.tspファイルをすべて取得
    tsp_files = [f for f in os.listdir(datasets_dir) if f.endswith('.col')]

    results = []
    for tsp_file in sorted(tsp_files):
        instance = os.path.join(datasets_dir, tsp_file)
        amfd_res = get_amfd_result(csv_file=os.path.join(amfd_results_dir, 'gcp_results.csv'), instance_name=Path(instance).stem)
        time_points = [t for t, v, best in amfd_res]
        best_known = amfd_res[-1][-1]
        # 評価実行
        results += eval_gcp(instance, time_limit=2*time_points[-1], target_obj=best_known, time_points=time_points, thread_num=thread_num)
        os.makedirs(os.path.dirname(__file__)+'/results', exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.dirname(__file__)+'/results/gcp_results.csv')
        print("-" * 60)


if __name__ == "__main__":
    datasets_dir = os.path.dirname(os.path.dirname(__file__)) + '/datasets'
    amfd_results_dir = os.path.dirname(os.path.dirname(__file__)) + '/fix'
    thread_num = 16

    eval_all_tsp(datasets_dir=datasets_dir + '/tsp', amfd_results_dir=amfd_results_dir, thread_num=thread_num)
    eval_all_qap(datasets_dir=datasets_dir + '/qap', amfd_results_dir=amfd_results_dir, thread_num=thread_num)
    eval_all_misp(datasets_dir=datasets_dir + '/misp', amfd_results_dir=amfd_results_dir, thread_num=thread_num)
    eval_all_mcp(datasets_dir=datasets_dir + '/mcp', amfd_results_dir=amfd_results_dir, thread_num=thread_num)
    # eval_all_gcp(datasets_dir=datasets_dir + '/gcp', amfd_results_dir=amfd_results_dir, thread_num=thread_num)




