
import mediator as md
import read_file as rf
import generator as gn
import torch
import math
import time
import os
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

def check_tsp_constraint(x):
    is_valid = torch.allclose(x.sum(dim=0), torch.ones_like(x.sum(dim=0)), atol=1e-5) and \
        torch.allclose(x.sum(dim=1), torch.ones_like(x.sum(dim=1)), atol=1e-5)
    return is_valid


def eval_tsp(instance, time_limit=60, target_obj=None, time_points=None, thread_num=8, obj_log=[]):

    dists = (rf.TSP().read_file(instance))

    # MILP solver
    results = []
    best_sol, best_obj, runtime, obj_log = go.TSP(dists).gurobi_optimize_MILP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=obj_log)
    for t, obj, sol in obj_log:
        result = {'instance': Path(instance).stem, 'process':'MILP', 'time':round(t,5), 'value': round(obj,3), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MILP', 'time':round(runtime,5), 'value': round(best_obj,3), 'constraint satisfaction': True, 'best known solution':target_obj})

    # MIQP solver
    best_sol, best_obj, runtime, obj_log = go.TSP(dists).gurobi_optimize_MIQP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=obj_log)  
    for t, obj, sol in obj_log:
        result = {'instance': Path(instance).stem, 'process':'MIQP', 'time':round(t,5), 'value': round(obj,3), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MIQP', 'time':round(runtime,5), 'value': round(best_obj,3), 'constraint satisfaction': True, 'best known solution':target_obj})

    # QUBO solver
    qubo, meta = md.get_qubo_save_memory(gn.TSP(torch.from_numpy(dists).float()).generator, {'x': torch.Size([dists.shape[0]-1, dists.shape[0]-1])}, device='cuda:0')
    best_sol, best_obj, runtime, obj_log = go.QUBO(qubo['Q'], qubo['h'], qubo['const']).gurobi_optimize_QUBO(time_limit=time_limit, time_points=time_points, thread_num=thread_num, target_obj=target_obj, obj_log=obj_log)
    for t, obj, sol in obj_log:
        sol = md.restore_variables(torch.tensor(sol),meta['index_map'])
        result = {'instance': Path(instance).stem, 'process':'QUBO', 'time':round(t,5), 'value': round(obj,3), 'constraint satisfaction': check_tsp_constraint(sol['x']), 'best known solution':target_obj}
        results.append(result)
    best_sol = md.restore_variables(torch.tensor(best_sol),meta['index_map'])
    results.append({'instance': Path(instance).stem, 'process':'QUBO', 'time':round(runtime,5), 'value': round(best_obj,3), 'constraint satisfaction': check_tsp_constraint(best_sol['x']), 'best known solution':target_obj})
    return results

def eval_qap(instance, time_limit=60, target_obj=None, time_points=None, thread_num=8, obj_log=[]):

    flows, dists = (rf.QAP().read_file(instance))

    # MILP solver
    results = []
    best_sol, best_obj, runtime, obj_log = go.QAP(flows, dists).gurobi_optimize_MILP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=obj_log)
    for t, obj, sol in obj_log:
        result = {'instance': Path(instance).stem, 'process':'MILP', 'time':round(t,5), 'value': round(obj,3), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MILP', 'time':round(runtime,5), 'value': round(best_obj,3), 'constraint satisfaction': True, 'best known solution':target_obj})

    # MIQP solver
    best_sol, best_obj, runtime, obj_log = go.QAP(flows, dists).gurobi_optimize_MIQP(time_limit=time_limit, thread_num=thread_num, target_obj=target_obj, time_points=time_points, obj_log=obj_log)  
    for t, obj, sol in obj_log:
        result = {'instance': Path(instance).stem, 'process':'MIQP', 'time':round(t,5), 'value': round(obj,3), 'constraint satisfaction': True, 'best known solution':target_obj}
        results.append(result)
    results.append({'instance': Path(instance).stem, 'process':'MIQP', 'time':round(runtime,5), 'value': round(best_obj,3), 'constraint satisfaction': True, 'best known solution':target_obj})

    # QUBO solver
    qubo, meta = md.get_qubo_save_memory(gn.QAP(torch.from_numpy(flows).float(), torch.from_numpy(dists).float(), coeff1=1, coeff2=1).generator, {'x': torch.Size([dists.shape[0], dists.shape[0]])}, device='cuda:0')
    best_sol, best_obj, runtime, obj_log = go.QUBO(qubo['Q'], qubo['h'], qubo['const']).gurobi_optimize_QUBO(time_limit=time_limit, time_points=time_points, thread_num=thread_num, target_obj=target_obj, obj_log=obj_log)
    for t, obj, sol in obj_log:
        sol = md.restore_variables(torch.tensor(sol),meta['index_map'])
        result = {'instance': Path(instance).stem, 'process':'QUBO', 'time':round(t,5), 'value': round(obj,3), 'constraint satisfaction': check_tsp_constraint(sol['x']), 'best known solution':target_obj}
        results.append(result)
    best_sol = md.restore_variables(torch.tensor(best_sol),meta['index_map'])
    results.append({'instance': Path(instance).stem, 'process':'QUBO', 'time':round(runtime,5), 'value': round(best_obj,3), 'constraint satisfaction': check_tsp_constraint(best_sol['x']), 'best known solution':target_obj})
    return results


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



def eval_all_tsp(datasets_dir, thread_num):
    # ディレクトリ内の.tspファイルをすべて取得
    tsp_files = [f for f in os.listdir(datasets_dir) if f.endswith('.tsp')]

    results = []
    for tsp_file in sorted(tsp_files):
        # ファイル名から都市数（yyy部分）を抽出
        filename = os.path.splitext(tsp_file)[0]  # 'xxxyyy'
        try:
            num_cities = int(''.join(filter(str.isdigit, filename[-6:])))
        except ValueError:
            continue  # 都市数が取得できない場合はスキップ

        # 150都市以下のみ対象
        if num_cities <= 150:
            instance = os.path.join(datasets_dir, tsp_file)
            amfd_res = get_amfd_result(csv_file='/work2/k-kuroki/AMFD/edit/eddited_tsp_results.csv', instance_name=Path(instance).stem)
            time_points = [t for t, v, best in amfd_res]
            best_known = amfd_res[-1][-1]
            print(f"Evaluating {tsp_file} ({num_cities} cities)...")
            # 評価実行
            results += eval_tsp(instance, time_limit=5*time_points[-1], target_obj=best_known, time_points=time_points, thread_num=thread_num, obj_log=[])
            pd.DataFrame(results).to_csv(os.path.dirname(__file__)+'/results/tsp_results.csv')
            print("-" * 60)



# if __name__ == "__main__":
#     instance = f'/work2/k-kuroki/AMFD/datasets/tsp/burma14.tsp'
#     amfd_res = get_amfd_result(csv_file='/work2/k-kuroki/AMFD/edit/eddited_tsp_results.csv', instance_name=Path(instance).stem)
#     time_points = [t for t, v, best in amfd_res]
#     best_known = amfd_res[-1][-1]
#     target = amfd_res[-1][-2]
#     result = eval_tsp(instance, time_limit=5*time_points[-1], target_obj=best_known, time_points=time_points, thread_num=8, obj_log=[])
#     pd.DataFrame(result).to_csv(os.path.dirname(__file__)+'/results/gurobi.csv')

if __name__ == "__main__":

    eval_all_tsp(datasets_dir='/work2/k-kuroki/AMFD/datasets/tsp', thread_num=16)

#     eval_all_qap(
#         k=5,
#         genetic=True,
#         tuning_step_scale=2,
#         step_scales=[1, 2, 5, 10, 20, 50],
#         device='cuda:0',
#         seed=0
#     )

#     eval_all_misp(
#         k=5,
#         genetic=True,
#         tuning_step_scale=2,
#         step_scales=[1, 2, 5, 10, 20, 50],
#         device='cuda:0',
#         seed=0
#     )

#     eval_all_mcp(
#         k=5,
#         genetic=True,
#         tuning_step_scale=2,
#         step_scales=[1, 2, 5, 10, 20, 50],
#         device='cuda:0',
#         seed=0
#     )   

#     eval_all_gcp(
#         k=5,
#         genetic=True,
#         tuning_step_scale=2,
#         step_scales=[1, 2, 5, 10, 20, 50],
#         device='cuda:0',
#         seed=0
#     )


