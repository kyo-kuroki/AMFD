# AMFD for QUBO

## 概要

QUBOマシンは、様々な組合せ最適化問題の近似解を高速に求めることができる汎用的なソルバです。  
本プログラムは、Annealed Mean-field Descent (AMFD) アルゴリズムの PyTorch 実装と、そのデモコードを提供します。

```bash
.
├── generator.py     # 典型問題のQUBO定式化を行う関数群
├── mediator.py      # QUBO定式化された関数をQUBO行列に変換 / 解の再構成
├── read_file.py     # AMFD（Approximate Minimization via Fast Descent）最適化アルゴリズム
├── datasets
        ├── gcp
        ├── mcp
        ├── misp
        ├── qap
        └── tsp
├── amfd 
        ├── main.py
        └── optimizer.py
├── gurobi
        ├── main.py
        └── gurobi_optimizer.py
└── README.md        # 本ファイル
```

- **`generator.py`**  
  典型問題のQUBO定式化コードを記述

- **`mediator.py`**  
  定式化した関数からQUBO行列への変換、およびQUBOの解を元の変数のshapeに戻す変換コードを記述

- **`optimizer.py`**  
  AMFDアルゴリズムによるQUBO最適化の実装を記述

- **`example.ipynb`**  
  各モジュールの使い方を示すチュートリアルデモコードを記述

## インストール

```bash
git clone https://github.com/kyo-kuroki/qubo-machine.git
cd qubo-machine
pip install -r requirements.txt
```

## モジュールのimport
```bash
import mediator as md
import optimizer as op
import generator as gn
```

## 基本的な使い方

```bash

# QUBO関数の定義 (典型問題はgenerator.py内に作成済み)
def f(x: torch.Tensor, y: torch.Tensor):
    cost = x.sum() + (y**2).sum() # 任意のpytorch関数
    return cost

# AMFDの実行
result, energy = op.auto_grid_amfd(f, [x.shape], zeta_vals=[0, 1, 2, 5, 10, 20, 50], eta_vals=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], t_st=0.35, t_en=0.001, num_rep=4, Nstep=None, device='cuda:0')

# 解の取得
x_sol, y_sol = result
```

## QUBO行列に変換してから実行する方法

```bash
# 関数をQUBO行列に変換
qubo, meta = md.get_qubo_save_memory(f, {'x':(x.shape), 'y':(y.shape)}, device='cuda:0')
const, h, Q = qubo['const'], qubo['h'], qubo['Q']

# 最適化の実行
variables, energy = op.grid_amfd(const, h, Q, device='cuda:0', t_st=0.35, t_en=0.001, eta=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], zeta=[0, 1, 2, 5, 10, 20, 50], Nstep=h.shape[0], rep_num=4)

# 解の取得
x_sol = md.restore_variables(variables, meta['index_map'])['x']
y_sol = md.restore_variables(variables, meta['index_map'])['y']
```
