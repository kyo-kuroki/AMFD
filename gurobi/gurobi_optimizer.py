import gurobipy as gp
from gurobipy import Model, quicksum, GRB
import numpy as np
import tqdm
from gurobipy import LinExpr, QuadExpr
import torch


def callback(time_limit=None, target_obj=None, time_points=None, obj_log=None):
    if time_points is None:
        time_points = []
    time_points = sorted(time_points)
    recorded = set()

    def my_callback(model, where):
        # RUNTIME を取得できる where のときのみ処理
        if where in [
            gp.GRB.Callback.MIP,
            gp.GRB.Callback.MIPSOL,
            gp.GRB.Callback.MIPNODE,
            gp.GRB.Callback.SIMPLEX,
            gp.GRB.Callback.BARRIER
        ]:
            runtime = model.cbGet(gp.GRB.Callback.RUNTIME)

            # 時間制限による打ち切り
            if time_limit is not None and runtime > time_limit:
                print(f"[{runtime:.2f}s] 時間制限 {time_limit} 秒を超えたので打ち切ります")
                model.terminate()
                return

            # 指定時刻での記録
            if obj_log is not None and where == gp.GRB.Callback.MIPSOL:
                for t in time_points:
                    if t not in recorded and runtime >= t:
                        try:
                            # obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                            obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                            
                            if obj != float('inf'):
                                # 解（変数名→値）の辞書を取得
                                # solution = {v.VarName: model.cbGetSolution(v) for v in model.getVars()}
                                vars = model.getVars()
                                solution = [model.cbGetSolution(v) for v in vars]
                                obj_log.append((t, obj, solution))
                                recorded.add(t)
                                print(f"[{runtime:.2f}s] {t:.1f} 秒で目的関数: {obj}")
                        except gp.GurobiError as e:
                             print(f"[{runtime:.2f}s] 解の取得エラー: {e}")

        # 目標値による打ち切り（MIPSOL時のみ）
        if target_obj is not None and where == gp.GRB.Callback.MIPSOL:
            obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            if obj <= target_obj:
                print(f"目標値 {target_obj} に達したので打ち切ります（現在: {obj}）")
                model.terminate()

    return my_callback




class QUBO():
    def __init__(self, Q, h, const=0):
        self.spin_num = len(h)
        if isinstance(Q, torch.Tensor):
            self.Q = Q.detach().cpu()
        else:
            self.Q = Q
        if isinstance(h, torch.Tensor):
            self.h = h.detach().cpu()
        else:
            self.h = h
        self.const = const
        self.energy_trans = []
        self.model = Model("QUBO")
        # バイナリ変数の追加
        self.x = []
        for i in range(self.spin_num):
            self.x.append(self.model.addVar(vtype=GRB.BINARY, name=f"x({i})"))

        # 目的関数の設定
        x_vars = np.array([self.x[i] for i in range(self.spin_num)])
        h = np.asarray(self.h)
        Q = np.asarray(self.Q)

        # 目的関数構築
        obj = QuadExpr()

        # 線形項
        obj.addTerms(h.tolist(), x_vars.tolist())

        # 二次項（上三角部分のみ）
        i_idx, j_idx = np.triu_indices(self.spin_num, k=1)
        q_vals = Q[i_idx, j_idx]

        mask = np.abs(q_vals) > 1e-10
        i_idx, j_idx, q_vals = i_idx[mask], j_idx[mask], q_vals[mask]

        obj.addTerms(q_vals.tolist(), x_vars[i_idx].tolist(), x_vars[j_idx].tolist())

        # 定数項
        if hasattr(self, "const") and self.const != 0:
            obj += self.const

        self.model.setObjective(obj, GRB.MINIMIZE)

        
    
    def gurobi_optimize_QUBO(self, time_limit=10, thread_num=0, target_obj=None, time_points=None, obj_log=None):

        # 時間制限の設定
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.setParam('Threads', thread_num)
        self.model._time_limit = time_limit

        # 最適化の実行
        cb = callback(time_limit=time_limit, target_obj=target_obj, time_points=time_points, obj_log=obj_log)
        self.model.optimize(cb)


        # 結果の取得
        solution = np.array([v.x for v in self.x])
        objective_value = self.model.objVal 
        runtime = self.model.Runtime

        return solution, objective_value, runtime, obj_log




class QAP:
    def __init__(self, flow=None, dist=None):
        self.Fij = flow
        self.Dij = dist
        self.factory_num = flow.shape[0]
        self.city_num = dist.shape[0]
    

    def gurobi_optimize_MIQP(self,time_limit=60, thread_num=0, target_obj=None, time_points=None, obj_log=None):
        self.model = Model("QAP")
        N = self.factory_num
        x = {}
        # 変数の定義
        for i in range(N):
            for j in range(N):
                x[i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"x[{i},{j}]")
        # 目的関数の定義

        Q = QuadExpr()
        terms = []
        vars1 = []
        vars2 = []

        for i in tqdm.tqdm(range(N)):
            for j in range(N):
                fij = self.Fij[i, j]
                for k in range(N):
                    for l in range(N):
                        coef = fij * self.Dij[k, l]
                        terms.append(coef)
                        vars1.append(x[i, k])
                        vars2.append(x[j, l])

        Q.addTerms(terms, vars1, vars2)
        self.model.setObjective(Q, GRB.MINIMIZE)

        # 制約の定義

        # 列ごとのワンホット制約（j固定）
        for j in tqdm.tqdm(range(N)):
            expr = LinExpr()
            vars_ = [x[i, j] for i in range(N)]
            coeffs = [1.0] * N
            expr.addTerms(coeffs, vars_)
            self.model.addConstr(expr == 1)

        # 行ごとのワンホット制約（i固定）
        for i in tqdm.tqdm(range(N)):
            expr = LinExpr()
            vars_ = [x[i, j] for j in range(N)]
            coeffs = [1.0] * N
            expr.addTerms(coeffs, vars_)
            self.model.addConstr(expr == 1)

        # パラメータ設定
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.setParam('Threads', thread_num)

        self.model._time_limit = time_limit

        # 実行
        self.model.optimize(callback(time_limit=time_limit, target_obj=target_obj, time_points=time_points, obj_log=obj_log))

        if self.model.SolCount > 0:
        # 結果の取得
            solution = np.zeros((N, N))  # N×Nの配列を初期化
            for i in range(N):
                for j in range(N):
                    solution[i, j] = x[i, j].x  # 各変数の値（解）を取得
            objective_value = self.model.objVal
            runtime = self.model.Runtime
        else: 
            solution = None
            objective_value = None
            runtime = self.model.Runtime

        return solution, objective_value, runtime, obj_log
    
    def gurobi_optimize_MILP(self,time_limit=60, thread_num=0, target_obj=None, time_points=None, obj_log=None):
        self.model = Model("QAP")
        N = self.factory_num
        V = list(range(N))
        M = np.einsum('ij,kl->ik', self.Fij, self.Dij)

        x, w = {}, {}
        for i in V:
            for j in V:
                w[i, j] = self.model.addVar(vtype="C", name=f"w[{i},{j}]")
                x[i, j] = self.model.addVar(vtype="B", name=f"x[{i},{j}]")
        self.model.update()


        # 列ごとのワンホット制約（j固定）
        for j in tqdm.tqdm(range(N)):
            expr = LinExpr()
            vars_ = [x[i, j] for i in range(N)]
            coeffs = [1.0] * N
            expr.addTerms(coeffs, vars_)
            self.model.addConstr(expr == 1)

        # 行ごとのワンホット制約（i固定）
        for i in tqdm.tqdm(range(N)):
            expr = LinExpr()
            vars_ = [x[i, j] for j in range(N)]
            coeffs = [1.0] * N
            expr.addTerms(coeffs, vars_)
            self.model.addConstr(expr == 1)

                
        for i in tqdm.tqdm(V):
            for k in V:
                linexpr = LinExpr()
                coeffs = []
                vars_ = []
                for j in V:
                    for ell in V:
                        coeffs.append(self.Fij[i,j] * self.Dij[k,ell])
                        vars_.append(x[j,ell])
                linexpr.addTerms(coeffs, vars_)
                lhs = M[i,k]*(x[i,k]-1) + linexpr
                self.model.addConstr(lhs <= w[i,k])

            
        self.model.setObjective(
            quicksum(
                w[i, k]
                for i in V
                for k in V
            ),
            GRB.MINIMIZE,
        )
        # パラメータ設定
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.setParam('Threads', thread_num)

        self.model._time_limit = time_limit

        # 実行
        self.model.optimize(callback(time_limit=time_limit, target_obj=target_obj, time_points=time_points, obj_log=obj_log))

        if self.model.SolCount > 0:

            # 結果の取得
            solution_x = np.zeros((N, N))  # N×Nの配列を初期化
            solution_w = np.zeros((N, N)) 
            for i in range(N):
                for j in range(N):
                    solution_x[i, j] = x[i, j].x  # 各変数の値（解）を取得
                    solution_w[i, j] = w[i, j].x

            solution = [solution_x, solution_w]
            objective_value = self.model.objVal
            runtime = self.model.Runtime
        
        else: 
            solution = None
            objective_value = None
            runtime = time_limit

        return solution, objective_value, runtime, obj_log
    
 


class TSP:
    def __init__(self, dist):
        self.Dij = dist
        self.city_num = self.Dij.shape[0] 

    def gurobi_optimize_MIQP(self, time_limit=60, thread_num=0, target_obj=None, time_points=None, obj_log=None):
        self.model = Model("TSP")
        N = self.city_num
        x = {}
        # 変数の定義
        for i in range(N-1):
            for j in range(N-1):
                x[i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"x[{i},{j}]")
        # 目的関数の定義(インデックスがN-1の都市に最初と最後に訪れる定式化で、時刻0に訪れる都市との距離と、時刻N-2に訪れる都市との距離を計算する)
        self.model.setObjective(quicksum(self.Dij[i, N-1] * x[0, i] + self.Dij[i, N-1] * x[N-2, i] for i in range(N-1)) \
                                + quicksum(self.Dij[i, j] * x[t, i] * x[t+1, j] for i in range(N-1) for j in range(N-1) for t in range(N-2)), GRB.MINIMIZE)
        # 制約の定義
        # 列ごとのワンホット制約
        for j in range(N - 1):
            expr = LinExpr()
            vars_ = [x[i, j] for i in range(N - 1)]
            coeffs = [1.0] * (N - 1)
            expr.addTerms(coeffs, vars_)
            self.model.addConstr(expr == 1)

        # 行ごとのワンホット制約
        for i in range(N - 1):
            expr = LinExpr()
            vars_ = [x[i, j] for j in range(N - 1)]
            coeffs = [1.0] * (N - 1)
            expr.addTerms(coeffs, vars_)
            self.model.addConstr(expr == 1)

        # パラメータ設定
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.setParam('Threads', thread_num)
        self.model._time_limit = time_limit

        # 実行
        cb = callback(time_limit=time_limit, target_obj=target_obj, time_points=time_points, obj_log=obj_log)
        self.model.optimize(cb)
        

        # 結果の取得
        if self.model.SolCount > 0:
            solution = np.zeros((N-1, N-1))  # N×Nの配列を初期化
            for i in range(N-1):
                for j in range(N-1):
                    solution[i, j] = x[i, j].x  # 各変数の値（解）を取得
            objective_value = self.model.objVal
            runtime = self.model.Runtime

        else: 
            solution = None
            objective_value = None
            runtime = self.model.Runtime

        return solution, objective_value, runtime, obj_log
    
    def gurobi_optimize_MILP(self, time_limit=60, thread_num=0, target_obj=None, time_points=None, obj_log=None):
        """scf: single-commodity flow formulation for the (asymmetric) traveling salesman problem
        Parameters:
            - n: number of nodes
            - c[i,j]: cost for traversing arc (i,j)
        Returns a model, ready to be solved.
        """
        n = self.city_num
        self.model = Model("TSP")

        x = self.model.addVars(
            [(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i != j],
            vtype="B", name="x"
        )

        f = {}
        for i, j in x.keys():
            ub = n - 1 if i == 1 else n - 2
            f[i, j] = self.model.addVar(lb=0, ub=ub, vtype="C", name=f"f({i},{j})")

        self.model.update()

        for i in range(1, n + 1):
            self.model.addConstr(
                quicksum(x[i, j] for j in range(1, n + 1) if j != i) == 1, "Out(%s)" % i
            )
            self.model.addConstr(
                quicksum(x[j, i] for j in range(1, n + 1) if j != i) == 1, "In(%s)" % i
            )

        self.model.addConstr(quicksum(f[1, j] for j in range(2, n + 1)) == n - 1, "FlowOut")

        for i in range(2, n + 1):
            self.model.addConstr(
                quicksum(f[j, i] for j in range(1, n + 1) if j != i)
                - quicksum(f[i, j] for j in range(1, n + 1) if j != i)
                == 1,
                "FlowCons(%s)" % i,
            )

        for j in range(2, n + 1):
            self.model.addConstr(f[1, j] <= (n - 1) * x[1, j], "FlowUB(%s,%s)" % (1, j))
            for i in range(2, n + 1):
                if i != j:
                    self.model.addConstr(f[i, j] <= (n - 2) * x[i, j], "FlowUB(%s,%s)" % (i, j))

        self.model.setObjective(quicksum(self.Dij[i-1, j-1] * x[i, j] for (i, j) in x), GRB.MINIMIZE)

        self.model.update()

        # パラメータ設定
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.setParam('Threads', thread_num)
        self.model._time_limit = time_limit

        # 実行
        cb = callback(time_limit=time_limit, target_obj=target_obj, time_points=time_points, obj_log=obj_log)
        self.model.optimize(cb)

        # 結果の取得
        if self.model.SolCount > 0:
            solution_x = np.zeros((n, n))  # N×Nの配列を初期化
            solution_f = np.zeros((n, n))  # N×Nの配列を初期化
            for i in range(1, n+1):
                for j in range(1, n+1):
                    if i != j:
                        solution_x[i-1, j-1] = x[i, j].x
                        solution_f[i-1, j-1] = f[i, j].x  # 各変数の値（解）を取得
            solution = [solution_x, solution_f]
            objective_value = self.model.objVal
            runtime = self.model.Runtime
        else: 
            solution = None
            objective_value = None
            runtime = self.model.Runtime
        return solution, objective_value, runtime, obj_log
    
    

        
# Max Cut Problem
class MCP:
    def __init__(self, edge):
        self.Eij = edge
        self.node_num = self.Eij.shape[0]

    
    def gurobi_optimize_MILP(self, time_limit=60, thread_num=0, target_obj=None, time_points=None, obj_log=None):
        """maxcut -- model for the graph maxcut problem
        Parameters:
            - V: set/list of nodes in the graph
            - E: set/list of edges in the graph
        Returns a model, ready to be solved.
        """
        V = list(range(self.node_num))

        self.model = Model("MCP")
        x = {}
        y = {}
        for i in V:
            x[i] = self.model.addVar(vtype="B", name=f"x({i})")
        for i in range(self.node_num):
            for j in range(i+1, self.node_num):
                if self.Eij[i, j] != 0:
                    y[i, j] = self.model.addVar(vtype="B", name=f"y({i},{j})")
        self.model.update()

        for i in range(self.node_num):
            for j in range(i+1, self.node_num):
                if self.Eij[i, j] != 0:
                    self.model.addConstr(x[i] + x[j] >= y[i, j], f"Edge({i},{j})")
                    self.model.addConstr(2 - x[j] - x[i] >= y[i, j], f"Edge({j},{i})")
                    self.model.addConstr(x[i] - x[j] <= y[i, j], f"EdgeLB1({i},{j})")
                    self.model.addConstr(x[j] - x[i] <= y[i, j], f"EdgeLB2({j},{i})")

        self.model.setObjective(-quicksum(self.Eij[i, j] * y[i, j] for i in range(self.node_num) for j in range(i+1, self.node_num) if self.Eij[i, j] != 0), GRB.MINIMIZE)

        self.model.update()

        # パラメータ設定
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.setParam('Threads', thread_num)
        self.model._time_limit = time_limit

        # 実行
        self.model.optimize(callback(time_limit=time_limit, target_obj=target_obj, time_points=time_points, obj_log=obj_log))

        # 結果の取得
        if self.model.SolCount > 0:
            solution_x = np.zeros(self.node_num)  # N×Nの配列を初期化
            for i in range(self.node_num):
                solution_x[i] = x[i].x  # 各変数の値（解）を取得

            solution_y = np.zeros((self.node_num, self.node_num)) 
            for i in range(self.node_num):
                for j in range(i+1, self.node_num):
                    if self.Eij[i, j] != 0:
                        solution_y[i, j] = y[i, j].x
            solution = [solution_x, solution_y]
            objective_value = self.model.objVal
            runtime = self.model.Runtime
        else: 
            solution = None
            objective_value = None
            runtime = self.model.Runtime

        return solution, objective_value, runtime, obj_log

    def gurobi_optimize_MIQP(self, time_limit=60, thread_num=0, target_obj=None, time_points=None, obj_log=None):
        """maxcut_MISOCP -- model for the graph maxcut problem
        Parameters:
            - V: set/list of nodes in the graph
            - E: set/list of edges in the graph
        Returns a model, ready to be solved.
        """
        self.model = Model("MCP")
        x, s, z = {}, {}, {}
        V = list(range(self.node_num))
        for i in V:
            x[i] = self.model.addVar(vtype="B", name=f"x({i})")
        for i in range(self.node_num):
            for j in range(i+1, self.node_num):
                if self.Eij[i, j] != 0:
                    s[i, j] = self.model.addVar(vtype="C", name=f"s({i},{j})")
                    z[i, j] = self.model.addVar(vtype="C", name=f"z({i},{j})")
        for i in range(self.node_num):
            for j in range(i+1, self.node_num):
                if self.Eij[i, j] != 0:
                    self.model.addConstr((x[i] + x[j] - 1) * (x[i] + x[j] - 1) <= s[i, j], f"S({i},{j})")
                    self.model.addConstr((x[j] - x[i]) * (x[j] - x[i]) <= z[i, j], f"Z({i},{j})")
                    self.model.addConstr(s[i, j] + z[i, j] == 1, f"P({i},{j})")

        self.model.setObjective(-quicksum(self.Eij[i, j] * z[i, j] for i in range(self.node_num) for j in range(i+1, self.node_num) if self.Eij[i, j] != 0), GRB.MINIMIZE)

        # パラメータ設定
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.setParam('Threads', thread_num)
        self.model._time_limit = time_limit

        # 実行
        self.model.optimize(callback(time_limit=time_limit, target_obj=target_obj, time_points=time_points, obj_log=obj_log))

        # 結果の取得
        if self.model.SolCount > 0:
            solution_x = np.zeros(self.node_num)  # N×Nの配列を初期化
            for i in range(self.node_num):
                solution_x[i] = x[i].x  # 各変数の値（解）を取得

            solution_s = np.zeros((self.node_num, self.node_num)) 
            solution_z = np.zeros((self.node_num, self.node_num)) 
            for i in range(self.node_num):
                for j in range(i+1, self.node_num):
                    if self.Eij[i, j] != 0:
                        solution_s[i, j] = s[i, j].x
                        solution_z[i, j] = z[i, j].x
            solution = [solution_x, solution_z, solution_s]
            objective_value = self.model.objVal
            runtime = self.model.Runtime
        else: 
            solution = None
            objective_value = None
            runtime = self.model.Runtime

        return solution, objective_value, runtime, obj_log


# Max Independent Set Problem (MISP) (equivalent to Max Clique Problem)
class MISP:
    def __init__(self, edge):
        self.Eij = edge
        self.node_num = self.Eij.shape[0]

    
    def gurobi_optimize_MILP(self, time_limit=60, thread_num=8, target_obj=None, time_points=None, obj_log=None):
        self.model = Model("MISP")
        N = self.node_num
        x = {}
        # 変数の定義
        for i in range(N):
            x[i] = self.model.addVar(vtype=GRB.BINARY, name=f"x[{i}]")
        # 目的関数の定義
        self.model.setObjective(-quicksum(x[i] for i in range(N)), GRB.MINIMIZE)
        E = 1 - self.Eij # 補グラフで考える
        # 制約の定義
        for i in range(N):
            for j in range(i+1, N):
                if E[i, j] == 1: # ベンチマークの都合上補グラフで考える
                    self.model.addConstr((x[i] + x[j]) <= 1)

        # パラメータ設定
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.setParam('Threads', thread_num)
        self.model._time_limit = time_limit

        # 実行
        self.model.optimize(callback(time_limit=time_limit, target_obj=target_obj, time_points=time_points, obj_log=obj_log))

        # 結果の取得
        if self.model.SolCount > 0:
            solution = np.zeros(N)  # Nの配列を初期化
            for i in range(N):
                solution[i] = x[i].x  # 各変数の値（解）を取得
            objective_value = self.model.objVal
            runtime = self.model.Runtime
        else: 
            solution = None
            objective_value = None
            runtime = self.model.Runtime

        return solution, objective_value, runtime, obj_log

    def gurobi_optimize_MIQP(self, time_limit=60, thread_num=8, target_obj=None, time_points=None, obj_log=None):
        self.model = Model("MISP")
        N = self.node_num
        x = {}
        # 変数の定義
        for i in range(N):
            x[i] = self.model.addVar(vtype=GRB.BINARY, name=f"x[{i}]")
        # 目的関数の定義
        self.model.setObjective(-quicksum(x[i] for i in range(N)), GRB.MINIMIZE)
        # 制約の定義
        for i in range(N):
            for j in range(i+1, N):
                if 1 - self.Eij[i, j] != 0: # 補グラフ
                    self.model.addConstr(x[i] * x[j] == 0)

        # パラメータ設定
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.setParam('Threads', thread_num)
        self.model._time_limit = time_limit

        # 実行
        self.model.optimize(callback(time_limit=time_limit, target_obj=target_obj, time_points=time_points, obj_log=obj_log))

        # 結果の取得
        if self.model.SolCount > 0:
            solution = np.zeros(N)  # Nの配列を初期化
            for i in range(N):
                solution[i] = x[i].x  # 各変数の値（解）を取得
            objective_value = self.model.objVal
            runtime = self.model.Runtime
        else: 
            solution = None
            objective_value = None
            runtime = self.model.Runtime

        return solution, objective_value, runtime, obj_log           
    

# Graph Coloring Problem (GCP) 
class GCP:
    def __init__(self, edge):
        self.Eij = edge
        self.node_num = self.Eij.shape[0]

    
    def gurobi_optimize_MIQP(self, time_limit, thread_num, target_obj=None, time_points=None, obj_log=None):
        self.model = Model('GCP')
        Ncolor = int(np.max(np.sum(self.Eij, axis=0))+1)
        Nvertex = self.node_num
        x = {}
        y = {}
        # 変数の定義
        for i in range(Nvertex):
            for j in range(Ncolor):
                x[i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"x[{i},{j}]")

        for i in range(Ncolor):
            y[i] = self.model.addVar(vtype=GRB.BINARY, name=f"y[{i}]")

        # 目的関数の定義
        self.model.setObjective(quicksum(y[i] for i in range(Ncolor)))

        # 制約関数の定義
        for i in range(Nvertex):
            self.model.addConstr(quicksum(x[i, j] for j in range(Ncolor)) == 1)  

        for i in range(Nvertex):
            for j in range(i+1, Nvertex):
                for k in range(Ncolor):
                    if self.Eij[i, j] != 0:
                        self.model.addConstr(x[i, k] * x[j, k] == 0) 

        for k in range(Ncolor):
            self.model.addConstr((1-y[k])*quicksum(x[i, k] for i in range(Nvertex)) == 0) 
        
        for k in range(Ncolor):
            self.model.addConstr(y[k] <= quicksum(x[i, k] for i in range(Nvertex))) 

        # パラメータ設定
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.setParam('Threads', thread_num)
        self.model._time_limit = time_limit

        # 実行
        self.model.optimize(callback(time_limit=time_limit, target_obj=target_obj, time_points=time_points, obj_log=obj_log))

        # 結果の取得
        if self.model.SolCount > 0:
            solution_x = np.zeros((Nvertex, Ncolor))  # N×Nの配列を初期化
            solution_y = np.zeros(Ncolor)  # N×Nの配列を初期化
            for i in range(Nvertex):
                for j in range(Ncolor):
                    solution_x[i, j] = x[i, j].x  # 各変数の値（解）を取得
            for i in range(Ncolor):
                solution_y = y[i]
            solution = [solution_x, solution_y]

            objective_value = self.model.objVal
            runtime = self.model.Runtime
        else: 
            solution = None
            objective_value = None
            runtime = self.model.Runtime

        return solution, objective_value, runtime, obj_log
    
    def gurobi_optimize_MILP(self, time_limit, thread_num, target_obj=None, time_points=None, obj_log=None):
        self.model = Model('GCP')
        Ncolor = int(np.max(np.sum(self.Eij, axis=0))+1)
        Nvertex = self.node_num
        x = {}
        y = {}
        # 変数の定義
        for i in range(Nvertex):
            for j in range(Ncolor):
                x[i, j] = self.model.addVar(vtype=GRB.BINARY, name=f"x[{i},{j}]")

        for i in range(Ncolor):
            y[i] = self.model.addVar(vtype=GRB.BINARY, name=f"y[{i}]")

        # 目的関数の定義
        self.model.setObjective(quicksum(y[i] for i in range(Ncolor)))

        # 制約関数の定義
        for i in range(Nvertex):
            self.model.addConstr(quicksum(x[i, j] for j in range(Ncolor)) == 1)  
        for i in range(Nvertex):
            for j in range(i+1, Nvertex):
                for k in range(Ncolor):
                    if self.Eij[i, j] != 0:
                        self.model.addConstr(x[i, k] + x[j, k] <= y[k]) 

        # パラメータ設定
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.setParam('Threads', thread_num)
        self.model._time_limit = time_limit

        # 実行
        self.model.optimize(callback(time_limit=time_limit, target_obj=target_obj, time_points=time_points, obj_log=obj_log))

        # 結果の取得
        if self.model.SolCount > 0:
            solution_x = np.zeros((Nvertex, Ncolor))  # N×Nの配列を初期化
            solution_y = np.zeros(Ncolor)
            for i in range(Nvertex):
                for j in range(Ncolor):
                    solution_x[i, j] = x[i, j].x  # 各変数の値（解）を取得
            for i in range(Ncolor):
                solution_y[i] = y[i].x
            solution = [solution_x, solution_y]

            objective_value = self.model.objVal
            runtime = self.model.Runtime
        else: 
            solution = None
            objective_value = None
            runtime = self.model.Runtime

        return solution, objective_value, runtime, obj_log
