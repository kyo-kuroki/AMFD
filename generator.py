import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import math
import torch.nn.functional
from collections import defaultdict, deque
import sympy as sp

class TSP():
    # input : 距離行列d, 制約係数1, 制約係数2
    def __init__(self, d, coeff1=1, coeff2=1, device='cuda'):
        self.d = d.to(device)
        self.coeff1 = (coeff1 * (self.d.mean(dim=0)).max()).to(device)
        self.coeff2 = (coeff2 * (self.d.mean(dim=0)).max()).to(device)
        self.num_city = self.d.shape[0]

    
    def generator(self, x):
        '''
        x[t][i]: torch.tensor (n-1, n-1), 時刻tに都市iにいるとき1, otherwise 0 (注意:最初と最後に訪問する場所は最後のインデックスの都市と決める)
        '''
        
        Const1 = self.coeff1 * ((1 - x.sum(dim=0))**2).sum()

        Const2 = self.coeff2 * ((1 - x.sum(dim=1))**2).sum()
        # 1) メインの和部分
        part1 = ((x[:self.num_city-2, :] @ self.d[:self.num_city-1, :self.num_city-1]) * x[1:self.num_city-1, :]).sum()

        # 2) 2つの単純な和
        part2 = torch.dot(self.d[:self.num_city-1, self.num_city-1], x[self.num_city-2, :])  # d[i, N] * x[T, i]
        part3 = torch.dot(self.d[:self.num_city-1, self.num_city-1], x[0, :])  # d[i, N] * x[0, i]

        Obj = part1 + part2 + part3

        H = Const1 + Const2 + Obj

        return H
    
    
    

    
    def get_route(self, spin_dim2):

        if spin_dim2.shape[0] != self.num_city:
            # (n+1)x(n+1) の新しい行列を作成（初期値 0）
            extended_spin = torch.zeros((self.num_city, self.num_city), dtype=spin_dim2.dtype)
            # 右下 (-1, -1) を 1 に設定 (最後に訪問する都市は決まっている)
            extended_spin[-1, -1] = 1
            # 元の spin を左上にコピー
            extended_spin[:-1, :-1] = spin_dim2
            spin_dim2 = extended_spin

        route = []
        for t in range(self.num_city):
            idx = torch.argmax(spin_dim2[t])
            route.append(idx)
        self.route = route
        self.total_distance = 0
        for t, r in enumerate(self.route):
            if t>0:
                self.total_distance += self.d[self.route[t-1]][r]
            else:
                self.total_distance += self.d[self.route[0]][self.route[-1]]
        return route
    

    def draw_route(self, spin_dim2, coordinate):
        coordinate = coordinate.to('cpu')
        self.route = self.get_route(spin_dim2)
        
        fig, ax = plt.subplots()
        try:
            for i in range(len(self.route)):
                key1 = self.route[i]
                key2 = self.route[(i + 1) % len(self.route)]  # 最後→最初のループ

                start = coordinate[key1]
                end = coordinate[key2]

                ax.arrow(start[0], start[1],
                        end[0] - start[0], end[1] - start[1],
                        head_width=0, head_length=0, fc='black', ec='black')

            # 座標の描画
            if isinstance(coordinate, dict):
                positions = list(coordinate.values())
            else:
                positions = coordinate

            for x, y in positions:
                ax.plot(x, y, color="red", marker='o')
            ax.set_title("Route")
            fig.show()
            return fig

        except Exception as error:
            print(error)
            print('DISPLAY DATAがありません!')

    
    

class QAP():
    def __init__(self, f, d, coeff1, coeff2, device='cuda'):

        self.factory_num = f.shape[0]
        self.city_num = d.shape[0]
        self.d = d.to(device)
        self.f = f.to(device)
        self.spin_num = self.factory_num**2
        # 工場iを都市kにおいた時のコストの平均をMikとする
        Mik = torch.zeros((self.city_num, self.city_num), device=self.d.device)
        d = self.d.sum(dim=0)
        f = self.f.sum(dim=0)
        for i in range(self.city_num):
            for k in range(self.city_num):
                Mik[i][k] = (1/(self.city_num-1)) * d[i] * f[k]
        # max(Mik)を制約係数とする
        self.coeff1 = coeff1 * (Mik).max()
        self.coeff2 = coeff2 * (Mik).max()


    def generator(self, x):
        """
        x: shape (factory_num, factory_num)
        工場 i が都市 k に割り当てられていれば 1、そうでなければ 0
        """

        # 定数制約項
        # Const1: 各都市にちょうど1つの工場が割り当てられる制約
        sum_over_i = torch.sum(x, dim=0)  # shape: (factory_num,)
        Const1 = self.coeff1 * torch.sum((1 - sum_over_i) ** 2)

        # Const2: 各工場がちょうど1つの都市に割り当てられる制約
        sum_over_k = torch.sum(x, dim=1)  # shape: (factory_num,)
        Const2 = self.coeff2 * torch.sum((1 - sum_over_k) ** 2)

        # 目的関数項: sum_{ijkl} F_ij * D_kl * x_ik * x_jl
        A = torch.einsum('ij,ik->jk', self.f, x)
        B = torch.einsum('kl,jl->jk', self.d, x)
        Obj = (A*B).sum()

        # 総合評価関数
        H = Const1 + Const2 + Obj

        return H
    



    def draw_graph(self, x, city_prefix="C", factory_prefix="F"):
        x = x.T # 元々i行j列目に工場iが都市jに配置される変数定義だったので修正
        """
        行列xに基づき、都市と工場の割当を2部グラフで可視化し、matplotlibのFigureを返す。

        Parameters:
        - x: 2次元numpy配列 (i行j列目が1なら都市iに工場jを割当)
        - city_prefix: 都市ノードの接頭辞（デフォルト: "City"）
        - factory_prefix: 工場ノードの接頭辞（デフォルト: "Factory"）

        Returns:
        - fig: matplotlib.figure.Figure オブジェクト
        """
        G = nx.Graph()
        num_cities, num_factories = x.shape

        # ノード名作成
        city_nodes = [f"{city_prefix}{i}" for i in range(num_cities)]
        factory_nodes = [f"{factory_prefix}{j}" for j in range(num_factories)]

        # ノード追加（2部に分類）
        G.add_nodes_from(city_nodes, bipartite=0)
        G.add_nodes_from(factory_nodes, bipartite=1)

        # エッジ追加（x[i,j] == 1 のとき）
        for i in range(num_cities):
            for j in range(num_factories):
                if x[i, j] == 1:
                    G.add_edge(city_nodes[i], factory_nodes[j])

        # レイアウト（都市を上、工場を下に並べる）
        pos = {}
        pos.update((node, (i, 1)) for i, node in enumerate(city_nodes))
        pos.update((node, (i, 0)) for i, node in enumerate(factory_nodes))

        # 描画
        fig, ax = plt.subplots(figsize=(8, 4))
        node_colors = ["skyblue" if n in city_nodes else "lightgreen" for n in G.nodes()]
        nx.draw(
            G, pos, with_labels=True, node_color=node_colors,
            node_size=5000/(max(6, num_cities + num_factories)), font_size=200/(max(6, num_cities + num_factories)), ax=ax
        )
        ax.set_title("Assignment", fontsize=10)
        ax.axis("off")

        return fig

class GCP():
    def __init__(self, E, coeff1=1, coeff2=1, coeff3=1, num_color=None, device='cuda'):
        self.E = E.float().to(device)
        self.num_node = E.shape[0]
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.coeff3 = coeff3
        if num_color == None:
            self.num_color = int((self.E.sum(dim=0)).max()+1)
        else: self.num_color = num_color

    def generator(self, x: torch.Tensor, y: torch.Tensor):
        """
        PyTorchを用いてQUBO目的関数を生成。
        
        Parameters:
            Eij: 隣接行列 (node_num x node_num), Eij[i, j] = 1 if edge(i,j)
            x: ノードiが色jに塗られるかのバイナリ変数 (node_num x color_num)
            y: 色jが使用されるかのバイナリ変数 (color_num,)
            coeff: ペナルティ係数

        Returns:
            H: QUBOの目的関数（スカラー値）
        """

        # 目的関数（使用された色の数を最小化）
        Obj = torch.sum(y)

        # Const1: 隣接ノードが同じ色だとペナルティ（x_i_k * x_j_k * E_ij）
        Const1 = self.coeff1 * torch.sum(x * (self.E @ x))

        # Const2: 色が使われたらy_jは1になっていないとペナルティ（Σ_i x_ij）*(1 - y_j)
        x_sum_per_color = torch.sum(x, dim=0)  # (color_num,)
        Const2 = 2 * self.coeff2 * torch.sum(x_sum_per_color * (1 - y))

        # Const3: 各ノードが1色だけに塗られるように制約（Σ_j x_ij = 1）
        color_sum_per_node = torch.sum(x, dim=1)  # (node_num,)
        Const3 = 2 * self.coeff3 * torch.sum((1 - color_sum_per_node) ** 2)

        # 合計ハミルトニアン
        H = Obj + Const1 + Const2 + Const3

        return H
    
    def draw_coloring(self, x: torch.Tensor):
        """
        ノードの色と接続を可視化する。

        Args:
            x (torch.Tensor): (num_nodes, num_colors) のバイナリテンソル。
                            x[i][j] = 1 → ノード i が色 j に塗られている。
            E (torch.Tensor): (num_nodes, num_nodes) の隣接行列。E[i][j] = 1 → エッジ(i,j)。

        Returns:
            fig: matplotlib の Figure オブジェクト
        """
        num_colors = int(torch.where(x.sum(dim=0)!=0, True, False).sum())
        x = x.detach().cpu().numpy()
        E = torch.where(self.E==1, True, False).cpu().numpy()
        num_nodes, _ = x.shape
        

        # ノードごとの色（最もスコアが高い色を選ぶ or 1のあるインデックス）
        node_colors = np.argmax(x, axis=1)  # 各ノードの色インデックス

        # ネットワークX グラフ作成
        G = nx.Graph()

        # ノードと色属性を追加
        for i in range(num_nodes):
            G.add_node(i, color=node_colors[i])

        # エッジ追加
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if E[i][j]:
                    G.add_edge(i, j)

        # ノード配置
        pos = nx.spring_layout(G, seed=42)

        # カラーマップ（num_colorsに対応）
        cmap = plt.get_cmap('tab10' if num_colors <= 10 else 'prism')

        # 可視化
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f'Coloring Graph (nodes:{num_nodes}, colors:{num_colors})', fontsize=10)
        nx.draw(
            G,
            pos,
            ax=ax,
            node_color=[cmap(c % cmap.N) for c in node_colors],
            with_labels=True,
            node_size=500,
            edge_color='gray',
            font_color='white'
        )

        return fig

class MISP():
    def __init__(self, E, coeff1=1, device='cuda'):
        self.device = device
        self.E = E.float().to(device)
        self.num_node = E.shape[0]
        self.coeff1 = coeff1 


    def generator(self, x: torch.Tensor):
        """
        PyTorchを用いてQUBO目的関数を生成。
        
        Parameters:
            Eij: 隣接行列 (node_num x node_num), Eij[i, j] = 1 if edge(i,j)
            x: ノードiが独立集合に含まれるかのバイナリ変数 (node_num)
            coeff: ペナルティ係数

        Returns:
            H: QUBOの目的関数（スカラー値）
        """

        # 目的関数
        Obj = -torch.sum(x)

        # Const1: 集合ノード間にエッジが張られていたらペナルティ（x_i * x_j * E_ij）
        Const1 = self.coeff1 * ((x @ self.E) * x).sum()

        # 合計ハミルトニアン
        H = Obj + Const1 

        return H
    
    def draw_independent_set(self, x: torch.Tensor):
        """
        ノードの色と接続を可視化する。

        Args:
            x (torch.Tensor): (num_nodes, num_colors) のバイナリテンソル。
                            x[i][j] = 1 → ノード i が色 j に塗られている。
            E (torch.Tensor): (num_nodes, num_nodes) の隣接行列。E[i][j] = 1 → エッジ(i,j)。

        Returns:
            fig: matplotlib の Figure オブジェクト
        """
        x = x.detach().cpu().numpy()
        E = torch.where(self.E==1, True, False).cpu().numpy()
        num_nodes = x.shape[0]
        

        # ネットワークX グラフ作成
        G = nx.Graph()

        # ノードと色属性を追加
        for i in range(num_nodes):
            G.add_node(i, color=x[i])

        # エッジ追加
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if E[i][j]:
                    G.add_edge(i, j)

        # ノード配置
        # pos = nx.spring_layout(G, seed=42)
        pos = nx.circular_layout(G)

        # カラーマップ（num_colorsに対応）
        cmap = plt.get_cmap('tab10')

        # 可視化
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f'Independent Set (nodes:{num_nodes}, independent set nodes:{int(x.sum())})', fontsize=10)
        nx.draw(
            G,
            pos,
            ax=ax,
            node_color=[cmap(0) if x[i].sum() > 0 else 'gray' for i in range(num_nodes)],
            with_labels=False,
            node_size=1000/num_nodes,
            edge_color='black',
            width=10/num_nodes,
            font_color='white'
        )

        return fig
    
class BPP():
    def __init__(self, weight, capacity, num_bins=None, bit_size=8, coeff1=1, coeff2=1, coeff3=1, soft_constraint=0.95):
        self.W = weight
        self.num_items = weight.shape[0]
        self.capacity = float(capacity)
        if num_bins == None:
            self.num_bins = int(math.ceil(self.num_items/(self.capacity/(self.W).max()))) # 最大で必要なビンの数
            # self.num_bins = int(self.greedy()[-1].sum())
        else: self.num_bins = num_bins
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.coeff3 = coeff3
        self.soft_constraint = soft_constraint
        self.bit_size = bit_size
        # ２の冪乗で表したときの最大値
        self.max_size = 2**self.bit_size - 1

    def greedy(self):
        """
        PyTorchベースのGreedy解法（first-fit）でビンにアイテムを詰める。
        結果は、(x, y) を返す:
        - x: (item_num, bin_num) バイナリ配置テンソル
        - y: (bin_num,) 使用されたビンに1、それ以外は0
        """
        device = self.W.device
        x = torch.zeros((self.num_items, self.num_bins), dtype=torch.float32, device=device)
        y = torch.zeros(self.num_bins, dtype=torch.float32, device=device)
        bin_remaining = torch.full((self.num_bins,), self.capacity, dtype=torch.float32, device=device)

        for i in range(self.num_items):
            placed = False
            for j in range(self.num_bins):
                if self.W[i] <= bin_remaining[j]:
                    x[i, j] = 1.0
                    y[j] = 1.0
                    bin_remaining[j] -= self.W[i]
                    placed = True
                    break
            if not placed:
                raise ValueError(f"Item {i} with weight {self.W[i]} cannot be placed in any bin. Increase num_bins or capacity.")

        return x, y
        
    def approx_generator(self, x: torch.Tensor, y: torch.Tensor):
        """
        PyTorchによるビンパッキング問題の目的関数（QUBO）計算。
        x: (item_num, bin_num) バイナリテンソル
        y: (bin_num,) バイナリテンソル
        z: (bin_num, bit_size) バイナリテンソル
        """
        W = self.W/self.W.mean()
        capacity = self.capacity/self.W.mean()

        # 目的関数: 使用したビンの数
        Obj = y.sum()

        # 制約1: 各アイテムは1つのビンにだけ
        H_const1 = ((1 - x.sum(dim=1)) ** 2).sum()

        # 制約2: ビンの容量制約
        weighted_x = (x * W.view(-1, 1)).sum(dim=0)  # (bin_num,)
        capacity_term = (weighted_x - self.soft_constraint*capacity*y)**2
        H_const2 = (capacity_term).sum()

        # 制約3
        H_const3 = ((1-y) * x.sum(dim=0)).sum()

        # 合計QUBO目的関数
        H_total = Obj + 2 * (self.coeff1*H_const1 + self.coeff2*H_const2 + self.coeff3*H_const3)
        return H_total
    
    def nonpoly_generator(self, x: torch.Tensor, y: torch.Tensor):
        """
        PyTorchによるビンパッキング問題の目的関数（QUBO）計算。
        x: (item_num, bin_num) バイナリテンソル
        y: (bin_num,) バイナリテンソル
        z: (bin_num, bit_size) バイナリテンソル
        """
        W = self.W/self.W.mean()
        capacity = self.capacity/self.W.mean()

        # 目的関数: 使用したビンの数
        Obj = y.sum()

        # 制約1: 各アイテムは1つのビンにだけ
        H_const1 = ((1 - x.sum(dim=1)) ** 2).sum()

        # 制約2: ビンの容量制約
        weighted_x = (x * W.view(-1, 1)).sum(dim=0)  # (bin_num,)
        # capacity_term = torch.nn.functional.relu(weighted_x - capacity)/W.min()
        diff = weighted_x - capacity
        capacity_term = diff * torch.sigmoid(diff)
        H_const2 = (capacity_term).sum()

        # 制約3
        H_const3 = ((1-y) * x.sum(dim=0)).sum()

        # 合計QUBO目的関数
        H_total = Obj + 2 * (self.coeff1*H_const1 + self.coeff2*H_const2 + self.coeff3*H_const3)
        return H_total

    def generator(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """
        PyTorchによるビンパッキング問題の目的関数（QUBO）計算。
        x: (item_num, bin_num) バイナリテンソル
        y: (bin_num,) バイナリテンソル
        z: (bin_num, bit_size) バイナリテンソル
        """
        device = x.device
        W = self.W/self.W.mean()
        capacity = self.capacity/self.W.mean()

        # 目的関数: 使用したビンの数
        Obj = y.sum()

        # 制約1: 各アイテムは1つのビンにだけ
        H_const1 = ((1 - x.sum(dim=1)) ** 2).sum()

        # 制約2: ビンの容量制約 & ビンを使用しない場合(y[j]=0)そのビンに詰めてはならない(sum_j x[i,j] = 0)
        powers_of_two = torch.tensor([2 ** k for k in range(self.bit_size)], device=device)
        z_weighted = (z * powers_of_two).sum(dim=1)  # (bin_num,)
        weighted_x = (x * W.view(-1, 1)).sum(dim=0)  # (bin_num,)
        capacity_term = weighted_x + capacity * z_weighted/self.max_size - capacity 
        H_const2 = (capacity_term ** 2).sum()

        # 制約3 (ビンを使わないときはそのビンに入っている重量が0)
        H_const3 = ((1-y) * x.sum(dim=0)).sum()

        # 合計QUBO目的関数
        H_total = Obj + 2*self.coeff1*H_const1 + self.coeff2*H_const2 + 2*self.coeff3*H_const3
        return H_total
    
    def draw_bin_packing_solution(self, x: torch.Tensor):
        """
        ビンパッキングの解を積み上げ棒グラフで描画する。

        Args:
            x (torch.Tensor): shape = (num_items, num_bins)、x[i,j]=1ならアイテムiはビンjに割り当てられている。
            w (torch.Tensor): shape = (num_items,)、各アイテムの重量。

        Returns:
            fig (matplotlib.figure.Figure): 描画された図。
        """
        w = self.W
        num_items, num_bins = x.shape
        fig, ax = plt.subplots(figsize=(max(6, num_bins), 6))

        # 各ビンの位置
        bin_positions = range(num_bins)
        # 各ビンの現在の積み上げ高さ（バーの底）
        bin_bottoms = [0] * num_bins

        # アイテムごとにバーを描画
        for i in range(num_items):
            for j in range(num_bins):
                if x[i, j].item() == 1:
                    weight = w[i].item()
                    ax.bar(j, weight, bottom=bin_bottoms[j], width=0.8)
                    # ラベル：アイテム番号（重量）
                    label = f"i{i}"
                    ax.text(j, bin_bottoms[j] + weight / 2, label,
                            ha='center', va='center', fontsize=8, color='white')
                    bin_bottoms[j] += weight

        ax.set_xlabel('Bin Index')
        ax.set_ylabel('Total Weight')
        ax.set_title('Bin Packing Solution')
        ax.set_xticks(list(bin_positions))
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        return fig
    

class QKP():
    def __init__(self, weight, capacity, value, co_value, bit_size=8, coeff1=1, soft_constraint=0.95):
        self.W = weight
        self.num_items = weight.shape[0]
        self.capacity = float(capacity)
        self.value = value
        self.co_value = co_value
        self.coeff1 = coeff1 * ((self.value + self.co_value.mean(dim=1))/self.W).mean() # 重さあたりの価値
        self.soft_constraint = soft_constraint
        self.bit_size = bit_size
        # ２の冪乗で表したときの最大値
        self.max_size = 2**self.bit_size - 1

    def generator(self, x: torch.Tensor, y: torch.Tensor):
        """
        PyTorchによるビンパッキング問題の目的関数（QUBO）計算。
        x: (item_num) バイナリテンソル
        y: (bit_size) バイナリテンソル
        """
        device = x.device

        # 目的関数: 使用したビンの数
        Obj = -(self.value*x).sum() - 0.5*((x@self.co_value)*x).sum()


        # 制約1: 合計容量制約
        powers_of_two = torch.tensor([2 ** k for k in range(self.bit_size)], device=device)
        y_weighted = (y * powers_of_two).sum()  # (bin_num,)
        total_weight = (x * self.W).sum()  # (bin_num,)
        capacity_term = total_weight + self.capacity * y_weighted/self.max_size - self.capacity 
        H_const1 = self.coeff1 * (capacity_term ** 2).sum()

        # 合計QUBO目的関数
        H_total = Obj + 2 * (self.coeff1*H_const1)
        return H_total


# シフトスケジューリング問題
class SSP():
    def __init__(self, cost: torch.Tensor, feasible_job:torch.Tensor, job_time:list, coeff1=1, coeff2=1, coeff3=1):
        self.cost = cost
        self.job_time = job_time
        self.feasible_job = feasible_job
        self.num_workes, self.num_jobs = self.feasible_job.shape
        self.G = (1.0 - self.create_conflict_graph(job_time)).float().to(feasible_job.device) # 実行時刻が被っているジョブ(衝突しているジョブ)に対して1, そうでない場合0(ただし同じジョブに対しては0)
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.coeff3 = coeff3


    def create_conflict_graph(self, jobs):
        """
        jobs: list of dicts, each with keys "job", "start", "end"
            e.g., [{"job": 0, "start": 1, "end": 5}, {"job": 1, "start": 3, "end": 6}]
        
        Returns:
            G: [num_jobs, num_jobs] tensor, where
            G[i, j] = 1 if job i and job j can be executed simultaneously (no time overlap),
            G[i, j] = 0 otherwise
        """
        num_jobs = len(jobs)
        G = torch.zeros((num_jobs, num_jobs), dtype=torch.int32)

        # 時間を抽出
        start_times = torch.tensor([job["start"] for job in jobs], dtype=torch.float32)
        end_times = torch.tensor([job["end"] for job in jobs], dtype=torch.float32)

        # job i と job j が衝突しない条件:
        # end[i] <= start[j] or end[j] <= start[i]
        no_conflict = (end_times[:, None] <= start_times[None, :]) | (end_times[None, :] <= start_times[:, None])

        # 衝突しないなら1、衝突するなら0
        G = no_conflict.to(dtype=torch.int32)

        # 対角成分は常に1（自身とは非衝突とみなす）
        torch.diagonal(G).fill_(1).float()

        return G

    def generator(self, x, y):
        """
        feasible_job: [num_workers, num_jobs]  可否行列（ワーカーがジョブを処理できるか）
        G: [num_jobs, num_jobs]     ジョブ同士の衝突（同時実行可否）グラフ
        x: [num_workers, num_jobs]  割当変数
        y: [num_workers]            ワーカー使用変数
        cost: [num_workers]            ワーカー固定コスト
        """


        # --- 項1: 総ワーカーコスト ---
        term1 = (self.cost * y).sum()  # 基本的にcostはみんな1で統一すれば人数に対応する(人数の最小化)

        # --- 項2: 各ジョブはちょうど実行可能なワーカーのうち1人に割り当てられる ---
        # 各ジョブについてワーカーごとのxの和をとる
        job_assignment = torch.sum(x * self.feasible_job, dim=0)  # [num_jobs]
        term2 = self.coeff1 * torch.sum((job_assignment - 1.0) ** 2)

        # --- 項3: 同時に処理できないジョブを同じワーカーに割り当てない ---
        # G_ij = 1 なら、i, j は同時実行不可 → 同じワーカーに両方割り当てるとペナルティ

        term3 = 0.5 * torch.einsum('ij,ki,kj->', self.G, x, x)

        # --- 項4 : ワーカーiを使っていたら、y[i]は1にならなければならない
        term4 = ((1 - y) * x.sum(dim=1)).sum()

        # --- 最終目的関数 ---
        loss = term1 + 2 * (self.coeff1 * term2 + self.coeff2 * term3 + self.coeff3 * term4)
        return loss
    
    
    def draw_shift_schedule(self, x):
        job_time = self.job_time

        num_workers, num_jobs = x.shape
        job_start = [job_time[j]["start"] for j in range(num_jobs)]
        job_end = [job_time[j]["end"] for j in range(num_jobs)]
        job_duration = [e - s for s, e in zip(job_start, job_end)]
        max_time = max(job_end)

        fig_width = max(8, max_time * 0.4)
        fig, ax = plt.subplots(figsize=(fig_width, max(4, 0.2 * num_workers)))

        colors = plt.cm.get_cmap('tab10', num_workers)

        for i in range(num_workers):
            for j in range(num_jobs):
                if x[i, j].item() > 0.5:
                    start = job_start[j]
                    duration = job_duration[j]
                    ax.add_patch(patches.Rectangle(
                        (start, num_workers - i - 1),
                        duration,
                        0.8,
                        color=colors(i),
                        edgecolor='black'
                    ))
                    ax.text(start + duration / 2, num_workers - i - 0.6,
                            f"J{j}", ha='center', va='center', fontsize=8, color='white')

        ax.set_yticks(range(num_workers))
        ax.set_yticklabels([f"Worker {i}" for i in reversed(range(num_workers))])
        ax.set_xlabel("Time")
        ax.set_title("Shift Schedule")
        ax.set_xlim(0, max_time + 1)
        ax.set_ylim(-0.5, num_workers)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)

        # 調整ポイント
        plt.margins(x=0.01)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

        return fig

# ジョブショップスケジューリング問題(JSP: jobshop scheduring problem)
class JSP():
    def __init__(self, use_time, use_machine, transitive=True, coeff1=1, coeff2=1, coeff3=1):
        self.T = use_time # T[i,j]: タスクiのj番目のジョブの所要時間(ジョブがない場合は0を格納)
        self.M = use_machine # M[i,j]:タスクiのj番目のジョブが使用するマシンインデックス(ジョブがない場合は-1を格納)
        self.coeff1, self.coeff2, self.coeff3 = coeff1 * self.T.sum(dim=1).max(), coeff2 * self.T.sum(dim=1).max(), coeff3
        if transitive:
            self.fixed = self.transitive_fixed_tensor().to(self.T.device) # 固定されている要素が-1
        else:
            self.fixed = self.build_fixed_tensor().to(self.T.device) # 固定されている要素が-1
        self.num_jobs, self.max_num_jobs = use_time.shape
        self.num_x_variables = (self.fixed == -1).sum().item()
        self.num_x_symmetric_variables = int(self.num_x_variables/2)
        # self.num_x_symmetric_variables = self.count_symmetric_x_variables()


    def most_used_machine_count(self) -> int:
        """
        M: [I, J] LongTensor - 各タスクに割り当てられたマシン番号 (-1 は未使用)
        return: int - 最も多く使われたマシンの使用回数
        """
        valid_machines = self.M[self.M >= 0].int()  # 無効な -1 を除去
        if valid_machines.numel() == 0:
            return 0  # すべて未使用なら0
        counts = torch.bincount(valid_machines)
        return counts.float().mean().item()
    
    def count_free_variables_unique(self, fixed, M):
        """
        fixed: (N, L, N, L) tensor with {-1, 0, 1}
        M: (N, L) マシンインデックス
        """
        N, L = fixed.shape[:2]
        free = (fixed == -1)

        count = 0
        for i in range(N):
            for j in range(L):
                if M[i, j] == -1:
                    continue
                for k in range(i+1, N):  # i < k に制限（対称性）
                    if M[k, j] == -1:
                        continue
                    if M[i, j] != M[k, j]:
                        continue
                    # 同じマシン、同じジョブ位置jで未固定なら加算
                    if free[i, j, k, j]:
                        count += 1
        return count
    
    def count_symmetric_x_variables(self):
        """
        M: (N, L) マシンインデックス (int), 未使用ジョブは -1

        戻り値:
            同じマシンで異なるタスク間にあるジョブペア数
        """
        M = self.M
        M_valid = M[M != -1]
        total_combs = 0
        same_task_combs = 0

        # --- 1. 全体の n_k C_2 を計算 ---
        unique_machines, counts = torch.unique(M_valid, return_counts=True)
        total_combs = (counts * (counts - 1) // 2).sum()

        # --- 2. 同一タスク内で同じマシンを使っているペア数を除く ---
        N, L = M.shape
        for i in range(N):
            row = M[i]
            row_valid = row[row != -1]
            if len(row_valid) < 2:
                continue
            unique_r, counts_r = torch.unique(row_valid, return_counts=True)
            same_task_combs += (counts_r * (counts_r - 1) // 2).sum()

        return (total_combs - same_task_combs).item()

    def x_reshape_dim1_to_dim4(self, x_dim1):
        # 1. マスク作成（None の位置）
        mask = (self.fixed == -1)

        # 2. flatten（1次元化）
        flat_fixed = self.fixed.flatten().float()

        # 3. None のインデックスを取得
        idx = mask.flatten().nonzero(as_tuple=True)[0]

        # 4. scatter で None 部分に x を埋め込む（ループなし）
        flat_fixed[idx] = x_dim1.to(flat_fixed.device)

        # 5. 元の形に戻す
        return flat_fixed.view(self.fixed.shape)
    

    
    def x_reshape_dim1_to_dim4_symmetric(self, x_dim1):
        """
        ベクトル化して -1 の対称変数に x_dim1 の値を代入する高速実装。
        - x[i,j,k,l] = x_dim1[m], かつ x[k,l,i,j] = 1 - x_dim1[m]
        - 自己ループ (i,j == k,l) は x[i,j,i,j] = x_dim1[m]

        Parameters:
            x: [N, L, N, L] テンソル（-1が未定義）
            x_dim1: [num_free_vars] テンソル、自由変数（対称性考慮後）

        Returns:
            x_filled: 同じサイズのテンソル（すべての -1 が埋まる）
        """
        x = self.fixed.clone().float()
        N, L = x.shape[0], x.shape[1]

        # 全てのインデックスを生成
        i, j, k, l = torch.meshgrid(
            torch.arange(N, device=x.device), torch.arange(L, device=x.device),
            torch.arange(N, device=x.device), torch.arange(L, device=x.device),
            indexing='ij'
        )

        # -1 である箇所を抽出（固定されていない変数）
        mask = (x == -1)

        # 対称性の代表条件: (i,j) <= (k,l) in 辞書順
        flat_ij = i * L + j
        flat_kl = k * L + l
        is_rep = (flat_ij < flat_kl) | ((flat_ij == flat_kl) & (i <= k))

        # 自由変数の代表インデックスマスク
        rep_mask = mask & is_rep

        # 代表インデックスの位置を取り出し（フラットに）
        rep_idx = torch.nonzero(rep_mask, as_tuple=False)  # shape [num_free_vars, 4]
        num_vars = rep_idx.size(0)
        assert num_vars == len(x_dim1), f"x_dim1の長さ {len(x_dim1)} ≠ 自由変数数 {num_vars}"

        # 対称変数の割当：x[i,j,k,l] = x_dim1, x[k,l,i,j] = 1 - x_dim1
        i1, j1, k1, l1 = rep_idx[:, 0], rep_idx[:, 1], rep_idx[:, 2], rep_idx[:, 3]
        x[i1, j1, k1, l1] = x_dim1.float()
        x[k1, l1, i1, j1] = 1 - x_dim1.float()

        return x


    def generator(self, x):
        """
        T[i,j]: タスクiのj番目のジョブの所要時間
        M[i,j]: タスクiのj番目のジョブが使うマシンインデックス
        x: [N^4] → reshape後 [i,j,k,l]
        """
        T, M = self.T, self.M
        N, L = T.shape
        x = self.x_reshape_dim1_to_dim4_symmetric(x)  # shape: [N, L, N, L]
        

        # --- term1: 中間ノードの処理時間 T[i,j] * x[i,j,k,l]
        paths_to_ij = x.sum(dim=(0,1))     # shape: (I,J)
        paths_from_ij = x.sum(dim=(2,3))   # shape: (I,J)

        path_count_ij = paths_to_ij * paths_from_ij  # shape: (I,J) ノード（i,j)を通る経路が最低何通りあるか

        H_obj = (path_count_ij * T).sum()


        # --- 巡回路除去制約 
        # --- 1. 各ジョブのマシンID展開 ---
        M1 = M.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)  # [N, L, 1, 1, 1, 1]
        M2 = M.unsqueeze(0).unsqueeze(1).unsqueeze(4).unsqueeze(5)  # [1, 1, N, L, 1, 1]
        M3 = M.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [1, 1, 1, 1, N, L]

        # --- 2. 有効なジョブ（M ≠ -1）かを判定 ---
        valid1 = (M1 != -1)
        valid2 = (M2 != -1)
        valid3 = (M3 != -1)
        all_valid = valid1 & valid2 & valid3

        # --- 3. インデックス抽出 ---
        idx = all_valid.nonzero(as_tuple=False)  # [K, 6]

        i, j, k, l, m, n = idx.T

        # --- 4. x変数の抽出 ---
        x1 = x[i, j, k, l]
        x2 = x[k, l, m, n]
        x3 = x[m, n, i, j]

        # --- 5. 閉路ペナルティ項 ---
        term = 1 - (x1 + x2 + x3) + (x1 * x2 + x2 * x3 + x3 * x1)
        H_const1 = (term).sum()

        return  self.coeff1 * H_const1 + H_obj
    
    def approx_generator_symmetric(self, x):
        """
        T[i,j]: タスクiのj番目のジョブの所要時間
        M[i,j]: タスクiのj番目のジョブが使うマシンインデックス
        x: [N^4] → reshape後 [i,j,k,l]
        """

        T, M = self.T, self.M
        N, L = T.shape
        x = self.x_reshape_dim1_to_dim4_symmetric(x)  # shape: [N, L, N, L]
        

        # --- term1: 中間ノードの処理時間 T[i,j] * x[i,j,k,l]
        term1 = (T + torch.einsum('kl,klij->ij', T, x) + torch.einsum('kl,ijkl->ij', T, x)).sum()

        # --- term2: 前段ノードの累積 T[q,s] * x[q,r,i,j] * x[i,j,k,l]
        S_pre = T.cumsum(dim=1)  # [N, L]
        term2 = torch.einsum('qr,qrij,ijkl->', S_pre, x, x)

        # --- term3: 後段ノードの累積 T[m,p] * x[i,j,k,l] * x[k,l,m,n] with p >= n
        S_post = torch.flip(torch.cumsum(torch.flip(T, dims=[1]), dim=1), dims=[1])  # [N, L]
        term3 = torch.einsum('mn,ijkl,klmn->', S_post, x, x)

        # --- 巡回除去制約

        # --- 1. マシンIDの展開 ---
        M1 = M.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)  # [N, L, 1, 1, 1, 1]
        M2 = M.unsqueeze(0).unsqueeze(1).unsqueeze(4).unsqueeze(5)  # [1, 1, N, L, 1, 1]
        M3 = M.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [1, 1, 1, 1, N, L]

        # --- 2. 有効なタスク判定 ---
        valid1 = (M1 != -1)
        valid2 = (M2 != -1)
        valid3 = (M3 != -1)
        all_valid = valid1 & valid2 & valid3

        # --- 3. 同じマシンかつ異なるタスク ---
        same_machine = (M1 == M2) & (M2 == M3)

        task_ids = torch.arange(N).to(M.device)
        I1 = task_ids.view(-1, 1, 1, 1, 1, 1)
        I2 = task_ids.view(1, 1, -1, 1, 1, 1)
        I3 = task_ids.view(1, 1, 1, 1, -1, 1)

        different_tasks = (I1 != I2) & (I2 != I3) & (I3 != I1)

        # --- 5. 最終マスク ---
        mask = same_machine & all_valid & different_tasks   # [N,L,N,L,N,L]
        
        # --- 6. インデックス抽出 ---
        idx = mask.nonzero(as_tuple=False)  # [K, 6]

        i, j, k, l, m, n = idx.T

        # --- 7. x変数の抽出 ---
        x1 = x[i, j, k, l]
        x2 = x[k, l, m, n]
        x3 = x[m, n, i, j]

        # --- 8. 閉路ペナルティ項 ---
        H_const1 = (1 - (x1 + x2 + x3) + (x1 * x2 + x2 * x3 + x3 * x1)).sum()

        # --- 4点巡回除去 ----

        # マスク作成
        i = torch.arange(N, device=x.device).view(-1, 1)
        k = torch.arange(N, device=x.device).view(1, -1)
        i_lt_k_mask = (i < k).float()  # [N, N]

        m = torch.arange(L, device=x.device).view(-1, 1)
        j = torch.arange(L, device=x.device).view(1, -1)
        m_lt_j_mask = (m < j).float()  # [L, L]

        l = torch.arange(L, device=x.device).view(-1, 1)
        n = torch.arange(L, device=x.device).view(1, -1)
        l_lt_n_mask = (l < n).float()  # [L, L]

        # x1: [N, L, N, L] → [N, L, N, L, 1, 1]
        x1 = x.unsqueeze(4).unsqueeze(5)

        # x2: [N, L, N, L] → [1, 1, N, L, L, L]
        x2 = x.permute(2, 3, 0, 1).unsqueeze(0).unsqueeze(1)

        # ブロードキャストされた形で掛け算できるように
        # 結果: [N, L, N, L, L, L]
        product = x1 * x2

        # マスク結合
        mask_ik = i_lt_k_mask.view(N, 1, N, 1, 1, 1).expand(N,L,N,L,N,L)
        mask_mj = m_lt_j_mask.view(1, L, 1, L, 1, 1).expand(N,L,N,L,N,L)
        mask_ln = l_lt_n_mask.view(1, 1, 1, L, 1, L).expand(N,L,N,L,N,L)
        mask = mask_ik * mask_mj * mask_ln

        H_const2 = (product * mask).sum()


        return  term1 + term2 + term3 + self.coeff1 * H_const1 + self.coeff2 * H_const2

    
    def approx_generator_asymmetric(self, x):
        """
        T[i,j]: タスクiのj番目のジョブの所要時間
        M[i,j]: タスクiのj番目のジョブが使うマシンインデックス
        x: [N^4] → reshape後 [i,j,k,l]
        """

        T, M = self.T, self.M
        N, L = T.shape
        x = self.x_reshape_dim1_to_dim4(x)  # shape: [N, L, N, L]
        

        # --- term1: 中間ノードの処理時間 T[i,j] * x[i,j,k,l]
        S_pre = T.cumsum(dim=1)  # [N, L]
        S_post = torch.flip(torch.cumsum(torch.flip(T, dims=[1]), dim=1), dims=[1])  # [N, L]
        # term1 = (T + torch.einsum('kl,klij->ij', S_pre, x) + torch.einsum('kl,ijkl->ij', S_post, x)).sum()
        term1 = (T + torch.einsum('kl,klij->ij', T, x)).sum()

        # --- term2: 前段ノードの累積 T[q,s] * x[q,r,i,j] * x[i,j,k,l]
        # S_pre = T.cumsum(dim=1)  # [N, L]
        term2 = torch.einsum('qr,qrij,ijkl->', S_pre, x, x)

        # --- term3: 後段ノードの累積 T[m,p] * x[i,j,k,l] * x[k,l,m,n] with p >= n
        # S_post = torch.flip(torch.cumsum(torch.flip(T, dims=[1]), dim=1), dims=[1])  # [N, L]
        term3 = torch.einsum('mn,ijkl,klmn->', S_post, x, x)

        H_obj = term1 + term2 + term3

        


        # --- 順序制約項（同一マシン上のジョブは一方向のみ許容）
        x_T = x.permute(2, 3, 0, 1)  # 軸を反転して対称項を生成
        M1 = M.unsqueeze(2).unsqueeze(3)  # [N, L, 1, 1]
        M2 = M.unsqueeze(0).unsqueeze(1)  # [1, 1, N, L]
        same_machine_mask = (M1 == M2) & (M1 != -1) & (M2 != -1)

        order_violation =  (x + x_T - 1)**2 
        H_const1 = (order_violation * same_machine_mask).sum()


        return H_obj + self.coeff1 * H_const1 


    def build_fixed_tensor(self):
        """
        T: (N,L) 所要時間（形状だけ使う）
        M: (N,L) マシンインデックス（int）、未使用ジョブは -1
        戻り値:
        fixed: (N,L,N,L) のテンソル。  
            固定値0または1はその値、固定なしは None (dtype=object)
        """
        N, L = self.T.shape

        fixed = torch.full((N, L, N, L), -1)

        for i in range(N):
            for j in range(L):
                for k in range(N):
                    for l in range(L):
                        # --- どちらかが無効ジョブなら 0 に固定 ---
                        if self.T[i, j] == 0 or self.M[i, j] == -1 or \
                        self.T[k, l] == 0 or self.M[k, l] == -1:
                            fixed[i, j, k, l] = 0
                            continue


                        # --- 同じジョブ内の順序固定 ---
                        elif i == k:
                            if l == j + 1:
                                # # 次のジョブが無効なら 0、それ以外は 1
                                # if self.T[i, l] == 0 or self.M[i, l] == -1:
                                #     fixed[i, j, k, l] = 0
                                # else:
                                fixed[i, j, k, l] = 1
                            else:
                                fixed[i, j, k, l] = 0
                            continue
                        # ---異なるジョブでの制約
                        else:
                            # --- 異なるマシンなら 0 に固定 ---
                            if self.M[i, j] != self.M[k, l]:
                                fixed[i, j, k, l] = 0
                                continue
        return fixed
    
    def transitive_fixed_tensor(self): # 推移閉包の場合の固定テンソル作成関数
        """
        T: (N,L) 所要時間（形状だけ使う）
        M: (N,L) マシンインデックス（int）、未使用ジョブは -1
        戻り値:
        fixed: (N,L,N,L) のテンソル。  
            固定値0または1はその値、固定なしは None (dtype=object)
        """
        N, L = self.T.shape

        fixed = torch.full((N, L, N, L), -1)

        for i in range(N):
            for j in range(L):
                for k in range(N):
                    for l in range(L):
                        # --- どちらかが無効ジョブなら 0 に固定 ---
                        if self.T[i, j] == 0 or self.M[i, j] == -1 or \
                        self.T[k, l] == 0 or self.M[k, l] == -1:
                            fixed[i, j, k, l] = 0
                            continue

                        # --- 同じジョブ内の順序固定 ---
                        elif i == k:
                            if j < l:
                                fixed[i, j, k, l] = 1
                            else:
                                fixed[i, j, k, l] = 0
                            continue

        return fixed
    
    def remove_duplicates_from_sorted_jobs(self, sorted_jobs):
        seen = set()
        result = []
        # 前から走査して、初めて出会ったジョブを追加（結果は逆順になる）
        for job in (sorted_jobs):
            if job not in seen:
                seen.add(job)
                result.append(job)
        return result

    def draw_job_schedule_symmetric(self, x):
        """
        M: [I, J] LongTensor - 各ジョブが使用するマシンのインデックス (-1: ジョブなし)
        T: [I, J] FloatTensor - 各ジョブの所要時間 (0: ジョブなし)
        x: [N] BoolTensor or FloatTensor - ジョブ(i,j)が(k,l)より優先なら1
        """

        M, T = self.M, self.T
        x = self.x_reshape_dim1_to_dim4_symmetric(x).cpu()
        fixed = self.build_fixed_tensor()
        x = torch.where(fixed==-1, x, fixed)
        I, J = M.shape

        # --- ステップ 1: 有効なジョブだけ抽出 ---
        valid_jobs = [(i, j) for i in range(I) for j in range(J) if T[i, j] > 0 and M[i, j] >= 0]

        # --- DAG 構築 ---
        edges = defaultdict(list)
        in_degree = defaultdict(int)

        for i, j in valid_jobs:
            for k, l in valid_jobs:
                if x[i, j, k, l] > 0.5:
                    edges[(i, j)].append((k, l))
                    in_degree[(k, l)] += 1

        for job in valid_jobs:
            in_degree.setdefault(job, 0)  # 初期化

        # --- トポロジカルソート ---
        sorted_jobs = []
        queue = deque([node for node in valid_jobs if in_degree[node] == 0])

        while queue:
            node = queue.popleft()
            sorted_jobs.append(node)
            for neighbor in edges[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)


        if len(sorted_jobs) != len(valid_jobs):
            print("有向サイクルを検出。サイクルを除去してトポロジカル順を再構築します。")
            sorted_jobs = self.remove_duplicates_from_sorted_jobs(sorted_jobs=sorted_jobs)

            

        # --- 最早スケジューリング ---
        start_times = torch.zeros_like(T)
        machine_available = defaultdict(float)

        for i, j in sorted_jobs:
            m = M[i, j].item()
            t = T[i, j].item()

            if j > 0 and T[i, j - 1] > 0:
                prev_end = start_times[i, j - 1].item() + T[i, j - 1].item()
            else:
                prev_end = 0.0

            ready_time = machine_available[m]
            start_time = max(prev_end, ready_time)
            start_times[i, j] = start_time
            machine_available[m] = start_time + t

        # --- 色設定：マシンごとに色を固定 ---
        unique_machines = sorted(set(int(M[i, j].item()) for i, j in valid_jobs))
        num_machines = len(unique_machines)
        cmap = plt.cm.get_cmap('tab20', num_machines)
        machine_to_color = {m: cmap(idx) for idx, m in enumerate(unique_machines)}

        # --- ガントチャート描画 ---
        fig, ax = plt.subplots(figsize=(8, 0.3*I))
        yticks = []
        yticklabels = []

        height = 0.2
        for i in range(I):
            y = I - i - 1
            yticks.append(y)
            yticklabels.append(f'Task {i}')
            for j in range(J):
                if T[i, j] <= 0 or M[i, j] < 0:
                    continue
                s = start_times[i, j].item()
                d = T[i, j].item()
                m = M[i, j].item()
                color = machine_to_color[m]  # ← マシンごとの色を使用
                ax.barh(y, d, left=s, height=height, color=color, edgecolor='black')

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Time")
        ax.set_title("Gantt Chart (Topological Scheduling)")
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig

    
    def draw_job_schedule_asymmetric(self, x):
        """
        M: [I, J] LongTensor - 各ジョブが使用するマシンのインデックス (-1: ジョブなし)
        T: [I, J] FloatTensor - 各ジョブの所要時間 (0: ジョブなし)
        x: [N] BoolTensor or FloatTensor - ジョブ(i,j)が(k,l)より優先なら1
        """

        M, T = self.M, self.T
        x = self.x_reshape_dim1_to_dim4(x).cpu()
        I, J = M.shape

        # --- ステップ 1: 有効なジョブだけ抽出 ---
        valid_jobs = [(i, j) for i in range(I) for j in range(J) if T[i, j] > 0 and M[i, j] >= 0]

        # --- DAG 構築 ---
        edges = defaultdict(list)
        in_degree = defaultdict(int)

        for i, j in valid_jobs:
            for k, l in valid_jobs:
                if x[i, j, k, l] > 0.5:
                    edges[(i, j)].append((k, l))
                    in_degree[(k, l)] += 1

        for job in valid_jobs:
            in_degree.setdefault(job, 0)  # 初期化

        # --- トポロジカルソート ---
        sorted_jobs = []
        queue = deque([node for node in valid_jobs if in_degree[node] == 0])

        while queue:
            node = queue.popleft()
            sorted_jobs.append(node)
            for neighbor in edges[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_jobs) != len(valid_jobs):
            raise ValueError("x に矛盾があり、有向サイクルが存在します（スケジュール不可能）")

        # --- 最早スケジューリング ---
        start_times = torch.zeros_like(T)
        machine_available = defaultdict(float)

        for i, j in sorted_jobs:
            m = M[i, j].item()
            t = T[i, j].item()

            if j > 0 and T[i, j - 1] > 0:
                prev_end = start_times[i, j - 1].item() + T[i, j - 1].item()
            else:
                prev_end = 0.0

            ready_time = machine_available[m]
            start_time = max(prev_end, ready_time)
            start_times[i, j] = start_time
            machine_available[m] = start_time + t

        # --- 色設定：マシンごとに色を固定 ---
        unique_machines = sorted(set(int(M[i, j].item()) for i, j in valid_jobs))
        num_machines = len(unique_machines)
        cmap = plt.cm.get_cmap('tab20', num_machines)
        machine_to_color = {m: cmap(idx) for idx, m in enumerate(unique_machines)}

        # --- ガントチャート描画 ---
        fig, ax = plt.subplots(figsize=(8, 0.3*I))
        yticks = []
        yticklabels = []

        height = 0.2
        for i in range(I):
            y = I - i - 1
            yticks.append(y)
            yticklabels.append(f'Task {i}')
            for j in range(J):
                if T[i, j] <= 0 or M[i, j] < 0:
                    continue
                s = start_times[i, j].item()
                d = T[i, j].item()
                m = M[i, j].item()
                color = machine_to_color[m]  # ← マシンごとの色を使用
                ax.barh(y, d, left=s, height=height, color=color, edgecolor='black')

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Time")
        ax.set_title("Gantt Chart (Topological Scheduling)")
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig
    
    def compute_makespan(self, x):
        """
        x: [N] Flattened BoolTensor or FloatTensor (i,j)が(k,l)より前なら1
        return: float, DAG上の最長パス長（例: makespan）
        """
        

        M, T = self.M, self.T
        x = self.x_reshape_dim1_to_dim4(x).cpu()
        I, J = M.shape

        # --- 有効なノード抽出 ---
        valid_jobs = [(i, j) for i in range(I) for j in range(J) if T[i, j] > 0 and M[i, j] >= 0]

        # --- DAG 構築 ---
        edges = defaultdict(list)
        in_degree = defaultdict(int)

        for i, j in valid_jobs:
            for k, l in valid_jobs:
                if x[i, j, k, l] > 0.5:
                    edges[(i, j)].append((k, l))
                    in_degree[(k, l)] += 1

        for job in valid_jobs:
            in_degree.setdefault(job, 0)

        # --- トポロジカルソート ---
        sorted_jobs = []
        queue = deque([node for node in valid_jobs if in_degree[node] == 0])

        while queue:
            node = queue.popleft()
            sorted_jobs.append(node)
            for neighbor in edges[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_jobs) != len(valid_jobs):
            raise ValueError("x に矛盾があり、有向サイクルが存在します")

        # --- 最長パス計算 ---
        longest = defaultdict(float)

        for i, j in sorted_jobs:
            node_time = T[i, j].item()
            current_end = longest[(i, j)] + node_time
            for k, l in edges[(i, j)]:
                longest[(k, l)] = max(longest[(k, l)], current_end)

        # 最長の終了時刻が DAG の最長パス長
        max_length = max((longest[(i, j)] + T[i, j].item() for i, j in valid_jobs), default=0.0)
        return max_length

    def compute_makespan_symmetric(self, x):
        """
        x: [N] Flattened BoolTensor or FloatTensor (i,j)が(k,l)より前なら1
        return: float, DAG上の最長パス長（例: makespan）
        """

        M, T = self.M, self.T
        x = self.x_reshape_dim1_to_dim4_symmetric(x).cpu()
        I, J = M.shape

        # --- 有効なノード抽出 ---
        valid_jobs = [(i, j) for i in range(I) for j in range(J) if T[i, j] > 0 and M[i, j] >= 0]

        # --- DAG 構築 ---
        edges = defaultdict(list)
        in_degree = defaultdict(int)

        for i, j in valid_jobs:
            for k, l in valid_jobs:
                if x[i, j, k, l] > 0.5:
                    edges[(i, j)].append((k, l))
                    in_degree[(k, l)] += 1

        for job in valid_jobs:
            in_degree.setdefault(job, 0)

        # --- トポロジカルソート ---
        sorted_jobs = []
        queue = deque([node for node in valid_jobs if in_degree[node] == 0])

        while queue:
            node = queue.popleft()
            sorted_jobs.append(node)
            for neighbor in edges[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_jobs) != len(valid_jobs):
            raise ValueError("x に矛盾があり、有向サイクルが存在します")

        # --- 最長パス計算 ---
        longest = defaultdict(float)

        for i, j in sorted_jobs:
            node_time = T[i, j].item()
            current_end = longest[(i, j)] + node_time
            for k, l in edges[(i, j)]:
                longest[(k, l)] = max(longest[(k, l)], current_end)

        # 最長の終了時刻が DAG の最長パス長
        max_length = max((longest[(i, j)] + T[i, j].item() for i, j in valid_jobs), default=0.0)
        return max_length






# 数分割問題(NPP: number partitioning problem)
class NPP():
    def __init__(self, number_set:torch.Tensor):
        self.number_set = number_set

    def generator(self, x: torch.Tensor):
        return ((self.number_set * x).sum() - (self.number_set * (1-x)).sum())**2
    


    def draw_stack(self, x: torch.Tensor):
        """
        グループ分けされた要素の重みを、要素ごとに色を変えて積み上げ棒グラフで可視化。

        Args:
            x (torch.Tensor): shape = (n,), i番目の要素がグループ1なら1, グループ0なら0のバイナリ
            w (torch.Tensor): shape = (n,), i番目の要素の重みや数値
        """
        w = self.number_set
        assert x.shape == w.shape, "x と w の形が一致している必要があります"
        assert x.dtype == torch.bool or set(x.tolist()).issubset({0, 1}), "x はバイナリ (0 or 1) のみ"

        n = len(w)
        colors = cm.get_cmap('tab20', n)  # 最大20個の異なる色を取得（多ければ別のcolormapに切り替え）

        fig, ax = plt.subplots()
        bar_width = 0.5

        # グループ0
        bottom0 = 0
        for i in range(n):
            if x[i] == 0:
                ax.bar(0, w[i].item(), bottom=bottom0, width=bar_width, color=colors(torch.randint(n, (1,)).int().item()%n), edgecolor='none', label=f"Item {i}")
                bottom0 += w[i].item()

        # グループ1
        bottom1 = 0
        for i in range(n):
            if x[i] == 1:
                ax.bar(1, w[i].item(), bottom=bottom1, width=bar_width, color=colors(torch.randint(n, (1,)).int().item()%n), edgecolor='none', label=f"Item {i}")
                bottom1 += w[i].item()

        # 軸と凡例
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Group 0', 'Group 1'])
        ax.set_ylabel('Total Sum')
        ax.set_title('Group-wise Sum (Stacked Bar, Colored by Item)')

        plt.tight_layout()
        return fig




# グラフ等分割問題(GPP: graph partitioning problem)
class GPP():
    def __init__(self, graph, num_groups=2, coeff1=1, coeff2=1):
        self.graph = graph
        self.num_groups = num_groups
        self.num_nodes = self.graph[0]
        self.coeff1 = 2 * coeff1 * self.graph.max() 
        self.coeff2 = coeff2 * self.graph.max()

    def generator(self, x: torch.Tensor):
        """
        QUBO for graph multi-partitioning minimizing cut cost.
        
        graph: [N, N] symmetric adjacency matrix (weights)
        x: [N, K] binary matrix, x[i,k] = 1 if node i in group k
        """
        # Cut cost (avoid double-counting)
        same_group = torch.einsum('ik,jk->ij', x, x)
        delta = 1 - same_group
        cut_cost = 0.5 * (self.graph * delta).sum()

        # Constraint: each node in exactly one group
        Q1 = ((x.sum(dim=1) - 1) ** 2).sum()

        # Constraint: optional equal-sized groups
        N, K = x.shape
        Q2 = ((x.sum(dim=0) - N / K) ** 2).sum()
        # group_sizes = x.sum(dim=0)  # shape: [K]
        # Q2 = ((group_sizes.unsqueeze(1) - group_sizes.unsqueeze(0)) ** 2).sum()

        return cut_cost + 2 * self.coeff1 * Q1 + self.coeff2 * Q2
    
    def reduction_generator(self, x: torch.Tensor):
        """
        QUBO for graph multi-partitioning using reduced variables (K-1 per node).
        
        Parameters:
            graph: [N, N] symmetric weight matrix
            x: [N, K-1] binary/relaxed variables
            alpha: constraint penalty weight
            beta: optional group-balance penalty
        """
        N, K_minus_1 = x.shape
        K = K_minus_1 + 1

        # 補完された最後のグループ列
        x_last = 1 - x.sum(dim=1, keepdim=True)  # [N, 1]
        x_full = torch.cat([x, x_last], dim=1)   # [N, K]

        # --- cut cost ---
        same_group = torch.einsum('ik,jk->ij', x_full, x_full)
        delta = 1 - same_group
        cut_cost = torch.triu(self.graph * delta, diagonal=1).sum()

        # --- ペナルティ: 各ノードは1つのグループに ---
        Q1 = ((x_full.sum(dim=1) - 1) ** 2).sum()

        # --- ペナルティ: 各グループのサイズ均等（任意） ---
        Q2 = ((x_full.sum(dim=0) - N / K) ** 2).sum()

        return cut_cost + self.coeff1 * Q1 + self.coeff2 * Q2
    
    def bipartition_generator(self, x: torch.Tensor):
        """
        QUBO for graph bipartitioning using binary vector x ∈ {0,1}^N.
        
        graph: [N, N] symmetric adjacency matrix
        x: [N] binary assignment, x[i] = 0 or 1
        alpha: penalty weight for group size balancing
        """
        N = self.graph.shape[0]

        # Pairwise term: cut cost
        x_col = x.view(-1, 1)
        x_row = x.view(1, -1)
        xor_term = x_col + x_row - 2 * x_col * x_row
        cut_matrix = self.graph * xor_term
        cut_cost = torch.triu(cut_matrix, diagonal=1).sum()

        # Optional: equal-size constraint (sum x ≈ N/2)
        Q1 = (x.sum() - N / 2) ** 2

        return cut_cost + 2 * self.coeff1 * Q1
    
    def draw_graph_partition(self, x: torch.Tensor, figsize=(8, 6)):
        graph = self.graph
        """
        グラフとクラスタリング結果に基づいて、クラスタごとにノードを色分けし、
        クラスタごとにノードを整理したレイアウトで描画する。

        Args:
            graph (torch.Tensor): (N, N) 隣接行列（非ゼロはエッジ）
            x (torch.Tensor): (N, K) 各ノードのクラスタ割当（バイナリテンソル、行ごとにワンホット）
            figsize (tuple): 描画サイズ
        """
        assert graph.ndim == 2 and graph.shape[0] == graph.shape[1], "graph は NxN の正方テンソルである必要があります"

        if len(x.shape)==1:
            x = torch.cat([x.unsqueeze(1), (1 - x).unsqueeze(1)], dim=1)
            N, K = x.shape
        elif len(x.shape)==2:
            N, K = x.shape
        cluster_ids = torch.argmax(x, dim=1).tolist()

        # グラフ作成
        G = nx.Graph()
        for i in range(N):
            G.add_node(i, cluster=cluster_ids[i])
        edges = (graph > 0).nonzero(as_tuple=False)
        for i, j in edges:
            if i.item() < j.item():  # 無向グラフの片方だけ追加
                G.add_edge(i.item(), j.item(), weight=graph[i, j].item())

        # ノード位置をクラスタごとに整理
        cluster_positions = {}
        radius = 2.0
        for k in range(K):
            angle = 2 * np.pi * k / K
            center = np.array([np.cos(angle), np.sin(angle)]) * radius
            cluster_positions[k] = center

        pos = {}
        cluster_counts = [cluster_ids.count(k) for k in range(K)]
        for k in range(K):
            nodes_k = [i for i in range(N) if cluster_ids[i] == k]
            theta = np.linspace(0, 2 * np.pi, len(nodes_k), endpoint=False)
            for i, t in zip(nodes_k, theta):
                offset = cluster_positions[k] + 0.7 * np.array([np.cos(t), np.sin(t)])
                pos[i] = offset

        # ノード色
        colors = cm.get_cmap('tab20', K)
        node_colors = [colors(cluster_ids[i]) for i in range(N)]

        # 描画
        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax, node_size=100, edgecolors='black')
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, alpha=0.7, edge_color='gray')

        ax.set_title("Graph Partition Visualization (Cluster-Aligned Layout)")
        ax.axis('off')
        plt.tight_layout()
        return fig




# 最大カット問題(MCP: maximum cut problem)
class MCP():
    def __init__(self, graph, device='cuda'):
        self.device = device
        self.graph = graph.to(device)  # [N, N] symmetric adjacency matrix

    def generator(self, x: torch.Tensor):
        """
        QUBO for Max-Cut problem using binary vector x ∈ {0,1}^N.

        Parameters:
            graph: [N, N] symmetric adjacency matrix (edge weights)
            x: [N] binary tensor, x[i] = 0 or 1 (group assignment)

        Returns:
            Negative cut-cost (since typical optimizers minimize): -MaxCut
        """

        # A = (x @ (self.graph + self.graph.T))  # [N] vector, sum of weights for each node
        # B = (x @ self.graph.T)
        # cut_value = 0.5 * (A.sum() - (A * x).sum())
        cut_value =  ((x @ (self.graph)) * (1-x)).sum()

        
        return -cut_value  # maximize cut → minimize (–cut)
    
    def generator2(self, x: torch.Tensor):
        """
        QUBO for Max-Cut problem using binary vector x ∈ {0,1}^N.

        Parameters:
            graph: [N, N] symmetric adjacency matrix (edge weights)
            x: [N] binary tensor, x[i] = 0 or 1 (group assignment)

        Returns:
            Negative cut-cost (since typical optimizers minimize): -MaxCut
        """

        A = (x @ (self.graph))  # [N] vector, sum of weights for each node
        B = (x @ self.graph.T)
        cut_value = 0.5 * (A.sum() + B.sum()) - (A * x).sum()

        
        return -cut_value  # maximize cut → minimize (–cut)
    




# 最小頂点被覆問題(VCP: vertex covering problem)
class VCP():
    def __init__(self, graph, coeff1=1):
        self.graph = graph
        self.coeff1 = coeff1
    
    def generator(self, x:torch.Tensor):
        """
        QUBO for Minimum Vertex Cover.
        
        Parameters:
            graph: [N, N] symmetric binary tensor or weighted adjacency matrix
            x: [N] binary (or relaxed [0,1]) tensor
            alpha: penalty weight for edge constraint violation
        Returns:
            QUBO loss (to minimize)
        """
        x = x.view(-1, 1)     # [N, 1]
        penalty = (1 - x) @ (1 - x).t()   # [N, N] where (i,j)=1 if both not covered

        constraint = torch.triu(self.graph * penalty, diagonal=1).sum()
        objective = x.sum()
        
        return objective + 2 * self.coeff1 * constraint


    def draw_graph(self, x: torch.Tensor):
        G = self.graph
        """
        G: [N, N] torch.Tensor 無向グラフの隣接行列（対称行列）
        x: [N] torch.Tensor 各頂点の色ラベル（赤=1, グレー=0）
        
        Returns:
            fig: matplotlib.figure.Figure グラフ描画結果のFigureオブジェクト
        """

        # PyTorch → NumPy → NetworkX グラフ作成
        N = G.shape[0]
        G_nx = nx.Graph()
        for i in range(N):
            for j in range(i + 1, N):
                if G[i, j] != 0:
                    G_nx.add_edge(i, j, weight=G[i, j].item())

        # ノード色マップ作成
        color_map = ['red' if x[i] == 1 else 'lightgray' for i in range(N)]

        # ノード配置
        pos = nx.spring_layout(G_nx, seed=42)

        # 描画用Figureを作成し、描画
        fig, ax = plt.subplots(figsize=(6, 6))
        nx.draw(
            G_nx,
            pos,
            node_color=color_map,
            edgecolors = 'black',
            with_labels=False,
            node_size=10000/N,
            edge_color='gray',
            ax=ax
        )
        ax.set_title("Vertex Covering Set")
        return fig



# グラフ同型性問題(GIP: graph isomorphism problem)
class GIP():
    def __init__(self, graph1, graph2, coeff1=1, coeff2=1):
        self.G1 = graph1
        self.G2 = graph2
        self.coeff1 = coeff1
        self.coeff2 = coeff2


    def generator(self, x: torch.Tensor):
        """
        QUBO for graph isomorphism detection based on 4 logical conditions.
        
        Parameters:
            x: [N, N] binary assignment matrix (x[i,j] = 1 if node i in G1 maps to node j in G2)
            G1: [N, N] adjacency matrix of G1
            G2: [N, N] adjacency matrix of G2
            alpha: weight for permutation constraints
            beta: weight for consistency constraints (condition 3 and 4)
            
        Returns:
            Scalar QUBO loss
        """
        N = x.shape[0]
        
        # --- 条件1：各行に1つだけ1 ---
        row_sum = x.sum(dim=1)           # [N]
        row_penalty = ((row_sum - 1) ** 2).sum()

        # --- 条件2：各列に1つだけ1 ---
        col_sum = x.sum(dim=0)           # [N]
        col_penalty = ((col_sum - 1) ** 2).sum()

        # --- 条件3： (u,v) in E1 and (s,t) not in E2 → q[u,s]*q[v,t] = 0 ---
        mask_3 = (self.G1 == 1).unsqueeze(2).unsqueeze(3) & (self.G2 == 0).unsqueeze(0).unsqueeze(1)  # [N,N,N,N]
        q_outer = torch.einsum('us,vt->uvst', x, x)  # [N,N,N,N]
        constraint_3 = q_outer[mask_3].sum()

        # --- 条件4： (u,v) not in E1 and (s,t) in E2 → q[u,s]*q[v,t] = 0 ---
        mask_4 = (self.G1 == 0).unsqueeze(2).unsqueeze(3) & (self.G2 == 1).unsqueeze(0).unsqueeze(1)  # [N,N,N,N]
        constraint_4 = q_outer[mask_4].sum()
        
        # --- 総和 ---
        constraint_perm = row_penalty + col_penalty
        constraint_consistency = constraint_3 + constraint_4
        
        return 2 * self.coeff1 * constraint_perm + self.coeff2 * constraint_consistency
    





