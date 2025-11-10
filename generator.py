import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class TSP():
    def __init__(self, d, coeff1=1, coeff2=1, device='cuda'):
        self.d = d.to(device).float()
        self.d.fill_diagonal_(0.0)
        self.num_city = self.d.shape[0]
        density = (self.d != 0).sum() / (self.d.numel() - self.num_city)
        self.coeff1 = (coeff1 * (self.d.sum(dim=1) / ((self.num_city-1)*density)).max()).to(device)
        self.coeff2 = (coeff2 * (self.d.sum(dim=1) / ((self.num_city-1)*density)).max()).to(device)

    
    def generator(self, x):
        '''
        x[t][i]: torch.tensor (n-1, n-1)
        '''
        
        Const1 = self.coeff1 * ((1 - x.sum(dim=0))**2).sum()
        Const2 = self.coeff2 * ((1 - x.sum(dim=1))**2).sum()

        part1 = ((x[:self.num_city-2, :] @ self.d[:self.num_city-1, :self.num_city-1]) * x[1:self.num_city-1, :]).sum()
        part2 = torch.dot(self.d[:self.num_city-1, self.num_city-1], x[self.num_city-2, :])  # d[i, N] * x[T, i]
        part3 = torch.dot(self.d[:self.num_city-1, self.num_city-1], x[0, :])  # d[i, N] * x[0, i]
        Obj = part1 + part2 + part3

        H = Const1 + Const2 + Obj

        return H
    
    def to_device(self, device):
        self.d = self.d.to(device)
        self.coeff1 = self.coeff1.to(device)
        self.coeff2 = self.coeff2.to(device)
    

    def build_qubo(self, device='cuda'):
        org_device = self.d.device
        self.to_device(device)

        T = N = self.num_city - 1
        Q = torch.zeros((T, N, T, N), device=self.d.device)  # Q[t,i,u,j]

        t = torch.arange(T, device=self.d.device)
        i = torch.arange(N, device=self.d.device)

        # --- 制約項1: 列制約 Qとh ---
        t1, i1, i2 = torch.meshgrid(t, i, i, indexing='ij')
        mask_diff = i1 != i2
        Q[t1, i1, t1, i2] = self.coeff1 * mask_diff.float()
        h = - self.coeff1 * torch.ones((T, N), device=self.d.device)

        # --- 制約項2: 行制約 Qとh ---
        i1, t1, t2 = torch.meshgrid(i, t, t, indexing='ij')
        mask_diff2 = t1 != t2
        Q[t1, i1, t2, i1] += self.coeff2 * mask_diff2.float()
        h += - self.coeff2 #* torch.ones((T, N), device=self.d.device)

        # --- 目的関数部分 Qとh ---
        i1, j1, t1 = torch.meshgrid(i, i, torch.arange(T - 1, device=self.d.device), indexing='ij')
        Q[t1, i1, t1 + 1, j1] += self.d[i1, j1]
        d_end = self.d[:N, N]  # shape: (N,)
        h[T - 1, :] += d_end
        h[0, :] += d_end

        # --- 対称化 & 対角成分0 ---
        t_idx = torch.arange(T, device=self.d.device)
        i_idx = torch.arange(N, device=self.d.device)
        Q = Q + Q.permute(2, 3, 0, 1)
        Q[t_idx[:, None], i_idx[None, :], t_idx[:, None], i_idx[None, :]] = 0.0

        self.to_device(org_device)

        return torch.reshape(h, (N*N, )), torch.reshape(Q, ((N) * (N), (N) * (N)))
    
    def get_const(self):
        x = torch.zeros((self.num_city-1, self.num_city-1), device=self.d.device)
        const = self.generator(x)
        return const

    
    
    def get_route(self, spin_dim2):

        if spin_dim2.shape[0] != self.num_city:
            extended_spin = torch.zeros((self.num_city, self.num_city), dtype=spin_dim2.dtype)
            extended_spin[-1, -1] = 1
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
                key2 = self.route[(i + 1) % len(self.route)]  

                start = coordinate[key1]
                end = coordinate[key2]

                ax.arrow(start[0], start[1],
                        end[0] - start[0], end[1] - start[1],
                        head_width=0, head_length=0, fc='black', ec='black')

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
            print('NO DISPLAY DATA!')

    
    

class QAP():
    def __init__(self, f, d, coeff1, coeff2, device='cuda'):
        self.factory_num = f.shape[0]
        self.city_num = d.shape[0]
        self.d = d.to(device).float()
        self.f = f.to(device).float()
        self.spin_num = self.factory_num**2

        f_sum = self.f.sum(dim=1)  # [n]
        f_nonzero = ((self.f != 0).sum())  

        d_sum = self.d.sum(dim=1)  # [n]
        d_nonzero = ((self.d != 0).sum())

        density = (d_nonzero + f_nonzero) / (self.d.numel() + self.f.numel() - 2*self.city_num)

        # 外積で [n, n] 行列 M[i,k] を構築
        M = f_sum.unsqueeze(1) * d_sum.unsqueeze(0) / ((self.city_num - 1) * density) # [n,1] × [1,n] → [n,n]

        self.coeff1 = coeff1 * (M).max()
        self.coeff2 = coeff2 * (M).max()


    def generator(self, x):
        """
        x: shape (factory_num, factory_num)
        """

        sum_over_i = torch.sum(x, dim=0)  # shape: (factory_num,)
        Const1 = self.coeff1 * torch.sum((1 - sum_over_i) ** 2)

        sum_over_k = torch.sum(x, dim=1)  # shape: (factory_num,)
        Const2 = self.coeff2 * torch.sum((1 - sum_over_k) ** 2)

        A = torch.einsum('ij,ik->jk', self.f, x)
        B = torch.einsum('kl,jl->jk', self.d, x)
        Obj = (A*B).sum()
        H = Const1 + Const2 + Obj

        return H
    
    def to_device(self, device):
        self.d = self.d.to(device)
        self.f = self.f.to(device)
        self.coeff1 = self.coeff1.to(device)
        self.coeff2 = self.coeff2.to(device)

    def build_qubo(self, device='cuda'):
        N = self.factory_num  # assume square matrices of size N x N
        org_device = self.f.device
        self.to_device(device)

        Q = torch.zeros((N, N, N, N), device=device)  # Q[i,j,k,l]
        h = torch.zeros((N, N), device=device)

        i = torch.arange(N, device=device)
        j = torch.arange(N, device=device)

        # === 制約項1: 各施設には1つの工場を割り当てる ===
        # sum_i x[i,j] = 1  ⇨  ((1 - sum_i x[i,j])^2)
        i1, i2, j1 = torch.meshgrid(i, i, j, indexing='ij')
        mask1 = i1 != i2
        Q[i1, j1, i2, j1] += self.coeff1 * mask1.float()
        h[:, :] += -self.coeff1  # 各 j に -2 * coeff1、全体に同一値加算

        # === 制約項2: 各工場には1つの施設を割り当てる ===
        i1, j1, j2 = torch.meshgrid(i, j, j, indexing='ij')
        mask2 = j1 != j2
        Q[i1, j1, i1, j2] += self.coeff2 * mask2.float()
        h[:, :] += -self.coeff2

        # === 目的関数項 ===
        # Obj = sum_{i,j,k,l} f[i,k] * d[j,l] * x[i,j] * x[k,l]
        i1, j1, i2, j2 = torch.meshgrid(i, j, i, j, indexing='ij')
        Q += self.f[i1, i2] * self.d[j1, j2]  # broadcasted outer product

        # === 対称化 ===
        Q = Q + Q.permute(2, 3, 0, 1)  # Q[k,l,i,j] を加算して対称化

        self.to_device(org_device)
        return h.reshape(N * N), Q.reshape(N * N, N * N)

    def get_const(self):
        x = torch.zeros((self.city_num, self.city_num), device=self.d.device)
        const = self.generator(x)
        return const


    def draw_graph(self, x, city_prefix="C", factory_prefix="F"):
        x = x.T 
        """
        Parameters:
        - x: variable

        Returns:
        - fig: matplotlib.figure.Figure 
        """
        G = nx.Graph()
        num_cities, num_factories = x.shape

        city_nodes = [f"{city_prefix}{i}" for i in range(num_cities)]
        factory_nodes = [f"{factory_prefix}{j}" for j in range(num_factories)]

        G.add_nodes_from(city_nodes, bipartite=0)
        G.add_nodes_from(factory_nodes, bipartite=1)

        for i in range(num_cities):
            for j in range(num_factories):
                if x[i, j] == 1:
                    G.add_edge(city_nodes[i], factory_nodes[j])

        pos = {}
        pos.update((node, (i, 1)) for i, node in enumerate(city_nodes))
        pos.update((node, (i, 0)) for i, node in enumerate(factory_nodes))

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
        self.coeff1 = torch.tensor(coeff1, device=device)
        self.coeff2 = torch.tensor(coeff2, device=device)
        self.coeff3 = torch.tensor(coeff3, device=device)
        if num_color == None:
            self.num_color = int((self.E.sum(dim=0)).max()+1)
        else: self.num_color = num_color

    def generator(self, x: torch.Tensor, y: torch.Tensor):
        """ 
        Parameters:
            Eij: edge matrix, Eij[i, j] = 1 if edge(i,j)
            x: variable (node_num x color_num)
            y: variable (color_num,)
            coeff: penalty coeff

        Returns:
            H: QUBO function
        """

        Obj = torch.sum(y)

        Const1 = self.coeff1 * torch.sum(x * (self.E @ x))

        x_sum_per_color = torch.sum(x, dim=0)  # (color_num,)
        Const2 = 2 * self.coeff2 * torch.sum(x_sum_per_color * (1 - y))

        color_sum_per_node = torch.sum(x, dim=1)  # (node_num,)
        Const3 = 2 * self.coeff3 * torch.sum((1 - color_sum_per_node) ** 2)
        H = Obj + Const1 + Const2 + Const3

        return H
    

    def to_device(self, device):
        self.E = self.E.to(device)
        self.coeff1 = self.coeff1.to(device)
        self.coeff2 = self.coeff2.to(device)
        self.coeff3 = self.coeff3.to(device)

    def build_qubo(self, device='cuda'):
        N = self.num_node
        C = self.num_color
        M = N * C + C  # 総変数数（xとyを連結した1次元ベクトル）

        org_device = self.E.device
        self.to_device(device)

        # QUBO行列と線形項
        Q = torch.zeros((M, M), device=device)
        h = torch.zeros(M, device=device)
        offset = 0.0

        # --- 目的関数: ∑ y[j] ---
        y_start = N * C
        h[y_start:] += 1.0

        # --- Const1: 隣接ノードが同じ色を取らない ---
        edge_i, edge_j = torch.nonzero(torch.triu(self.E), as_tuple=True)  # i<j のみ
        for k in range(C):
            xi = edge_i * C + k
            xj = edge_j * C + k
            Q[xi, xj] += 2 * self.coeff1
            Q[xj, xi] += 2 * self.coeff1  # 対称性

        # --- Const2: x[i,j]=1 ⇒ y[j]=1 ---
        for j in range(C):
            y_idx = y_start + j
            x_idx = torch.arange(N, device=device) * C + j
            Q[x_idx, x_idx] += 2 * self.coeff2  # x[i,j]^2 の項
            Q[x_idx, y_idx] += -2 * self.coeff2
            Q[y_idx, x_idx] += -2 * self.coeff2  # 対称項

        # --- Const3: ノードごとに1色のみ（ワンホット） ---
        for i in range(N):
            x_idx = torch.arange(C, device=device) + i * C

            Q[x_idx[:, None], x_idx[None, :]] += 4.0 * self.coeff3  # ∑x[i,j1]x[i,j2]
            h[x_idx] += -6.0 * self.coeff3  # -2 * 2 * x[i,j]（線形項）

        # --- 対角成分を h に吸収 ---
        diag = torch.arange(M, device=device)
        h += Q[diag, diag]
        Q[diag, diag] = 0.0

        self.to_device(org_device)
        return h.detach(), Q.detach()
    
    def get_const(self):
        x = torch.zeros((self.num_node, self.num_color), device=self.E.device)
        y = torch.zeros((self.num_color,), device=self.E.device)
        const = self.generator(x, y)
        return const


    
    def draw_coloring(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): variable (num_nodes, num_colors)  
            E (torch.Tensor): edge matrix (num_nodes, num_nodes) 

        Returns:
            fig: matplotlib figure
        """
        num_colors = int(torch.where(x.sum(dim=0)!=0, True, False).sum())
        x = x.detach().cpu().numpy()
        E = torch.where(self.E==1, True, False).cpu().numpy()
        num_nodes, _ = x.shape
        node_colors = np.argmax(x, axis=1)  

        G = nx.Graph()

        for i in range(num_nodes):
            G.add_node(i, color=node_colors[i])

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if E[i][j]:
                    G.add_edge(i, j)

        pos = nx.spring_layout(G, seed=42)

        cmap = plt.get_cmap('tab10' if num_colors <= 10 else 'prism')

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
        self.coeff1 = torch.tensor(coeff1, device=device) 


    def generator(self, x: torch.Tensor):
        """
        Parameters:
            Eij: edge matrix (node_num x node_num), Eij[i, j] = 1 if edge(i,j) (Eij[i,j]=Eij[j,i])
            x: variable
            coeff: penalty coefficient

        Returns:
            H: QUBO function
        """
        Obj = -torch.sum(x)
        Const1 = self.coeff1 * ((x @ self.E) * x).sum()
        H = Obj + Const1 

        return H
    
    def to_device(self, device):
        self.E = self.E.to(device)
        self.coeff1 = self.coeff1.to(device)

    
    def build_qubo(self, device='cuda'):
        """
        QUBO形式:  0.5 * x^T Q x + h^T x + offset
        """
        N = self.E.shape[0]
        org_device = self.E.device
        self.to_device(device)

        Q = torch.zeros((N, N), device=device)
        h = torch.zeros(N, device=device)

        # --- Const1: xᵀ E x ---
        # Q_ij = coeff1 * E_ij
        Q += 2 * self.coeff1 * self.E

        # --- 目的関数: -sum(x) ⇒ 線形項 h[i] += -1
        h += -1.0

        # --- 対角成分を h に吸収 ---
        diag = torch.arange(N, device=device)
        h += Q[diag, diag]
        Q[diag, diag] = 0.0

        self.to_device(org_device)

        return h.detach(), Q.detach()
    
    def get_const(self):
        x = torch.zeros((self.num_node, ), device=self.E.device)
        const = self.generator(x)
        return const
    
    def draw_independent_set(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (num_nodes, num_colors) 
            E (torch.Tensor): (num_nodes, num_nodes) 

        Returns:
            fig: matplotlib figure
        """
        x = x.detach().cpu().numpy()
        E = torch.where(self.E==1, True, False).cpu().numpy()
        num_nodes = x.shape[0]
        
        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(i, color=x[i])

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if E[i][j]:
                    G.add_edge(i, j)

        pos = nx.circular_layout(G)
        cmap = plt.get_cmap('tab10')

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



# MCP: maximum cut problem
class MCP():
    def __init__(self, graph, device='cuda'):
        self.device = device
        self.graph = graph.to(device)  # [N, N] symmetric adjacency matrix
        self.num_node = self.graph.shape[0]

    def generator(self, x: torch.Tensor):
        """
        QUBO for Max-Cut problem using binary vector x ∈ {0,1}^N.

        Parameters:
            graph: [N, N] symmetric adjacency matrix (edge weights)
            x: [N] binary tensor, x[i] = 0 or 1 (group assignment)

        Returns:
            Negative cut-cost (since typical optimizers minimize): -MaxCut
        """

        return -((x @ (self.graph)) * (1-x)).sum()  # maximize cut → minimize (–cut)
    
    def to_device(self, device):
        self.graph = self.graph.to(device)
    
    def build_qubo(self, device='cuda'):
        """
        非対称グラフ対応のMax-Cut QUBO行列作成

        Returns:
            h: 一次係数ベクトル (N,)
            Q: 二次係数行列 (N, N), 対角成分は0でhに吸収済み
        """
        W = self.graph  # 非対称隣接行列 (N x N)
        N = W.shape[0]
        org_device = W.device
        self.to_device(device)

        Q = 2 * W.clone()  # x_i x_j の係数は W_ij
        h = -torch.sum(W, dim=1)  # x_i の係数は -sum_j W_ij

        # 対角成分を h に吸収
        diag = torch.arange(N, device=device)
        h += Q[diag, diag]
        Q[diag, diag] = 0.0

        self.to_device(org_device)

        return h.detach(), Q.detach()
    
    def get_const(self):
        x = torch.zeros((self.num_node, ), device=self.graph.device)
        const = self.generator(x)
        return const
    
    


# MCP: maximum cut problem
class BQP():
    def __init__(self, graph, device='cuda'):
        self.device = device
        self.graph = graph.to(device)  # [N, N] symmetric adjacency matrix
        self.num_node = self.graph.shape[0]

    def generator(self, x: torch.Tensor):
        """
        QUBO for Max-Cut problem using binary vector x ∈ {0,1}^N.

        Parameters:
            graph: [N, N] symmetric adjacency matrix (edge weights)
            x: [N] binary tensor, x[i] = 0 or 1 (group assignment)

        Returns:
            Negative cut-cost (since typical optimizers minimize): -MaxCut
        """

        return ((x @ (self.graph)) * (x)).sum()  