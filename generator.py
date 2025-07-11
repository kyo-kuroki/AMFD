import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class TSP():
    def __init__(self, d, coeff1=1, coeff2=1, device='cuda'):
        self.d = d.to(device)
        self.coeff1 = (coeff1 * (self.d.mean(dim=0)).max()).to(device)
        self.coeff2 = (coeff2 * (self.d.mean(dim=0)).max()).to(device)
        self.num_city = self.d.shape[0]

    
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
        self.d = d.to(device)
        self.f = f.to(device)
        self.spin_num = self.factory_num**2
        Mik = torch.zeros((self.city_num, self.city_num), device=self.d.device)
        d = self.d.sum(dim=0)
        f = self.f.sum(dim=0)
        for i in range(self.city_num):
            for k in range(self.city_num):
                Mik[i][k] = (1/(self.city_num-1)) * d[i] * f[k]
        self.coeff1 = coeff1 * (Mik).max()
        self.coeff2 = coeff2 * (Mik).max()


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
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.coeff3 = coeff3
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
        self.coeff1 = coeff1 


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
    
    


