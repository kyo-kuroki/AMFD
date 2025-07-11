import numpy as np
import math
import re



class QAP:
    def __init__(self):
        pass

    def read_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # =====================
        # 1. 都市数を取得（最初の非空行から）
        # =====================
        for line in lines:
            if line.strip():  # 空行でない
                numbers = [int(s) for s in re.findall(r'-?\d+', line)]
                if not numbers:
                    continue
                self.city_num = numbers[0]
                self.factory_num = self.city_num
                break

        n = self.city_num
        line_idx = lines.index(line) + 1  # 次の行からマトリクスを読む

        # 空行をスキップ
        while line_idx < len(lines) and not lines[line_idx].strip():
            line_idx += 1

        # =====================
        # 2. 流量行列Fを読み取り
        # =====================
        F_data = []
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if not line:  # 空行が来たら終了
                line_idx += 1
                break
            F_data.extend([int(x) for x in line.split()])
            line_idx += 1

        if len(F_data) != n * n:
            raise ValueError(f"流量行列のサイズが不正です（{len(F_data)} 要素, 期待値: {n*n}）")
        self.Fij = np.array(F_data, dtype=int).reshape((n, n))

        # =====================
        # 3. 距離行列Dを読み取り
        # =====================
        D_data = []
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if not line:
                line_idx += 1
                continue
            D_data.extend([int(x) for x in line.split()])
            line_idx += 1

        if len(D_data) != n * n:
            raise ValueError(f"距離行列のサイズが不正です（{len(D_data)} 要素, 期待値: {n*n}）")
        self.Dij = np.array(D_data, dtype=int).reshape((n, n))
        return self.Fij, self.Dij
    



class TSP:
    def __init__(self):
        pass

        
    def read_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        header = {}
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line in ('NODE_COORD_SECTION', 'EDGE_WEIGHT_SECTION'):
                break
            if line == 'DISPLAY_DATA_SECTION':
                # DISPLAY_DATA_SECTION は無視して EOF までスキップ
                while i < len(lines) and 'EOF' not in lines[i]:
                    i += 1
                return None  # または continue, raise, pass など適宜処理
            if ':' in line:
                key, value = line.split(':', 1)
            elif line:
                key, value = line.split(None, 1)
            else:
                i += 1
                continue
            header[key.strip()] = value.strip()
            i += 1

        edge_type = header.get("EDGE_WEIGHT_TYPE", "EUC_2D")
        dimension = int(header["DIMENSION"])

        if edge_type == "EUC_2D" or edge_type == "GEO" or edge_type == "ATT":
            coords = []
            i += 1
            while i < len(lines):
                if 'EOF' in lines[i] or 'DISPLAY_DATA_SECTION' in lines[i]:
                    break
                parts = lines[i].strip().split()
                if len(parts) < 3:
                    i += 1
                    continue
                _, x, y = parts
                coords.append((float(x), float(y)))
                i += 1
            coords = np.array(coords)

            if edge_type == "EUC_2D":
                return self.euclidean_distance_matrix(coords)
            elif edge_type == "GEO":
                return self.geo_distance_matrix(coords)
            elif edge_type == "ATT":
                return self.att_distance_matrix(coords)

        elif edge_type == "EXPLICIT":
            edge_format = header.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")
            i += 1
            data = []
            while i < len(lines):
                if 'EOF' in lines[i] or 'DISPLAY_DATA_SECTION' in lines[i]:
                    break
                data.extend([int(x) for x in lines[i].strip().split()])
                i += 1
            return self.explicit_distance_matrix(data, dimension, edge_format)

        else:
            raise NotImplementedError(f"EDGE_WEIGHT_TYPE {edge_type} is not supported.")


    def euclidean_distance_matrix(self, coords):
        n = len(coords)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx, dy = coords[i] - coords[j]
                D[i][j] = int(round(math.hypot(dx, dy)))
        return D
    
    def att_distance_matrix(self, coords):
        n = len(coords)
        D = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                dx, dy = coords[i] - coords[j]
                rij = math.sqrt((dx*dx + dy*dy) / 10.0)
                tij = int(round(rij))
                if tij < rij:
                    dij = tij + 1
                else:
                    dij = tij
                D[i][j] = dij
        return D

    def geo_distance_matrix(self, coords):
        def deg2rad(deg):
            deg_int = int(deg)
            min_part = deg - deg_int
            return math.pi * (deg_int + 5.0 * min_part / 3.0) / 180.0

        n = len(coords)
        D = np.zeros((n, n))
        RRR = 6378.388
        for i in range(n):
            lat_i, lon_i = map(deg2rad, coords[i])
            for j in range(n):
                lat_j, lon_j = map(deg2rad, coords[j])
                q1 = math.cos(lon_i - lon_j)
                q2 = math.cos(lat_i - lat_j)
                q3 = math.cos(lat_i + lat_j)
                D[i][j] = int(RRR * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1)
        return D

    def explicit_distance_matrix(self, data, dim, fmt):
        D = np.zeros((dim, dim), dtype=int)
        if fmt == 'FULL_MATRIX':
            k = 0
            for i in range(dim):
                for j in range(dim):
                    D[i][j] = data[k]
                    k += 1
        elif fmt == 'UPPER_ROW':
            k = 0
            for i in range(dim):
                for j in range(i+1, dim):
                    D[i][j] = D[j][i] = data[k]
                    k += 1
        elif fmt == 'LOWER_DIAG_ROW':
            k = 0
            for i in range(dim):
                for j in range(i+1):
                    D[i][j] = D[j][i] = data[k]
                    k += 1
        else:
            raise NotImplementedError(f"EDGE_WEIGHT_FORMAT {fmt} not supported")
        return D

 
        
# Max Cut Problem
class MCP:
    def __init__(self):
        pass

    def read_file(self, file_path):
        with open(file_path) as f:
            x = [int(s) for s in f.readline().split()]
            self.node_num = x[0]
            self.edge_num = x[1]
            self.spin_num = self.node_num
            self.Eij = np.zeros((self.node_num, self.node_num))

            for line in f:
                x = [int(s) for s in line.split()]
                self.Eij[x[0]-1][x[1]-1] = x[2]
                self.Eij[x[1]-1][x[0]-1] = x[2]
        return self.Eij


# Max Independent Set Problem (MISP) (equivalent to Max Clique Problem)
class MISP:
    def __init__(self):
        pass

    def read_file(self, file_path):
        with open(file_path) as f:

            for line in f:
                try:
                    l = line.split()
                except Exception:
                    continue
                if l[0] == 'p':
                    self.edge_num = int(l[-1])
                    self.node_num = int(l[-2])
                    self.Eij = np.zeros((self.node_num, self.node_num))
                elif l[0] == 'e':
                    i = int(l[1])-1
                    j = int(l[2])-1
                    self.Eij[i][j] = 1
                    self.Eij[j][i] = 1
        complemented_adj_matrix = 1 - self.Eij
        np.fill_diagonal(complemented_adj_matrix, 0)  
        return complemented_adj_matrix # complement of adjacency matrix
 
# Graph Coloring Problem (GCP) 
class GCP:
    def __init__(self):
        pass

    def read_file(self, file_path):
        with open(file_path) as f:

            for line in f:
                try:
                    l = line.split()
                except Exception:
                    continue
                if l[0] == 'p':
                    self.edge_num = int(l[-1])
                    self.node_num = int(l[-2])
                    self.Eij = np.zeros((self.node_num, self.node_num))
                elif l[0] == 'e':
                    i = int(l[1])-1
                    j = int(l[2])-1
                    self.Eij[i][j] = 1
                    self.Eij[j][i] = 1

        return self.Eij
