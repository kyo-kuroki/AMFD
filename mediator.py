import torch
import inspect
import itertools
from collections import defaultdict
import sympy as sp
import torch

def get_qubo(f, arg_shapes, device='cuda:0'):
    """
    f: 任意の PyTorch 関数（スカラ出力）
    arg_shapes: dict[str, tuple] — 引数名とその形状
    
    Returns:
      result: {
          'const': f(0),
          'h': 一次項ベクトル,
          'Q': 二次項行列 (1/2 * Hesse)
      },
      meta: {
          'arg_names': 引数名のリスト,
          'split_sizes': 各引数の要素数,
          'arg_shapes': 各引数の形状,
          'index_map': flattenされた添え字 → (引数名, 元のインデックス)
      }
    """
    device = device if torch.cuda.is_available() else 'cpu'
    if isinstance(arg_shapes, dict):
        arg_names = list(inspect.signature(f).parameters)
    elif isinstance(arg_shapes, list):
        arg_names = list(range(len(arg_shapes)))
    else:
        raise ValueError('arg_shapesは辞書かリストで入力してください')
    vars = []
    split_sizes = []
    index_map = {}

    flat_index = 0
    for name in arg_names:
        shape = arg_shapes[name]
        t = torch.zeros(shape, requires_grad=True, device=device)
        vars.append(t)
        split_sizes.append(t.numel())

        # 元インデックスの列挙（多次元対応）
        for idx in itertools.product(*[range(s) for s in shape]):
            index_map[flat_index] = (name, idx)
            flat_index += 1

    # 出力計算
    output = f(*vars)  # スカラー出力を仮定

    # 定数項
    const = output.item()

    # 勾配（各変数に対して）
    grads = torch.autograd.grad(output, vars, create_graph=True)
    h = torch.cat([g.reshape(-1) for g in grads])

    # 全変数を1ベクトルとして扱う
    v = torch.cat([(v+0.5).reshape(-1) for v in vars]) # 0.5まわりでテイラー展開

    # flat_f: 1次元ベクトルを元に戻して関数を評価
    def flat_f(v_flat):
        split = torch.split(v_flat, split_sizes)
        reshaped = [s.view(arg_shapes[name]) for s, name in zip(split, arg_names)]
        return f(*reshaped)

    # ヘッセ行列（要素数 n の変数に対して n×n）
    Q = torch.func.hessian(flat_f)(v)
    Q = Q.clone()
    h += torch.diagonal(0.5*Q)
    # 対角項をゼロにする
    Q.fill_diagonal_(0)

    return {
        'const': const,
        'h': h.detach(),
        'Q': Q.detach()
    }, {
        'arg_names': arg_names,
        'split_sizes': split_sizes,
        'arg_shapes': arg_shapes,
        'index_map': index_map
    }




def get_qubo_save_memory(f, arg_shapes, device='cuda:0'):
    device = device if torch.cuda.is_available() else 'cpu'

    # 引数名の取得
    if isinstance(arg_shapes, dict):
        arg_names = list(inspect.signature(f).parameters)
    elif isinstance(arg_shapes, list):
        arg_names = list(range(len(arg_shapes)))
    else:
        raise ValueError('arg_shapesはdictかlistである必要があります')

    # 引数をflattenしてまとめる
    vars = []
    split_sizes = []
    index_map = {}
    flat_index = 0
    for name in arg_names:
        shape = arg_shapes[name]
        t = torch.zeros(shape, requires_grad=True, device=device)
        vars.append(t)
        split_sizes.append(t.numel())
        for idx in itertools.product(*[range(s) for s in shape]):
            index_map[flat_index] = (name, idx)
            flat_index += 1

    
    # 定数項と勾配
    output = f(*vars)
    const = output.item()
    grads = torch.autograd.grad(output, vars, create_graph=True)
    h = torch.cat([g.reshape(-1) for g in grads])

    try:
        Q = torch.func.jacrev(torch.func.grad(f))(*vars)
        n = int(Q.numel() ** 0.5)
        Q = torch.reshape(Q, (n, n)).detach().clone()  # [n, n] の形に変形
    except Exception as e:
        print(f"making hessian serially due to : {e}")
        def rowwise_hesse(h):
            Q_rows = []
            for i in range(h.numel()):
                grad_i = torch.autograd.grad(h[i], vars, retain_graph=True)
                Q_rows.append(torch.cat([g.reshape(-1) for g in grad_i]).detach())
            return torch.stack(Q_rows).detach() 
        Q = rowwise_hesse(h).requires_grad_(False)


    # 一次項の補正（Q の対角成分を使う）
    h = h + torch.diagonal(0.5 * Q)
    Q.fill_diagonal_(0)

    return {
        'const': const,
        'h': h.detach(),
        'Q': Q.detach()
    }, {
        'arg_names': arg_names,
        'split_sizes': split_sizes,
        'arg_shapes': arg_shapes,
        'index_map': index_map
    }







def restore_variables(flat_tensor: torch.Tensor, index_map: dict):
    """
    フラット化されたテンソルを、index_map を使って元の各変数のテンソルに戻す。
    
    Args:
        flat_tensor (torch.Tensor): フラット化された1次元テンソル（長さn）
        index_map (dict): {flat_index: (var_name, index_tuple)} の辞書
    
    Returns:
        dict: {var_name: tensor of original shape}
    """
    # 1. 各変数ごとにshape情報を集める
    var_shapes = defaultdict(lambda: set())
    for _, (var_name, idx_tuple) in index_map.items():
        var_shapes[var_name].add(idx_tuple)

    # 2. shape情報をもとにテンソルを初期化
    var_tensors = {}
    for var, indices in var_shapes.items():
        shape = tuple(max(id[i] for id in indices) + 1 for i in range(len(next(iter(indices)))))
        var_tensors[var] = torch.zeros(shape, dtype=flat_tensor.dtype)

    # 3. 値を復元
    for flat_i, (var_name, idx) in index_map.items():
        var_tensors[var_name][idx] = flat_tensor[flat_i]

    return var_tensors





def get_qubo_sympy(expr, as_torch=False, dtype=torch.float32, device='cpu'):
    """
    expr: sympy のスカラー式（目的関数）
    symbols: expr に含まれる sympy.Symbol のリスト（順序を固定するため）
    as_torch: True なら torch.Tensor で返す

    Returns:
      result: {
          'const': float,
          'h': 一次項ベクトル (sympy.Matrix or torch.Tensor),
          'Q': 二次項行列 (sympy.Matrix or torch.Tensor)
      },
      meta: {
          'symbols': symbols,
          'index_map': {idx: symbol}  # フラットインデックス→シンボル
      }
    """
    symbols = list(expr.free_symbols)
    symbols.sort(key=lambda s: s.name) 
    subs0 = {s: 0 for s in symbols}
    const = expr.subs(subs0)

    h_linear = [sp.diff(expr, s).subs(subs0) for s in symbols]
    h_diag = [0.5 * sp.diff(expr, s, s).subs(subs0) for s in symbols]
    h = sp.Matrix([hl + hd for hl, hd in zip(h_linear, h_diag)])

    n = len(symbols)
    Q = sp.Matrix(n, n, lambda i, j: 0)
    for i in range(n):
        for j in range(i + 1, n):
            val = 0.5 * sp.diff(expr, symbols[i], symbols[j]).subs(subs0)
            Q[i, j] = val
            Q[j, i] = val

    if as_torch:
        h_tensor = torch.tensor([float(e) for e in h], dtype=dtype, device=device)
        Q_tensor = torch.tensor([[float(Q[i, j]) for j in range(n)] for i in range(n)],
                                dtype=dtype, device=device)
        const_val = float(const)
        return {'const': const_val, 'h': h_tensor, 'Q': Q_tensor}, {
            'symbols': symbols,
            'index_map': {i: s for i, s in enumerate(symbols)}
        }
    else:
        return {'const': const, 'h': h, 'Q': Q}, {
            'symbols': symbols,
            'index_map': {i: s for i, s in enumerate(symbols)}
        }


