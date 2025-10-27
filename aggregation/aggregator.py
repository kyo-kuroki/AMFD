import pandas as pd
import numpy as np
import re
import os
import sys


# --- カテゴリ分類 ---
def categorize_process(p):
    if p == "MILP":
        return "MILP"
    elif p == "MIQP":
        return "MIQP"
    elif p == "QUBO":
        return "QUBO"
    else:
        return "Other"



# --- time_group ルール ---
def round_time_group(time):
    return round(time, 2) if int(time) == 0 else round(time, 1)



# --- フィルタ関数（Other だけ無視） ---
def filter_group(group):
    category = group["category"].iloc[0]
    
    # Other はフィルタしない
    if category == "Other":
        return group.copy()

    max_time_idx = group['time'].idxmax()
    max_row = group.loc[[max_time_idx]]
    other_rows = group.drop(index=max_time_idx)

    def pick_min(row_group):
        min_value = row_group['value'].min()
        candidates = row_group[row_group['value'] == min_value]
        return candidates.loc[[candidates['time'].idxmin()]]

    filtered = (
        other_rows.groupby("time_group", group_keys=False)
                  .apply(pick_min)
    )

    return pd.concat([filtered, max_row], ignore_index=True)




def select_n_plus_1_pairs(times, values, n):
    if len(times) == 0:
        return [(None, None)] * (n + 1)

    pairs = sorted(zip(times, values))
    if len(pairs) <= n + 1:
        return pairs + [(None, None)] * (n + 1 - len(pairs))

    max_pair = max(pairs, key=lambda x: x[0])
    pairs.remove(max_pair)

    times_only = np.array([t for t, _ in pairs])
    values_only = np.array([v for _, v in pairs])

    while len(times_only) > n:
        intervals = np.diff(times_only)
        min_idx = np.argmin(intervals)
        drop_idx = min_idx if (min_idx == 0 or (min_idx < len(intervals) - 1 and intervals[min_idx + 1] < intervals[min_idx])) else min_idx + 1

        times_only = np.delete(times_only, drop_idx)
        values_only = np.delete(values_only, drop_idx)

    reduced = list(zip(times_only.tolist(), values_only.tolist())) + [max_pair]
    return reduced

def natural_key(s):
    """例: G2, G10 を数値順で比較できるよう分解"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def sort_instances_naturally_with_groupby(df):
    grouped = list(df.groupby('instance'))
    grouped.sort(key=lambda x: natural_key(x[0]))  # x[0] は instance名
    sorted_df = pd.concat([group for _, group in grouped], ignore_index=True)
    return sorted_df

def reshape_full_table(df):
    methods = ['MILP', 'MIQP', 'QUBO']
    result = []

    for instance, group in df.groupby('instance'):
        n = group['Other_time'].notna().sum()
        instance_rows = []

        selected_data = {}

        # 各 method の time & value
        for method in methods:
            time_col = f'{method}_time'
            value_col = f'{method}_value'

            times = group[time_col].dropna().tolist()
            values = group[value_col].dropna().tolist()

            selected_pairs = select_n_plus_1_pairs(times, values, n)
            selected_data[f'{method}_time'] = [p[0] for p in selected_pairs]
            selected_data[f'{method}_value'] = [p[1] for p in selected_pairs]

        # Other_time & value
        other_times = group['Other_time'].dropna().tolist()
        other_values = group['Other_value'].dropna().tolist()
        other_selected = select_n_plus_1_pairs(other_times, other_values, n)
        selected_data['Other_time'] = [p[0] for p in other_selected]
        selected_data['Other_value'] = [p[1] for p in other_selected]

        # best_known (全て同じ値を n+1 行繰り返す)
        best_known = group['best_known'].iloc[0] if 'best_known' in group else None
        selected_data['best_known'] = [best_known] * (n + 1)

        # instance 名も追加
        selected_data['instance'] = [instance] * (n + 1)

        # DataFrame にして格納
        result.append(pd.DataFrame(selected_data))

        reshaped_df = pd.concat(result, ignore_index=True)

        # 列の順番を変更: 'instance' を先頭, 'best_known' を2番目に
        cols = reshaped_df.columns.tolist()
        cols.remove('instance')
        cols.remove('best_known')
        reshaped_df = reshaped_df[['instance', 'best_known'] + cols]


    reshaped_df = sort_instances_naturally_with_groupby(reshaped_df)

    reshaped_df = reshaped_df.rename(columns={
    'Other_time': 'AMFD_time',
    'Other_value': 'AMFD_value'
    })
    reshaped_df = reshaped_df.dropna(
    subset=['MILP_time', 'MIQP_time', 'QUBO_time', 'AMFD_time'],
    how='all'
    )


    return reshaped_df



def get_closest_rows(df):
    result_rows = []

    for instance, group in df.groupby('instance'):
        best_known = group['best_known'].iloc[0]
        row_dict = {'instance': instance, 'best_known': best_known}

        for method in ['MILP', 'MIQP', 'QUBO', 'Other']:
            val_col = f'{method}_value'
            time_col = f'{method}_time'

            if val_col in group.columns:
                # 有効な行だけ抽出（NaN除外）
                valid = group[['instance', val_col, time_col]].dropna(subset=[val_col, time_col])

                if not valid.empty:
                    # best_knownとの差の絶対値が最小の行を抽出
                    valid['abs_diff'] = (valid[val_col] - best_known).abs()

                    # 最小差分を持つ行の中で、time が最小のものを選ぶ
                    min_diff = valid['abs_diff'].min()
                    closest = valid[valid['abs_diff'] == min_diff]
                    closest_row = closest.loc[closest[time_col].idxmin()]

                    row_dict[f'{method}_value'] = closest_row[val_col]
                    row_dict[f'{method}_time'] = closest_row[time_col]
                else:
                    row_dict[f'{method}_value'] = None
                    row_dict[f'{method}_time'] = None

        result_rows.append(row_dict)

    reshaped_df = pd.DataFrame(result_rows)
    reshaped_df = sort_instances_naturally_with_groupby(reshaped_df)

    reshaped_df = reshaped_df.rename(columns={
    'Other_time': 'AMFD_time',
    'Other_value': 'AMFD_value'
    })
    reshaped_df = reshaped_df.dropna(
    subset=['MILP_time', 'MIQP_time', 'QUBO_time', 'AMFD_time'],
    how='all'
    )
    return reshaped_df






if __name__ == '__main__':

    # --- ファイル名 ---
    p_names = ['mcp', 'misp', 'tsp', 'qap', 'gcp']
    results = []
    for p_name in p_names:
        dir = os.path.dirname(__file__)
        file1 = os.path.dirname(dir) + f'/gurobi/results/{p_name}_results.csv'
        file2 = dir + f"/{p_name}_results.csv"

        # --- ファイル読み込み ---
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # best known solution の取得（file2 のみ持っている想定）
        best_known_map = df2.set_index('instance')['best known solution'].to_dict()

        # --- 結合＆前処理 ---
        df = pd.concat([df1, df2], ignore_index=True)
        df = df[df["constraint satisfaction"] == True]
        df["time"] = df["time"].astype(float)
        df["value"] = df["value"].astype(float)

        df["category"] = df["process"].apply(categorize_process)

        df["time_group"] = df["time"].apply(round_time_group)



        # --- フィルタ実行 ---
        filtered_df = (
            df.groupby(['category', 'instance', 'process'], group_keys=False)
            .apply(filter_group)
        )

        filtered_df = filtered_df.drop(columns=["time_group"])

        # --- カテゴリ別に整形 ---
        grouped_by_cat = filtered_df.groupby("category")
        all_instances = sorted(filtered_df["instance"].unique())
        instance_max_lens = {inst: 0 for inst in all_instances}
        category_tables = {}

        for cat_name, cat_group in grouped_by_cat:
            inst_tables = []
            for inst_name in all_instances:
                inst_group = cat_group[cat_group["instance"] == inst_name].sort_values("time")
                inst_len = len(inst_group)
                instance_max_lens[inst_name] = max(instance_max_lens[inst_name], inst_len)
                inst_tables.append(inst_group[["instance", "time", "value"]])

            cat_df = pd.concat(inst_tables, axis=0, ignore_index=True)
            category_tables[cat_name] = cat_df

        # --- instance列の作成 ---
        instance_column = []
        best_known_column = []

        for inst in all_instances:
            for _ in range(instance_max_lens[inst]):
                instance_column.append(inst)
                best_known_column.append(best_known_map.get(inst, None))

        result = pd.DataFrame({
            'instance': instance_column,
            'best_known': best_known_column
        })
        # --- 各カテゴリの列追加 ---
        category_order = ["Other", "MILP", "MIQP", "QUBO"]  # 明示的な順序指定
        added_columns = []

        for cat_name in category_order:
            if cat_name not in category_tables:
                continue  # データが存在しない場合スキップ
            df_cat = category_tables[cat_name]
            padded_tables = []
            for inst in all_instances:
                inst_group = df_cat[df_cat["instance"] == inst]
                n_pad = instance_max_lens[inst] - len(inst_group)
                if n_pad > 0:
                    padding = pd.DataFrame([{'instance': inst, 'time': None, 'value': None}] * n_pad)
                    inst_group = pd.concat([inst_group, padding], ignore_index=True)
                padded_tables.append(inst_group[["time", "value"]])

            df_padded = pd.concat(padded_tables, axis=0, ignore_index=True)
            result[f'{cat_name}_time'] = df_padded['time']
            result[f'{cat_name}_value'] = df_padded['value']
            added_columns.extend([f'{cat_name}_time', f'{cat_name}_value'])

        # --- カラム順序の調整 ---
        # instance, best_known, Other_time/value を最初、その後に他カテゴリを配置
        first_cols = ['instance', 'best_known']
        if 'Other_time' in result.columns and 'Other_value' in result.columns:
            first_cols += ['Other_time', 'Other_value']

        remaining_cols = [col for col in result.columns if col not in first_cols]
        result = result[first_cols + remaining_cols]

        for col in result.columns:
            if col.endswith('_value'):
                result[col] = result[col].abs()


        results.append(get_closest_rows(result))
        fix_result = reshape_full_table(result) # 同じinstance内において合計n件までに圧縮
        # --- 出力 ---
        output_path = dir + f'/{p_name}_fixed_combined.csv'
        fix_result.to_csv(output_path, index=False)

output_path = dir + f'/all_integrated.csv'
pd.concat(results).to_csv(output_path, index=False)
