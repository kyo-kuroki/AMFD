import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# 入力CSVファイル名（適宜変更）
prb = 'mcp'
input_csv = os.path.dirname(os.path.dirname(__file__)) + '/amfd/results_k5/{}_results.csv'.format(prb)
output_csv = os.path.dirname(__file__) + '/{}_results.csv'.format(prb)
output_plot_dir = os.path.dirname(__file__) + '/plots'
# solutionsファイルの読み込み（例: 'solutions.txt'）
solutions_file = os.path.dirname(os.path.dirname(__file__)) +'/datasets/{}/solutions'.format(prb)

# CSV読み込み
df = pd.read_csv(input_csv)

# 1. tuningの行をinstanceごとに1つだけ残す
tuning_df = df[df['process'] == 'tuning'].drop_duplicates(subset='instance')
non_tuning_df = df[df['process'] != 'tuning']
df_filtered = pd.concat([tuning_df, non_tuning_df], ignore_index=True)

# 2. constraint satisfactionがFalseの行のvalue, eta, zetaをNoneにする
df_filtered = df_filtered[df_filtered['step_scale'] != 1]

# 3. best known solutionを読み込み、instance列でマージ

# 行を1つずつ読み込み、辞書に変換
best_solutions = {}
with open(solutions_file, 'r') as f:
    for line in f:
        if ':' in line:
            instance, value = line.strip().split(':')
            best_solutions[instance.strip()] = int(value.strip())

# 辞書をSeriesにしてマージ
solution_series = pd.Series(best_solutions, name='best known solution')
solution_series.index.name = 'instance'

# instance列でマージ
df_filtered = df_filtered.merge(solution_series, on='instance', how='left')
df_filtered.sort_values(by='instance', inplace=True, kind='stable')
df_filtered.reset_index(drop=True, inplace=True)

# 4. 処理後のCSVを保存
df_filtered.to_csv(output_csv, index=False)


# インスタンスごとのグループ
groups = list(df_filtered.groupby('instance'))
n = len(groups)

# サブプロットの行数・列数（例: 3列にする）
ncols = 3
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

for idx, (instance, group) in enumerate(groups):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]

    group = group.dropna(subset=['value'])  # NaN除去
    if group.empty:
        ax.set_visible(False)
        continue

    ax.plot(group['time'], group['value'], marker='o')
    ax.set_title(instance)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.grid(True)

# 空白のサブプロットを非表示にする
for idx in range(n, nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row][col].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_plot_dir, f"{prb}_all.pdf"))
plt.close()

print("全インスタンスのグラフを1枚にまとめて保存しました: combined_plot.png")

