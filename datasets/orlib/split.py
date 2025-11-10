import re
import os

def split_file_by_marker(input_path: str, output_dir: str):
    """
    ファイルを「2500 xxx」形式の行で区切って分割保存する。

    Args:
        input_path: 元のテキストファイルパス
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    # 2列形式の区切り行を正規表現で検出
    pattern = re.compile(r'^\s*\d+\s+\d+\s*$')

    with open(input_path, 'r') as f:
        lines = f.readlines()

    parts = []
    current_part = []

    for line in lines:
        if pattern.match(line.strip()):
            # 新しいセクションが始まるので保存
            if current_part:
                parts.append(current_part)
                current_part = []
            current_part.append(line)
        else:
            current_part.append(line)

    if current_part:
        parts.append(current_part)

    # 出力
    for i, part in enumerate(parts, start=0):
        output_path = os.path.join(output_dir, f"bqp2500_{i:02d}.bqp")
        with open(output_path, 'w') as out:
            out.writelines(part)
        print(f"Saved: {output_path} ({len(part)} lines)")

# 使用例
if __name__ == "__main__":
    split_file_by_marker("/work2/k-kuroki/AMFD/datasets/orlib/bqp2500", "/work2/k-kuroki/AMFD/datasets/orlib")
