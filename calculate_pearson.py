import pandas as pd
from scipy.stats import pearsonr
from itertools import combinations

csv_files = [
    (
        "IF=1",
        "file_name.csv",
    ),
    (
        "IF=200",
        "file_name.csv",
    ),
]

columns = [
    "avg_age",
    "per_class_acc_last",
    "per_class_avg_margin_last",
    "per_class_avg_loss_last",
]


def compute_pairwise_pearson(file_path):
    df = pd.read_csv(file_path)
    rows = []

    for col_x, col_y in combinations(columns, 2):
        data = df[[col_x, col_y]].dropna()
        r, p_value = pearsonr(data[col_x], data[col_y])

        rows.append({
            "variable_1": col_x,
            "variable_2": col_y,
            "pearson_r": r,
            "p_value": p_value,
        })

    return pd.DataFrame(rows)


for table_title, csv_file_path in csv_files:
    result_table = compute_pairwise_pearson(csv_file_path)

    print(f"\n===== {table_title} =====")
    print(
        result_table.to_string(
            index=False,
            formatters={
                "pearson_r": lambda x: f"{x:.6f}",
                "p_value": lambda x: f"{x:.6e}",
            },
        )
    )
