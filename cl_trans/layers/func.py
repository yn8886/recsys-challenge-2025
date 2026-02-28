import numpy as np
import polars as pl
from datetime import timedelta
from datetime import datetime
import os
from sklearn.preprocessing import MinMaxScaler
from config import SEMANTIC_GROUPS

def calc_ewa_and_window_counts(df: pl.DataFrame, action_type: str, last_timestamp: datetime):
    """
    计算短期窗口计数 (7d, 30d) 和 指数衰减计数 (EWA)。
    """
    half_lives = [1, 7, 14, 30, 90, 180]

    # 计算时间差 (天)
    df = df.with_columns(
        (last_timestamp - pl.col("timestamp")).dt.total_days().alias("days")
    )

    agg_exprs = [
        # 窗口计数
        pl.col("days").filter(pl.col("days") <= 7).len().alias(f"{action_type}_count_7d"),
        pl.col("days").filter(pl.col("days") <= 30).len().alias(f"{action_type}_count_30d")
    ]

    # EWA 计数
    for tau in half_lives:
        lam = np.log(2) / tau
        col_name = f"{action_type}_ewa_{tau}d"
        # math: sum( exp(-lambda * days) )
        agg_exprs.append(
            (-pl.col("days") * lam).exp().sum().alias(col_name)
        )

    return df.group_by("client_id").agg(agg_exprs)


def calc_category_entropy_30d(buy_df, add_df, remove_df, relevant_clients, last_timestamp: datetime):
    """
    计算过去 30 天用户交互类别的香农熵。
    """
    cutoff_time = last_timestamp - timedelta(days=30)

    dfs = []
    for d in [buy_df, add_df, remove_df]:
        if not d.is_empty():
            d_recent = d.filter((pl.col("timestamp") >= cutoff_time) & pl.col("category").is_not_null())
            dfs.append(d_recent.select(["client_id", "category"]))

    if not dfs:
        return pl.DataFrame({"client_id": relevant_clients, "category_entropy_30d": 0.0})

    combined_df = pl.concat(dfs)

    # 统计每个 client_id 下每个类别的频次
    cat_counts = combined_df.group_by(["client_id", "category"]).len("cnt")
    # 统计每个 client_id 的总交互数
    totals = cat_counts.group_by("client_id").agg(pl.col("cnt").sum().alias("total"))

    # 计算概率和熵
    entropy_df = cat_counts.join(totals, on="client_id")
    entropy_df = entropy_df.with_columns(
        (pl.col("cnt") / pl.col("total")).alias("p")
    ).with_columns(
        (-pl.col("p") * pl.col("p").log()).alias("h_part")
    )

    # 聚合得到最终的 log1p(H)
    result = entropy_df.group_by("client_id").agg(
        (pl.col("h_part").sum()).log1p().alias("category_entropy_30d")
    )

    # 补全缺失的用户
    base_df = pl.DataFrame({"client_id": relevant_clients})
    return base_df.join(result, on="client_id", how="left").fill_null(0.0)


def save_dataframe(df: pl.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    scaler = MinMaxScaler()

    # 1. 提取并保存各语义分组的统计特征矩阵
    saved_cols = set()
    for group_name, cols in SEMANTIC_GROUPS.items():
        # 确保 DataFrame 中包含这些列，缺失则补 0
        existing_cols = [c for c in cols if c in df.columns]
        missing_cols = [c for c in cols if c not in df.columns]

        group_df = df.select(existing_cols)
        if missing_cols:
            group_df = group_df.with_columns([pl.lit(0.0).alias(c) for c in missing_cols])

        # 提取当前 group 的特征顺序
        group_df = group_df.select(cols).fill_null(0.0).fill_nan(0.0)
        feats_matrix = group_df.to_numpy()

        # 归一化并保存
        scaled_feats = scaler.fit_transform(feats_matrix)
        norm = np.linalg.norm(scaled_feats, axis=1, keepdims=True) + 1e-6
        scaled_feats = scaled_feats / norm

        file_name = f"{group_name}.npy"
        save_path = os.path.join(save_dir, file_name)
        print(f"Saving {file_name} with shape {scaled_feats.shape}")
        np.save(save_path, scaled_feats)

        saved_cols.update(cols)

    # 2. 保存非稠密统计特征外的其他序列/目标列 (如 query_ids, target_sku_id 等)
    df_rest = df.drop(list(saved_cols))
    for col_name in df_rest.columns:
        array = df_rest[col_name].to_numpy()
        file_name = col_name + ".npy"
        save_path = os.path.join(save_dir, file_name)
        np.save(save_path, array)