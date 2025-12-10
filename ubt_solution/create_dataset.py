import argparse
import os
from datetime import datetime
from enum import Enum
import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

DATASET_DIR = "../dataset/ubc_data_tiny"

SAVE_DIR = "../dataset/mtl"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TARGET_DIR = os.path.join(DATASET_DIR, "target")
ONLY_RELEVANT_CLIENTS = True


TARGET_TARGET_START = datetime(2022, 10, 13, 0, 0, 0)
VALID_TARGET_START = datetime(2022, 10, 27, 0, 0, 0)
TARGET_END_DATETIME = datetime(2022, 11, 11, 0, 0, 0)

CANDIDATES_STATS_COLS = [
    "sku_add_count",
    "category_add_count",
    "price_add_count",
    "sku_buy_count",
    "category_buy_count",
    "price_buy_count",
    "sku_remove_count",
    "category_remove_count",
    "price_remove_count",
]


STATS_COLS = [
    "add_flag",
    "unique_add_sku",
    "unique_add_category",
    "unique_add_price",
    "unique_add_name",
    "add_price_mean",
    "buy_flag",
    "unique_buy_sku",
    "unique_buy_category",
    "unique_buy_price",
    "unique_buy_name",
    "buy_price_mean",
    "remove_flag",
    "unique_remove_sku",
    "unique_remove_category",
    "unique_remove_price",
    "unique_remove_name",
    "remove_price_mean",
    "page_visit_flag",
    "unique_url",
    "search_query_flag",
    "unique_search_query",
    "all_from_last_to_end_days",
    "min_all_from_first_to_end_days",
    "all_duration_days",
    "unique_all_timestamp",
    "add_from_last_to_end_days",
    "min_add_from_first_to_end_days",
    "add_duration_days",
    "unique_add_timestamp",
    "buy_from_last_to_end_days",
    "min_buy_from_first_to_end_days",
    "buy_duration_days",
    "unique_buy_timestamp",
    "remove_from_last_to_end_days",
    "min_remove_from_first_to_end_days",
    "remove_duration_days",
    "unique_remove_timestamp",
    "visit_from_last_to_end_days",
    "min_visit_from_first_to_end_days",
    "visit_duration_days",
    "unique_visit_timestamp",
    "search_from_last_to_end_days",
    "min_search_from_first_to_end_days",
    "search_duration_days",
    "unique_search_timestamp",
]


class EventType(Enum):
    PAD_IDX = 0
    MASK = 1
    PRODUCT_BUY = 2
    ADD_TO_CART = 3
    REMOVE_FROM_CART = 4
    PAGE_VISIT = 5
    SEARCH_QUERY = 6


def filter_by_client_id(df, client_ids):
    df = df.filter(pl.col("client_id").is_in(client_ids))
    return df


def count_target_column_value(
    df: pl.DataFrame,
    candidates: np.ndarray,
    action_type: str,
    target_column: str,
) -> pl.DataFrame:
    assert target_column in ["sku", "category", "price"]

    target_df = df.filter(pl.col(target_column).is_in(candidates))
    count_df = target_df.group_by("client_id", target_column).len(
        f"{target_column}_{action_type}_count"
    )
    candidate_item_df = pl.DataFrame({target_column: candidates})
    all_combinations_df = count_df.select(pl.col("client_id").unique()).join(
        candidate_item_df, how="cross"
    )

    count_df = count_df.with_columns([
        pl.col(target_column).cast(all_combinations_df[target_column].dtype)
    ])

    candidate_item_count_df = (
        all_combinations_df.join(count_df, on=["client_id", target_column], how="left")
        .fill_null(0)
        .sort("client_id", target_column)
        .group_by("client_id")
        .agg(
            pl.col(f"{target_column}_{action_type}_count").sort_by(target_column),
        )
    )
    return candidate_item_count_df


def add_timestamp_feature(
    df: pl.DataFrame,
    action_type: str,
    end_timestamp: datetime,
) -> pl.DataFrame:
    timestamp_feature_df = df.group_by("client_id").agg(
        (end_timestamp - pl.col("timestamp").max())
        .dt.total_days()
        .alias(f"{action_type}_from_last_to_end_days"),
        (end_timestamp - pl.col("timestamp").min())
        .dt.total_days()
        .alias(f"min_{action_type}_from_first_to_end_days"),
        (pl.col("timestamp").max() - pl.col("timestamp").min())
        .dt.total_days()
        .alias(f"{action_type}_duration_days"),
        pl.col("timestamp").n_unique().alias(f"unique_{action_type}_timestamp"),
    )
    return timestamp_feature_df


def calc_sku_statistical_feature(
    df: pl.DataFrame,
    candidate_sku: np.ndarray,
    candidate_cat: np.ndarray,
    candidate_price: np.ndarray,
    relevant_clients: np.ndarray,
    action_type: str,
    last_timestamp: datetime,
) -> pl.DataFrame:
    assert action_type in ["add", "buy", "remove"]
    sku_stats_df = pl.DataFrame({"client_id": relevant_clients})
    sku_stats_df = sku_stats_df.with_columns(
        pl.when(pl.col("client_id").is_in(df["client_id"]))
        .then(1)
        .otherwise(0)
        .alias(f"{action_type}_flag")
    )

    # candidate sku/category/price count
    sku_count_df = count_target_column_value(df, candidate_sku, action_type, "sku")
    cat_count_df = count_target_column_value(df, candidate_cat, action_type, "category")
    price_count_df = count_target_column_value(
        df, candidate_price, action_type, "price"
    )

    # unique sku/category/price/name count
    unique_count_df = (
        df.select("client_id", "sku", "category", "price", "name")
        .group_by("client_id")
        .n_unique()
        .rename(
            {
                "sku": f"unique_{action_type}_sku",
                "category": f"unique_{action_type}_category",
                "price": f"unique_{action_type}_price",
                "name": f"unique_{action_type}_name",
            },
        )
    )

    # avg price
    avg_price_df = df.group_by("client_id").agg(
        pl.col("price").mean().alias(f"{action_type}_price_mean")
    )

    sku_stats_df = (
        sku_stats_df.join(sku_count_df, on="client_id", how="left")
        .join(cat_count_df, on="client_id", how="left")
        .join(price_count_df, on="client_id", how="left")
        .join(unique_count_df, on="client_id", how="left")
        .join(avg_price_df, on="client_id", how="left")
    )

    sku_stats_df = (
        sku_stats_df.with_columns(
            pl.col(f"sku_{action_type}_count").fill_null([0] * 100),
            pl.col(f"category_{action_type}_count").fill_null([0] * 100),
            pl.col(f"price_{action_type}_count").fill_null([0] * 100),
        )
        .fill_null(0)
        .with_columns(
            pl.col(f"sku_{action_type}_count").list.to_array(100),
            pl.col(f"category_{action_type}_count").list.to_array(100),
            pl.col(f"price_{action_type}_count").list.to_array(100),
        )
    )
    timefeatures_df = add_timestamp_feature(df, action_type, last_timestamp)

    sku_stats_df = sku_stats_df.join(
        timefeatures_df, on="client_id", how="left"
    ).fill_null(0)

    return sku_stats_df

def calc_url_statistical_feature(
    df: pl.DataFrame,
    relevant_clients: np.ndarray,
    action_type: str,
    last_timestamp: datetime,
):
    stats_df = pl.DataFrame({"client_id": relevant_clients})
    stats_df = stats_df.with_columns(
        pl.when(pl.col("client_id").is_in(df["client_id"]))
        .then(1)
        .otherwise(0)
        .alias("page_visit_flag")
    )
    url_stats_df = df.group_by("client_id").agg(
        pl.col("url").n_unique().alias("unique_url"),
    )

    timefeatures_df = add_timestamp_feature(df, action_type, last_timestamp)
    stats_df = stats_df.join(timefeatures_df, on="client_id", how="left").fill_null(0)

    stats_df = stats_df.join(url_stats_df, on="client_id", how="left").fill_null(0)
    return stats_df


def calc_search_statistical_feature(
    df: pl.DataFrame,
    relevant_clients: np.ndarray,
    action_type: str,
    last_timestamp: datetime,
):
    stats_df = pl.DataFrame({"client_id": relevant_clients})
    stats_df = stats_df.with_columns(
        pl.when(pl.col("client_id").is_in(df["client_id"]))
        .then(1)
        .otherwise(0)
        .alias("search_query_flag")
    )
    search_stats_df = df.group_by("client_id").agg(
        pl.col("query").n_unique().alias("unique_search_query"),
    )
    timefeatures_df = add_timestamp_feature(df, action_type, last_timestamp)
    stats_df = stats_df.join(timefeatures_df, on="client_id", how="left").fill_null(0)

    stats_df = stats_df.join(search_stats_df, on="client_id", how="left").fill_null(0)
    return stats_df


def save_dataframe(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    stats_feats = df.select(STATS_COLS).to_numpy()
    scaler = MinMaxScaler()
    scaled_stats_features = scaler.fit_transform(stats_feats)

    file_name = "stats_features.npy"
    save_path = os.path.join(save_dir, file_name)
    print(f"Saveing {file_name}")
    np.save(save_path, scaled_stats_features)

    df = df.drop(CANDIDATES_STATS_COLS + STATS_COLS)
    col_names = df.columns

    for col_name in col_names:
        array = df[col_name].to_numpy()
        file_name = col_name + ".npy"
        save_path = os.path.join(save_dir, file_name)
        np.save(save_path, array)


def create_positive_negative_samples(
        agg_product_buy_df: pl.DataFrame,
        all_client_ids: np.ndarray,
        sku2id_mapping: dict,
        propensity_sku_original: np.ndarray,
        propensity_cat_original: np.ndarray,
        product_prop_df: pl.DataFrame,
        num_negatives: int = 40,
) :
    buy_map = {row["client_id"]: row["buy_sku"] for row in agg_product_buy_df.iter_rows(named=True)}

    all_sku_ids = product_prop_df["sku_id"].unique().to_numpy()
    all_sku_set = set(all_sku_ids)
    all_cat_ids = product_prop_df["category_id"].unique().to_numpy()
    all_cat_set = set(all_cat_ids)

    sku_to_cat_map = \
        product_prop_df.select(['sku', pl.col("category_id")]).unique().to_pandas().set_index(
            'sku')['category_id'].to_dict()

    propensity_sku_mapped = [sku2id_mapping.get(sku, 2) for sku in propensity_sku_original]
    propensity_sku_ids = [sku_id for sku_id in propensity_sku_mapped if sku_id >= 3]

    propensity_cat_ids = [cat + 2 for cat in propensity_cat_original]

    client_ids = []
    pos_sku_ids_list = []
    neg_sku_ids_list = []
    pos_cat_ids_list = []
    neg_cat_ids_list = []
    is_churn_list = []

    for client_id in tqdm(all_client_ids, desc="Creating Pos/Neg/Churn Samples"):
        buy_sku_original = buy_map.get(client_id)

        # 确定正例 (Positives)
        if buy_sku_original is not None and len(buy_sku_original) > 0:
            is_churn = 0
            pos_sku_ids_mapped = [sku2id_mapping.get(sku, 2) for sku in buy_sku_original]
            pos_sku_ids = list(set([sku_id for sku_id in pos_sku_ids_mapped if sku_id >= 3]))

            # 计算正例类别 ID
            pos_cat_ids_mapped = [sku_to_cat_map.get(sku_id, 2) for sku_id in pos_sku_ids]
            pos_cat_ids = list(set([cat_id for cat_id in pos_cat_ids_mapped if cat_id >= 3]))
        else:
            is_churn = 1
            pos_sku_ids = []
            pos_cat_ids = []

        pos_sku_set = set(pos_sku_ids)
        pos_cat_set = set(pos_cat_ids)

        neg_sku_ids = []

        for sku_id in propensity_sku_ids:
            if sku_id not in pos_sku_set:
                neg_sku_ids.append(sku_id)
                if len(neg_sku_ids) >= num_negatives:
                    break

        if len(neg_sku_ids) < num_negatives:
            remaining_negatives_needed = num_negatives - len(neg_sku_ids)
            available_for_sampling = list(all_sku_set - pos_sku_set - set(neg_sku_ids))

            if available_for_sampling:
                num_to_sample = min(remaining_negatives_needed, len(available_for_sampling))
                new_neg_sku_ids = np.random.choice(
                    available_for_sampling,
                    size=num_to_sample,
                    replace=False
                ).tolist()
                neg_sku_ids.extend(new_neg_sku_ids)

        neg_cat_ids = []

        for cat_id in propensity_cat_ids:
            if cat_id not in pos_cat_set:
                neg_cat_ids.append(cat_id)
                if len(neg_cat_ids) >= num_negatives:
                    break

        if len(neg_cat_ids) < num_negatives:
            remaining_negatives_needed = num_negatives - len(neg_cat_ids)
            available_for_sampling = list(all_cat_set - pos_cat_set - set(neg_cat_ids))

            if available_for_sampling:
                num_to_sample = min(remaining_negatives_needed, len(available_for_sampling))
                new_neg_cat_ids = np.random.choice(
                    available_for_sampling,
                    size=num_to_sample,
                    replace=False
                ).tolist()
                neg_cat_ids.extend(new_neg_cat_ids)

        client_ids.append(client_id)
        pos_sku_ids_list.append(pos_sku_ids)
        neg_sku_ids_list.append(neg_sku_ids)
        pos_cat_ids_list.append(pos_cat_ids)
        neg_cat_ids_list.append(neg_cat_ids)
        is_churn_list.append(is_churn)


    data = {
        "client_id": client_ids,
        "pos_sku_ids": pos_sku_ids_list,
        "neg_sku_ids": neg_sku_ids_list,
        "pos_cat_ids": pos_cat_ids_list,
        "neg_cat_ids": neg_cat_ids_list,
        "is_churn": is_churn_list,
    }

    schema = {
        "client_id": pl.Int64,
        "pos_sku_ids": pl.List(pl.Int64),
        "neg_sku_ids": pl.List(pl.Int64),
        "pos_cat_ids": pl.List(pl.Int64),
        "neg_cat_ids": pl.List(pl.Int64),
        "is_churn": pl.Int32,
    }

    label_df = pl.from_dict(data=data, schema=schema)
    return label_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="valid",
        choices=["train", "valid"],
    )

    args = parser.parse_args()
    dataset_type = args.dataset_type
    INPUT_DIR = TRAIN_DIR if dataset_type == 'train' else VALID_DIR

    # 1. Load ubc_data
    relevant_client_ids = np.load(os.path.join(DATASET_DIR, "relevant_clients.npy"))
    candidate_sku = np.load(os.path.join(TARGET_DIR, "propensity_sku.npy"))
    candidate_cat = np.load(os.path.join(TARGET_DIR, "propensity_category.npy"))
    product_prop_df = pl.read_parquet(
        os.path.join(DATASET_DIR, "product_properties.parquet")
    )
    add_to_cart_df = pl.read_parquet(os.path.join(INPUT_DIR, "add_to_cart.parquet"))
    add_to_cart_df = filter_by_client_id(add_to_cart_df, relevant_client_ids)
    page_visit_df = pl.read_parquet(os.path.join(DATASET_DIR, "page_visit.parquet"))
    page_visit_df = filter_by_client_id(page_visit_df, relevant_client_ids)
    product_buy_df = pl.read_parquet(os.path.join(DATASET_DIR, "product_buy.parquet"))
    product_buy_df = filter_by_client_id(product_buy_df, relevant_client_ids)
    remove_from_cart_df = pl.read_parquet(
        os.path.join(DATASET_DIR, "remove_from_cart.parquet")
    )
    remove_from_cart_df = filter_by_client_id(remove_from_cart_df, relevant_client_ids)
    search_query_df = pl.read_parquet(os.path.join(DATASET_DIR, "search_query.parquet"))
    search_query_df = filter_by_client_id(search_query_df, relevant_client_ids)

    # 2. make user stats data
    candidate_price = np.arange(100)

    buy_with_prop_df = product_buy_df.join(product_prop_df, on="sku", how="inner")
    add_with_prop_df = add_to_cart_df.join(product_prop_df, on="sku", how="inner")
    remove_with_prop_df = remove_from_cart_df.join(
        product_prop_df, on="sku", how="inner"
    )
    LAST_TIMESTAMP = (
        TARGET_TARGET_START if dataset_type == "train" else VALID_TARGET_START
    )
    add_stats_df = calc_sku_statistical_feature(
        add_with_prop_df,
        candidate_sku,
        candidate_cat,
        candidate_price,
        relevant_client_ids,
        "add",
        LAST_TIMESTAMP,
    )

    buy_stats_df = calc_sku_statistical_feature(
        buy_with_prop_df,
        candidate_sku,
        candidate_cat,
        candidate_price,
        relevant_client_ids,
        "buy",
        LAST_TIMESTAMP,
    )
    remove_stats_df = calc_sku_statistical_feature(
        remove_with_prop_df,
        candidate_sku,
        candidate_cat,
        candidate_price,
        relevant_client_ids,
        "remove",
        LAST_TIMESTAMP,
    )

    visit_stats_df = calc_url_statistical_feature(
        page_visit_df, relevant_client_ids, "visit", LAST_TIMESTAMP
    )

    search_stats_df = calc_search_statistical_feature(
        search_query_df, relevant_client_ids, "search", LAST_TIMESTAMP
    )

    all_timestamp_df = pl.concat(
        [
            add_with_prop_df.select(["client_id", "timestamp"]),
            buy_with_prop_df.select(["client_id", "timestamp"]),
            remove_with_prop_df.select(["client_id", "timestamp"]),
            page_visit_df.select(["client_id", "timestamp"]),
            search_query_df.select(["client_id", "timestamp"]),
        ]
    )
    all_timefeature_df = add_timestamp_feature(all_timestamp_df, "all", LAST_TIMESTAMP)

    stats_df = (
        add_stats_df.join(buy_stats_df, on="client_id", how="inner")
        .join(remove_stats_df, on="client_id", how="inner")
        .join(visit_stats_df, on="client_id", how="inner")
        .join(search_stats_df, on="client_id", how="inner")
        .join(all_timefeature_df, on="client_id", how="inner")
    )
    del add_stats_df, buy_stats_df, remove_stats_df, visit_stats_df, search_stats_df

    # 3. Mapping original id
    # Mapping starts from 3 (PAD_IDX=0, MASK=1, UNK=2)
    start_id = 3
    mapping_skus = np.sort(product_prop_df["sku"].unique().to_numpy())
    sku2id_mapping = {
        sku_id: i for i, sku_id in enumerate(mapping_skus, start=start_id)
    }
    product_buy_df = product_buy_df.with_columns(
        pl.col("sku").replace(sku2id_mapping, default=2).alias("sku_id")
    )
    add_to_cart_df = add_to_cart_df.with_columns(
        pl.col("sku").replace(sku2id_mapping, default=2).alias("sku_id")
    )
    remove_from_cart_df = remove_from_cart_df.with_columns(
        pl.col("sku").replace(sku2id_mapping, default=2).alias("sku_id")
    )

    # Mapping starts from 3 (PAD_IDX=0, MASK=1, UNK=2)
    pv_for_mapping_df = pl.read_parquet(os.path.join(DATASET_DIR, "page_visit.parquet"))
    pv_for_mapping_df = pv_for_mapping_df.filter(
        pl.col("client_id").is_in(relevant_client_ids)
    )

    MIN_CNT = 10
    url_count_df = pv_for_mapping_df.group_by("url").len("count")
    mapping_urls = url_count_df.filter(pl.col("count") >= MIN_CNT)["url"].to_numpy()
    mapping_urls = np.sort(mapping_urls)
    url2id_mapping = {
        url_id: i for i, url_id in enumerate(mapping_urls, start=start_id)
    }
    page_visit_df = page_visit_df.with_columns(
        pl.col("url").replace(url2id_mapping, default=2).alias("url_id")
    )
    del pv_for_mapping_df

    product_prop_df = product_prop_df.with_columns(
        pl.col("sku").replace(sku2id_mapping, default=2).alias("sku_id"),
        (pl.col("category") + 2).alias("category_id"),
        (pl.col("price") + 2).alias("price_id"),
    )

    # Mapping starts from 3 (PAD_IDX=0, MASK=1, UNK=2)
    # Convert string to list[pl.int64]
    search_query_df = search_query_df.with_columns(
        pl.col("query")
        .str.extract_all(r"\d+")  # 文字列中の数値をすべて抽出 (結果は List[String] 型)
        .list.eval(pl.element().cast(pl.Int64), parallel=True)
        .alias("word_ids")
    )

    # Adding 2 to word_ids for PAD_IDX and MASK
    query_id_offset = 2
    search_query_df = search_query_df.with_columns(
        pl.col("word_ids").list.eval(pl.element() + query_id_offset)
    )

    product_prop_df = product_prop_df.with_columns(
        pl.col("name")
        .str.extract_all(r"\d+")
        .list.eval(pl.element().cast(pl.Int64), parallel=True)
        .alias("word_ids")
    )
    # Adding 2 to word_ids for PAD_IDX and MASK
    product_prop_df = product_prop_df.with_columns(
        pl.col("word_ids").list.eval(pl.element() + query_id_offset)
    )

    target_cols = [
        "client_id",
        "event_type",
        "timestamp",
        "sku_id",
        "url_id",
        "word_ids",
        "category_id",
        "price_id",
    ]
    add_to_cart_df = add_to_cart_df.join(product_prop_df, on="sku_id", how="inner")
    add_to_cart_df = add_to_cart_df.with_columns(
        pl.lit(EventType.ADD_TO_CART.value).alias("event_type"),
        pl.lit(-1).alias("url_id").cast(pl.Int64),
    ).select(target_cols)

    product_buy_df = product_buy_df.join(product_prop_df, on="sku_id", how="inner")
    product_buy_df = product_buy_df.with_columns(
        pl.lit(EventType.PRODUCT_BUY.value).alias("event_type"),
        pl.lit(-1).alias("url_id").cast(pl.Int64),
    ).select(target_cols)

    remove_from_cart_df = remove_from_cart_df.join(
        product_prop_df, on="sku_id", how="inner"
    )
    remove_from_cart_df = remove_from_cart_df.with_columns(
        pl.lit(EventType.REMOVE_FROM_CART.value).alias("event_type"),
        pl.lit(-1).alias("url_id").cast(pl.Int64),
    ).select(target_cols)

    page_visit_df = page_visit_df.with_columns(
        pl.lit(EventType.PAGE_VISIT.value).alias("event_type"),
        pl.lit(-1).alias("sku_id").cast(pl.Int64),
        pl.lit([-1] * 16).alias("word_ids"),
        pl.lit(-1).alias("category_id").cast(pl.Int64),
        pl.lit(-1).alias("price_id").cast(pl.Int64),
    ).select(target_cols)

    search_query_df = search_query_df.with_columns(
        pl.lit(EventType.SEARCH_QUERY.value).alias("event_type"),
        pl.lit(-1).alias("sku_id").cast(pl.Int64),
        pl.lit(-1).alias("url_id").cast(pl.Int64),
        pl.lit(-1).alias("category_id").cast(pl.Int64),
        pl.lit(-1).alias("price_id").cast(pl.Int64),
    ).select(target_cols)

    event_df = pl.concat(
        [
            add_to_cart_df,
            product_buy_df,
            remove_from_cart_df,
            page_visit_df.unique(),
            search_query_df.unique(),
        ]
    )

    event_df = event_df.with_columns(
        pl.col("timestamp").min().over("client_id").alias("min_ts")
    ).with_columns(
        (pl.col("timestamp") - pl.col("min_ts")).dt.total_seconds().floordiv(86400.0)
        .alias("norm_timestamp")
    ).drop("min_ts")

    target_cols = [
        "client_id",
        "event_type",
        "timestamp",
        "norm_timestamp",
        "sku_id",
        "url_id",
        "word_ids",
        "category_id",
        "price_id",
    ]
    event_df = event_df.select(target_cols)

    input_df = event_df.group_by("client_id").agg(
        [
            pl.col("timestamp").sort(),
            pl.col("norm_timestamp").sort_by("timestamp"),
            pl.col("event_type").sort_by("timestamp"),
            pl.col("sku_id").sort_by("timestamp"),
            pl.col("url_id").sort_by("timestamp"),
            pl.col("word_ids").sort_by("timestamp"),
            pl.col("category_id").sort_by("timestamp"),
            pl.col("price_id").sort_by("timestamp"),
        ]
    )

    # 4. Create target data
    if dataset_type == 'train':
        product_buy_target_df = pl.read_parquet(os.path.join(TARGET_DIR, "train_target.parquet"))
    else:
        product_buy_target_df = pl.read_parquet(os.path.join(TARGET_DIR, "validation_target.parquet"))

    candidate_sku = np.load(os.path.join(TARGET_DIR, "propensity_sku.npy"))
    candidate_cat = np.load(os.path.join(TARGET_DIR, "propensity_category.npy"))

    agg_product_buy_df = product_buy_target_df.group_by("client_id").agg(
        pl.col("sku").alias("buy_sku"),
    )

    label_df = create_positive_negative_samples(
        agg_product_buy_df,
        relevant_client_ids,
        sku2id_mapping,
        candidate_sku,
        candidate_cat,
        product_prop_df,
        num_negatives=120,
    )

    all_df = input_df.join(label_df, on="client_id", how="inner")

    all_df = all_df.with_columns(
        pl.col("pos_sku_ids").fill_null(pl.lit([])).alias("pos_sku_ids"),
        pl.col("neg_sku_ids").fill_null(pl.lit([])).alias("neg_sku_ids"),
        pl.col("pos_cat_ids").fill_null(pl.lit([])).alias("pos_cat_ids"),
        pl.col("neg_cat_ids").fill_null(pl.lit([])).alias("neg_cat_ids"),
    )


    all_df = all_df.join(stats_df, on="client_id", how="inner")
    all_df = all_df.sort(by="client_id")
    del input_df, label_df, stats_df

    save_dir = os.path.join(SAVE_DIR, dataset_type)
    save_dataframe(all_df, save_dir)


if __name__ == "__main__":
    main()
