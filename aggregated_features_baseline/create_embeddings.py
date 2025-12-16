import argparse
import logging
from typing import List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.utils import load_with_properties
from data_utils.data_dir import DataDir
from features_aggregator import FeaturesAggregator
from constants import EVENT_TYPE_TO_COLUMNS, EventTypes
from calculators import parse_to_array
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
from collections import defaultdict
import numpy as np

K = 5        # Top-K 品类数
EMB_DIM = 16 # 每个商品名称的向量维度
client_skus = defaultdict(list)
sku_to_category = {}
def load_relevant_clients_ids(input_dir: Path) -> np.ndarray:
    #return np.load(input_dir / "relevant_clients.npy")
    return np.load(input_dir / "relevant_clients.npy")

def load_product_name_embeddings(product_properties_path: Path) -> dict:
    logger.info("Loading product_properties.parquet to get product_name embeddings...")
    product_properties = pd.read_parquet(product_properties_path)
    sku_to_name_embed = {}
    for row in product_properties.itertuples():
        sku = row.sku
        name_vector = parse_to_array(row.name)
        sku_to_name_embed[sku] = name_vector
        sku_to_category[sku] = row.category
        
    product_properties = pd.read_parquet(product_properties_path)
    sku_to_name_embed = {}
    for row in product_properties.itertuples():
        sku = row.sku
        name_vector = parse_to_array(row.name)
        sku_to_name_embed[sku] = name_vector
    return sku_to_name_embed

def save_embeddings(
    embeddings_dir: Path, embeddings: np.ndarray, client_ids: np.ndarray, mode: str
):
    logger.info("Saving embeddings")
    embeddings_dir = os.path.join(embeddings_dir, mode)
    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(os.path.join(embeddings_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(embeddings_dir, "client_ids.npy"),  client_ids)

def create_embeddings(
    data_dir: DataDir,
    mode: str,
    num_days: List[int],
    top_n: int,
    relevant_client_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    aggregator = FeaturesAggregator(
        num_days=num_days,
        top_n=top_n,
        relevant_client_ids=relevant_client_ids,
    )

    product_name_embeddings = load_product_name_embeddings(
        data_dir.data_dir / "product_properties.parquet"
    )

    logger.info("Generating features and collecting product name embeddings...")
    all_events = {}
    product_name_vectors_by_client = {}


    for event_type in EVENT_TYPE_TO_COLUMNS.keys():
        logger.info("Processing event type: %s", event_type.value)
        event_df = load_with_properties(data_dir=data_dir, event_type=event_type.value, mode=mode)
        event_df["timestamp"] = pd.to_datetime(event_df.timestamp)
        all_events[event_type] = event_df

        if event_type in [EventTypes.PRODUCT_BUY, EventTypes.ADD_TO_CART, EventTypes.REMOVE_FROM_CART]:
            filtered_df = event_df[event_df["client_id"].isin(relevant_client_ids)]
            for row in filtered_df.itertuples():
                client_id = row.client_id
                sku = row.sku
                if sku in product_name_embeddings:
                    vector = product_name_embeddings[sku]
                    product_name_vectors_by_client.setdefault(client_id, []).append(vector)
                    client_skus[client_id].append(sku)


        aggregator.generate_features(
            event_type=event_type,
            client_id_column="client_id",
            df=event_df,
            columns=EVENT_TYPE_TO_COLUMNS[event_type],
        )

    logger.info("Merging features with recency, frequency, and product name embeddings...")
    client_ids, base_embeddings = aggregator.merge_features(all_events=all_events)

    emb = np.array(base_embeddings, dtype=np.float64)

    core_emb = emb

    # 对core_emb做标准化及对数变换
    mu = core_emb.mean(axis=0)
    sigma = core_emb.std(axis=0) + 1e-6
    sigma[sigma < 1e-3] = 1.0
    high_var = sigma > 10
    core_emb[:, high_var] = np.log1p(core_emb[:, high_var])
    mu[high_var] = core_emb[:, high_var].mean(axis=0)
    sigma[high_var] = core_emb[:, high_var].std(axis=0) + 1e-6
    core_emb_std = (core_emb - mu) / sigma

    # 按子块进行 L2 归一化
    block_ranges = [
        (0, 181),   # product_buy_stats
        (181, 362), # add_to_cart_stats
        (362, 543), # remove_from_cart_stats
        (543, 604), # page_visit_stats
        (604, 665), # search_query_stats
        (665, 669), # recency
        (669, 677), # counts
        (677, 678), # entropy
        (678, 683), # 30d_ratios
        (683, 686), # lifecycle
        (686, 710), # ewa
    ]
    for start, end in block_ranges:
        blk = core_emb_std[:, start:end]
        norm = np.linalg.norm(blk, axis=1, keepdims=True) + 1e-6
        core_emb_std[:, start:end] = blk / norm

    emb_standard = core_emb_std.astype(np.float64)

    print("standard shape:", emb_standard.shape)
    return np.array(client_ids), emb_standard

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default='../dataset/ubc_data_tiny/',
        help="Directory with input and target data – produced by data_utils.split_data",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='valid',
        help="Directory with input and target data – produced by data_utils.split_data",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default='../submit/agg_feats/',
        help="Directory where to store generated embeddings",
    )
    parser.add_argument(
        "--num-days",
        nargs="*",
        type=int,
        default=[1, 7, 14, 30, 90, 180],
        help="Numer of days to compute features",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top column values to consider in feature generation",
    )
    return parser

def main(params):
    data_dir = DataDir(Path(params.data_dir))
    embeddings_dir = Path(params.embeddings_dir)

    relevant_client_ids = load_relevant_clients_ids(input_dir=data_dir.data_dir)
    client_ids, embeddings = create_embeddings(
        data_dir=data_dir,
        mode=params.mode,
        num_days=params.num_days,
        top_n=params.top_n,
        relevant_client_ids=relevant_client_ids,
    )

    save_embeddings(
        client_ids=client_ids,
        embeddings=embeddings,
        embeddings_dir=embeddings_dir,
        mode=params.mode
    )

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params=params)
