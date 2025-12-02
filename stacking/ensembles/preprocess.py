from datetime import datetime

import numpy as np
import polars as pl
from loguru import logger


def preprocess(
    df_product_properties: pl.DataFrame,
    df_product_buy: pl.DataFrame,
    df_add_to_cart: pl.DataFrame,
    df_remove_from_cart: pl.DataFrame,
    df_page_visit: pl.DataFrame,
    df_search_query: pl.DataFrame,
    train_start_datetime: datetime,
    train_end_datetime: datetime,
    valid_start_datetime: datetime,
    valid_end_datetime: datetime,
    arr_relevant_clients: np.ndarray,
    arr_propensity_sku: np.ndarray,
    arr_propensity_category: np.ndarray,
):
    split_data = SplitData(
        df_product_buy=df_product_buy,
        df_add_to_cart=df_add_to_cart,
        df_remove_from_cart=df_remove_from_cart,
        df_page_visit=df_page_visit,
        df_search_query=df_search_query,
        train_start_datetime=train_start_datetime,
        train_end_datetime=train_end_datetime,
        valid_start_datetime=valid_start_datetime,
        valid_end_datetime=valid_end_datetime,
    )
    logger.info("Splitting data")
    df_train_period, df_valid_period = split_data.main()

    logger.info("Preprocessing train data")
    train_preprocess_labels = PreprocessLabels(
        df_events=df_train_period,
        df_product_properties=df_product_properties,
        arr_relevant_clients=arr_relevant_clients,
        arr_propensity_sku=arr_propensity_sku,
        arr_propensity_category=arr_propensity_category,
    )
    train_preprocess_labels.add_stacking_labels()

    logger.info("Preprocessing valid data")
    valid_preprocess_labels = PreprocessLabels(
        df_events=df_valid_period,
        df_product_properties=df_product_properties,
        arr_relevant_clients=arr_relevant_clients,
        arr_propensity_sku=arr_propensity_sku,
        arr_propensity_category=arr_propensity_category,
    )
    valid_preprocess_labels.add_stacking_labels()

    return train_preprocess_labels.df_labels, valid_preprocess_labels.df_labels


class SplitData:
    def __init__(
        self,
        df_product_buy: pl.DataFrame,
        df_add_to_cart: pl.DataFrame,
        df_remove_from_cart: pl.DataFrame,
        df_page_visit: pl.DataFrame,
        df_search_query: pl.DataFrame,
        train_start_datetime: datetime,
        train_end_datetime: datetime,
        valid_start_datetime: datetime,
        valid_end_datetime: datetime,
    ):
        logger.info(f"Train start datetime: {train_start_datetime}")
        logger.info(f"Train end datetime: {train_end_datetime}")
        logger.info(f"Valid start datetime: {valid_start_datetime}")
        logger.info(f"Valid end datetime: {valid_end_datetime}")

        logger.info(f"Product buy: {df_product_buy.shape}")
        logger.info(f"Add to cart: {df_add_to_cart.shape}")
        logger.info(f"Remove from cart: {df_remove_from_cart.shape}")
        logger.info(f"Page visit: {df_page_visit.shape}")
        logger.info(f"Search query: {df_search_query.shape}")

        logger.info(
            f"Product buy period: {df_product_buy['timestamp'].min()} - {df_product_buy['timestamp'].max()}"
        )
        logger.info(
            f"Add to cart period: {df_add_to_cart['timestamp'].min()} - {df_add_to_cart['timestamp'].max()}"
        )
        logger.info(
            f"Remove from cart period: {df_remove_from_cart['timestamp'].min()} - {df_remove_from_cart['timestamp'].max()}"
        )
        logger.info(
            f"Page visit period: {df_page_visit['timestamp'].min()} - {df_page_visit['timestamp'].max()}"
        )
        logger.info(
            f"Search query period: {df_search_query['timestamp'].min()} - {df_search_query['timestamp'].max()}"
        )

        self.train_start_datetime = train_start_datetime
        self.train_end_datetime = train_end_datetime
        self.valid_start_datetime = valid_start_datetime
        self.valid_end_datetime = valid_end_datetime

        self.schema = {
            "client_id": pl.UInt32,
            "timestamp": pl.Datetime,
            "sku": pl.UInt32,
            "url": pl.UInt32,
            "query": pl.String,
            "event_type": pl.String,
        }

        # Cast input DataFrames to ensure they match the expected schema
        self.df_product_buy = self._cast_to_schema(df_product_buy)
        self.df_add_to_cart = self._cast_to_schema(df_add_to_cart)
        self.df_remove_from_cart = self._cast_to_schema(df_remove_from_cart)
        self.df_page_visit = self._cast_to_schema(df_page_visit)
        self.df_search_query = self._cast_to_schema(df_search_query)

    def _cast_to_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cast DataFrame columns to match the expected schema."""
        if df.is_empty():
            return pl.DataFrame(schema=self.schema)

        # Cast client_id to UInt32
        if "client_id" in df.columns:
            df = df.with_columns(pl.col("client_id").cast(pl.UInt32))

        # Cast sku to UInt32 if it exists
        if "sku" in df.columns:
            df = df.with_columns(pl.col("sku").cast(pl.UInt32))

        # Cast url to UInt32 if it exists
        if "url" in df.columns:
            df = df.with_columns(pl.col("url").cast(pl.UInt32))

        # Cast timestamp to Datetime if it exists
        if "timestamp" in df.columns:
            if df["timestamp"].dtype == pl.String:
                df = df.with_columns(pl.col("timestamp").str.to_datetime())
            else:
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

        # Cast query to String if it exists
        if "query" in df.columns:
            df = df.with_columns(pl.col("query").cast(pl.String))

        return df

    def main(
        self,
    ):
        df_train_period = self.get_train_period()
        df_valid_period = self.get_valid_period()
        logger.info(f"Train period: {df_train_period.shape}")
        logger.info(f"Valid period: {df_valid_period.shape}")
        return df_train_period, df_valid_period

    def get_train_period(
        self,
    ):
        # Create a list of DataFrames with event_type column added
        dfs = []

        if not self.df_product_buy.is_empty():
            dfs.append(
                self.df_product_buy.with_columns(
                    pl.lit("product_buy").alias("event_type")
                )
            )
        if not self.df_add_to_cart.is_empty():
            dfs.append(
                self.df_add_to_cart.with_columns(
                    pl.lit("add_to_cart").alias("event_type")
                )
            )
        if not self.df_remove_from_cart.is_empty():
            dfs.append(
                self.df_remove_from_cart.with_columns(
                    pl.lit("remove_from_cart").alias("event_type")
                )
            )
        if not self.df_page_visit.is_empty():
            dfs.append(
                self.df_page_visit.with_columns(
                    pl.lit("page_visit").alias("event_type")
                )
            )
        if not self.df_search_query.is_empty():
            dfs.append(
                self.df_search_query.with_columns(
                    pl.lit("search_query").alias("event_type")
                )
            )

        if not dfs:
            return pl.DataFrame(schema=self.schema)

        df = pl.concat(dfs, how="diagonal")

        df = df.filter(
            (df["timestamp"] >= self.train_start_datetime)
            & (df["timestamp"] < self.train_end_datetime)
        )

        return df

    def get_valid_period(
        self,
    ):
        # Create a list of DataFrames with event_type column added
        dfs = []

        if not self.df_product_buy.is_empty():
            dfs.append(
                self.df_product_buy.with_columns(
                    pl.lit("product_buy").alias("event_type")
                )
            )
        if not self.df_add_to_cart.is_empty():
            dfs.append(
                self.df_add_to_cart.with_columns(
                    pl.lit("add_to_cart").alias("event_type")
                )
            )
        if not self.df_remove_from_cart.is_empty():
            dfs.append(
                self.df_remove_from_cart.with_columns(
                    pl.lit("remove_from_cart").alias("event_type")
                )
            )
        if not self.df_page_visit.is_empty():
            dfs.append(
                self.df_page_visit.with_columns(
                    pl.lit("page_visit").alias("event_type")
                )
            )
        if not self.df_search_query.is_empty():
            dfs.append(
                self.df_search_query.with_columns(
                    pl.lit("search_query").alias("event_type")
                )
            )

        if not dfs:
            return pl.DataFrame(schema=self.schema)

        df = pl.concat(dfs, how="diagonal")

        df = df.filter(
            (df["timestamp"] >= self.valid_start_datetime)
            & (df["timestamp"] < self.valid_end_datetime)
        )

        return df


class PreprocessLabels:
    def __init__(
        self,
        df_events: pl.DataFrame,
        df_product_properties: pl.DataFrame,
        arr_relevant_clients: np.ndarray,
        arr_propensity_sku: np.ndarray,
        arr_propensity_category: np.ndarray,
    ) -> None:
        df_events = df_events.with_columns(pl.col("sku").cast(pl.Int64))
        df_product_properties = df_product_properties.with_columns(pl.col("sku").cast(pl.Int64))
        self.df_events = df_events.join(
            df_product_properties,
            on="sku",
            how="left",
            validate="m:1",
        )
        self.arr_relevant_clients = np.sort(arr_relevant_clients)
        self.arr_propensity_sku = np.sort(arr_propensity_sku)
        self.arr_propensity_category = np.sort(arr_propensity_category)

        self.df_labels = pl.DataFrame({"client_id": arr_relevant_clients}).with_columns(
            pl.col("client_id").cast(pl.UInt32)
        )

    def add_stacking_labels(
        self,
    ):
        self.add_churn_label()
        self.add_propensity_sku_labels()
        self.add_propensity_category_labels()

    def add_churn_label(
        self,
    ):
        arr_active_clients = (
            self.df_events.filter(pl.col("event_type") == "product_buy")["client_id"]
            .unique()
            .to_numpy()
        )

        df_active_clients = pl.DataFrame(
            {"client_id": arr_active_clients}
        ).with_columns(pl.col("client_id").cast(pl.UInt32))
        df_active_clients = df_active_clients.with_columns(pl.lit(0).alias("churn"))

        self.df_labels = self.df_labels.join(
            df_active_clients,
            on="client_id",
            how="left",
            validate="1:1",
        )
        self.df_labels = self.df_labels.with_columns(pl.col("churn").fill_null(1))

        logger.info(f"Added churn label to {len(self.df_labels)} clients")
        logger.info(
            f"Churn label: {self.df_labels['churn'].value_counts().sort('churn')}"
        )

    def add_propensity_sku_labels(
        self,
    ):
        df_buy_product = self.df_events.filter(pl.col("event_type") == "product_buy")
        positives_nums = []
        for sku in self.arr_propensity_sku:
            propensity_user = (
                df_buy_product.filter(pl.col("sku") == sku)["client_id"]
                .unique()
                .to_numpy()
            )
            self.df_labels = self.df_labels.with_columns(
                pl.col("client_id")
                .is_in(propensity_user)
                .alias(f"propensity_sku_{sku}")
            )
            positives_nums.append(self.df_labels[f"propensity_sku_{sku}"].sum())
        arr_positives_nums = np.array(positives_nums)
        logger.info("Propensity sku:")
        logger.info(f"min: {arr_positives_nums.min()}")
        logger.info(f"max: {arr_positives_nums.max()}")
        logger.info(f"mean: {arr_positives_nums.mean()}")
        logger.info(f"std: {arr_positives_nums.std()}")

    def add_propensity_category_labels(
        self,
    ):
        df_buy_product = self.df_events.filter(pl.col("event_type") == "product_buy")
        positives_nums = []
        for category in self.arr_propensity_category:
            propensity_user = (
                df_buy_product.filter(pl.col("category") == category)["client_id"]
                .unique()
                .to_numpy()
            )
            self.df_labels = self.df_labels.with_columns(
                pl.col("client_id")
                .is_in(propensity_user)
                .alias(f"propensity_category_{category}")
            )
            positives_nums.append(
                self.df_labels[f"propensity_category_{category}"].sum()
            )
        arr_positives_nums = np.array(positives_nums)
        logger.info("Propensity category:")
        logger.info(f"min: {arr_positives_nums.min()}")
        logger.info(f"max: {arr_positives_nums.max()}")
        logger.info(f"mean: {arr_positives_nums.mean()}")
        logger.info(f"std: {arr_positives_nums.std()}")

