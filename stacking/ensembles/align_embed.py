import copy

import numpy as np
from loguru import logger
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

from ensembles.embeddings import Embedding


class StatisticalAlignment:
    """
    Statistical alignment of embeddings using Procrustes analysis.

    This class aligns one embedding space to another using PCA and orthogonal Procrustes
    transformation, ensuring statistical compatibility between different embedding spaces.

    The alignment process involves:
    1. Finding common clients between embeddings
    2. Computing PCA components for both embeddings
    3. Finding orthogonal transformation using Procrustes analysis
    4. Applying the transformation to align the target embedding to the base embedding
    """

    def __init__(
        self,
        embedding_base: Embedding,
        embedding_to_align: Embedding,
        n_components: int,
    ):
        """
        Initialize the statistical alignment.

        Args:
            embedding_base: Reference embedding to align to
            embedding_to_align: Embedding to be aligned
            n_components: Number of PCA components to use for alignment

        Raises:
            ValueError: If n_components is invalid or embeddings have incompatible dimensions
        """
        # Input validation
        if n_components <= 0:
            raise ValueError("n_components must be positive")

        if embedding_base.arr_embeddings.shape[1] < n_components:
            raise ValueError(
                f"n_components ({n_components}) cannot be larger than base embedding dimension "
                f"({embedding_base.arr_embeddings.shape[1]})"
            )

        if embedding_to_align.arr_embeddings.shape[1] < n_components:
            raise ValueError(
                f"n_components ({n_components}) cannot be larger than target embedding dimension "
                f"({embedding_to_align.arr_embeddings.shape[1]})"
            )

        self.embedding_base = embedding_base
        self.embedding_to_align = embedding_to_align
        self.n_components = n_components

        logger.info(
            f"Initializing StatisticalAlignment with {n_components} PCA components"
        )
        logger.info(
            f"Base embedding: {embedding_base.name}, shape: {embedding_base.arr_embeddings.shape}"
        )
        logger.info(
            f"Target embedding: {embedding_to_align.name}, shape: {embedding_to_align.arr_embeddings.shape}"
        )

        self.embedding_base_common, self.embedding_to_align_common = (
            self.match_embeddings(
                embedding_base,
                embedding_to_align,
            )
        )

    def match_embeddings(
        self,
        embedding_1: Embedding,
        embedding_2: Embedding,
    ) -> tuple[Embedding, Embedding]:
        """
        Find common clients between two embeddings and create aligned copies.

        Args:
            embedding_1: First embedding
            embedding_2: Second embedding

        Returns:
            Tuple of embeddings with only common clients, aligned in the same order

        Raises:
            ValueError: If no common clients are found or insufficient common clients
        """
        # Find common clients
        common_clients = np.intersect1d(
            embedding_1.arr_clients,
            embedding_2.arr_clients,
        )

        if len(common_clients) == 0:
            raise ValueError("No common clients found between embeddings")

        if len(common_clients) < self.n_components:
            raise ValueError(
                f"Number of common clients ({len(common_clients)}) is less than "
                f"n_components ({self.n_components}). This may cause PCA to fail."
            )

        logger.info(
            f"Found {len(common_clients)} common clients out of "
            f"{len(embedding_1.arr_clients)} and {len(embedding_2.arr_clients)} total clients"
        )

        # Create indices for common clients
        idx_1 = np.isin(embedding_1.arr_clients, common_clients)
        idx_2 = np.isin(embedding_2.arr_clients, common_clients)

        # Create new embeddings with only common clients
        embedding_1_common = Embedding(
            embedding_1.name,
            embedding_1.arr_clients[idx_1].copy(),
            embedding_1.arr_embeddings[idx_1].copy(),
        )
        embedding_2_common = Embedding(
            embedding_2.name,
            embedding_2.arr_clients[idx_2].copy(),
            embedding_2.arr_embeddings[idx_2].copy(),
        )

        # Align both embeddings to the same client order
        embedding_1_common.align_to(common_clients)
        embedding_2_common.align_to(common_clients)

        return embedding_1_common, embedding_2_common

    def align(self) -> Embedding:
        """
        Perform statistical alignment of the target embedding to the base embedding.

        Returns:
            Aligned embedding with the same statistical properties as the base embedding

        Raises:
            RuntimeError: If PCA or Procrustes analysis fails
        """
        logger.info(
            f"Starting statistical alignment with {len(self.embedding_base_common.arr_clients)} common clients"
        )
        logger.info(f"Requested {self.n_components} PCA components")

        try:
            # Compute statistics for base embedding using safer methods
            # Use float64 for better numerical stability
            base_embeddings_float64 = self.embedding_base_common.arr_embeddings.astype(
                np.float64
            )
            target_embeddings_float64 = (
                self.embedding_to_align_common.arr_embeddings.astype(np.float64)
            )

            # Clip extreme values before computing statistics
            base_embeddings_float64 = np.clip(base_embeddings_float64, -1e6, 1e6)
            target_embeddings_float64 = np.clip(target_embeddings_float64, -1e6, 1e6)

            mean_base = np.mean(base_embeddings_float64, axis=0)
            std_base = np.std(base_embeddings_float64, axis=0)

            # Compute statistics for target embedding using safer methods
            mean_to_align = np.mean(target_embeddings_float64, axis=0)
            std_to_align = np.std(target_embeddings_float64, axis=0)

            # Handle numerical stability issues with more robust thresholds
            std_base = np.where(std_base < 1e-8, 1e-8, std_base)
            std_to_align = np.where(std_to_align < 1e-8, 1e-8, std_to_align)

            # Check for numerical issues
            if np.any(np.isnan(mean_base)) or np.any(np.isinf(mean_base)):
                raise RuntimeError("Base embedding mean contains NaN or Inf values")
            if np.any(np.isnan(mean_to_align)) or np.any(np.isinf(mean_to_align)):
                raise RuntimeError("Target embedding mean contains NaN or Inf values")
            if np.any(np.isnan(std_base)) or np.any(np.isinf(std_base)):
                raise RuntimeError("Base embedding std contains NaN or Inf values")
            if np.any(np.isnan(std_to_align)) or np.any(np.isinf(std_to_align)):
                raise RuntimeError("Target embedding std contains NaN or Inf values")

            # Dynamically adjust n_components if needed for numerical stability
            actual_n_components = min(
                self.n_components,
                len(self.embedding_base_common.arr_clients) - 1,
                self.embedding_base_common.arr_embeddings.shape[1],
                self.embedding_to_align_common.arr_embeddings.shape[1],
            )

            if actual_n_components != self.n_components:
                logger.info(
                    f"Adjusted n_components from {self.n_components} to {actual_n_components} for numerical stability"
                )

            # Fit PCA models
            pca_base = PCA(n_components=actual_n_components, random_state=42)
            pca_to_align = PCA(n_components=actual_n_components, random_state=42)

            # Use clipped embeddings for PCA fitting
            pca_base.fit(base_embeddings_float64)
            pca_to_align.fit(target_embeddings_float64)

            logger.info(
                f"PCA explained variance ratio - Base: {pca_base.explained_variance_ratio_.sum():.3f}, "
                f"Target: {pca_to_align.explained_variance_ratio_.sum():.3f}"
            )
            logger.info(f"Using {actual_n_components} PCA components for alignment")

            # Compute orthogonal Procrustes transformation
            R, scale = orthogonal_procrustes(
                pca_to_align.components_.T, pca_base.components_.T
            )

            logger.info(f"Procrustes scale factor: {scale:.6f}")

            # Apply transformation to the full target embedding
            # Use clipped embeddings for transformation
            target_embeddings_full = self.embedding_to_align.arr_embeddings.astype(
                np.float64
            )
            target_embeddings_full = np.clip(target_embeddings_full, -1e6, 1e6)

            normalized = (target_embeddings_full - mean_to_align) / std_to_align

            # Clip extreme values to prevent overflow
            normalized = np.clip(normalized, -10, 10)

            # Project to PCA space, apply transformation, then project back
            # First, project to PCA space
            projected = normalized @ pca_to_align.components_.T

            # Apply the transformation in PCA space
            transformed = projected @ R

            # Project back to original space
            rotated = transformed @ pca_base.components_

            # Apply inverse normalization with clipping
            aligned_embeddings = rotated * std_base + mean_base
            aligned_embeddings = np.clip(
                aligned_embeddings, -100, 100
            )  # Prevent extreme values

            # Create aligned embedding
            aligned_embedding = copy.deepcopy(self.embedding_to_align)
            aligned_embedding.arr_embeddings = aligned_embeddings

            logger.info("Statistical alignment completed successfully")
            logger.info(
                f"Aligned embedding shape: {aligned_embedding.arr_embeddings.shape}"
            )

            return aligned_embedding

        except Exception as e:
            logger.error(f"Statistical alignment failed: {str(e)}")
            raise RuntimeError(f"Statistical alignment failed: {str(e)}") from e
