import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from loguru import logger


def read_embedding_zip(file_path: Path):
    with zipfile.ZipFile(file_path, "r") as zipf:
        with zipf.open("client_ids.npy") as file:
            arr_clients = np.load(file)
        with zipf.open("embeddings.npy") as file:
            arr_embeddings = np.load(file)
    return arr_clients, arr_embeddings


def save_embeddings(
    arr_clients: np.ndarray, arr_embeddings: np.ndarray, shuffle: bool, output_dir: Path
):
    os.makedirs(output_dir, exist_ok=True)
    if shuffle:
        indices = np.random.permutation(len(arr_clients))
        arr_clients = arr_clients[indices]
        arr_embeddings = arr_embeddings[indices]

    # check first 5 client ids
    logger.info(f"first 5 client ids: {arr_clients[:5]}")

    np.save(output_dir / "client_ids.npy", arr_clients)
    np.save(output_dir / "embeddings.npy", arr_embeddings.astype(np.float16))


class Embedding:
    def __init__(self, name: str, arr_clients: np.ndarray, arr_embeddings: np.ndarray):
        self.name = name
        self.arr_clients = arr_clients
        self.arr_embeddings = arr_embeddings

    def align_to(self, reference_clients: np.ndarray):
        # check if the client sets are identical except for the order
        if not np.array_equal(np.sort(self.arr_clients), np.sort(reference_clients)):
            logger.warning(
                f"Client sets are not identical for {self.name}. Aligning to reference clients."
            )
            # check if arr_clients is a subset of reference_clients
            if not np.all(np.isin(self.arr_clients, reference_clients)):
                raise ValueError(
                    f"Client set {self.name} is not a subset of reference clients"
                )
            else:
                logger.info("clients are subset of reference clients")
                reference_clients = reference_clients[np.isin(reference_clients, self.arr_clients)]

        # Create a mapping from client_id to index for the current embedding
        current_idx_map = {client: idx for idx, client in enumerate(self.arr_clients)}

        # Create reordering indices based on reference clients
        reorder_indices = np.array(
            [current_idx_map[client] for client in reference_clients]
        )

        # Reorder the current embedding's clients and embeddings
        self.arr_clients = self.arr_clients[reorder_indices]
        self.arr_embeddings = self.arr_embeddings[reorder_indices]

    def normalize(self):
        arr_embeddings = self.arr_embeddings.copy().astype(np.float32)
        self.arr_embeddings = arr_embeddings / np.linalg.norm(
            arr_embeddings, axis=-1, keepdims=True
        )

    def info(self):
        logger.info(f"name: {self.name}")
        logger.info(f"arr_clients: {self.arr_clients.shape}")
        logger.info(f"arr_embeddings: {self.arr_embeddings.shape}")


class Embeddings:
    def __init__(self, list_embeddings: list[Embedding]):
        # Check if client sets are identical and handle non-identical cases
        if not self.check_clients_identical(list_embeddings):
            logger.warning("Client sets are not identical across embeddings. Filtering to common clients.")
            self._filter_to_common_clients(list_embeddings)
        else:
            # Keep the original order of the first embedding
            self.arr_clients = list_embeddings[0].arr_clients.copy()

        # Align all other embeddings to match the first embedding's order
        for embedding in list_embeddings[1:]:
            embedding.align_to(self.arr_clients)

        self.arr_embeddings = self.concat_embeddings(list_embeddings)

    def _filter_to_common_clients(self, list_embeddings: list[Embedding]):
        """Filter all embeddings to only include clients that are present in all embeddings."""
        if not list_embeddings:
            raise ValueError("No embeddings provided")
        
        # Find common clients across all embeddings
        common_clients = set(list_embeddings[0].arr_clients)
        for embedding in list_embeddings[1:]:
            common_clients = common_clients.intersection(set(embedding.arr_clients))
        
        if not common_clients:
            raise ValueError("No common clients found across all embeddings")
        
        logger.info(f"Found {len(common_clients)} common clients across all embeddings")
        
        # Convert common_clients to sorted array for consistent ordering
        common_clients_array = np.array(sorted(common_clients))
        
        # Filter each embedding to only include common clients
        for embedding in list_embeddings:
            # Find indices of common clients in this embedding
            common_indices = np.where(np.isin(embedding.arr_clients, common_clients_array))[0]
            
            # Filter clients and embeddings
            embedding.arr_clients = embedding.arr_clients[common_indices]
            embedding.arr_embeddings = embedding.arr_embeddings[common_indices]
            
            logger.info(f"Filtered {embedding.name}: {len(embedding.arr_clients)} clients remaining")
        
        # Set the common clients as the reference
        self.arr_clients = common_clients_array

    def info(self):
        logger.info(f"arr_clients: {self.arr_clients.shape}")
        logger.info(f"arr_embeddings: {self.arr_embeddings.shape}")

    @staticmethod
    def check_clients_identical(list_embeddings: list[Embedding]) -> bool:
        if not list_embeddings:
            logger.warning("No embeddings to compare")
            return True

        # Convert first embedding's clients to set as reference
        reference_clients = set(list_embeddings[0].arr_clients)
        logger.info(
            f"Reference clients from {list_embeddings[0].name}: {len(reference_clients)} unique clients"
        )

        # Compare with all other embeddings
        for embedding in list_embeddings[1:]:
            current_clients = set(embedding.arr_clients)
            logger.info(
                f"Comparing with {embedding.name}: {len(current_clients)} unique clients"
            )

            if current_clients != reference_clients:
                logger.warning(
                    f"Client sets differ between {list_embeddings[0].name} and {embedding.name}"
                )
                # Calculate differences for debugging
                only_in_reference = reference_clients - current_clients
                only_in_current = current_clients - reference_clients
                common_clients = reference_clients.intersection(current_clients)
                
                logger.warning(f"Common clients: {len(common_clients)}")
                if only_in_reference:
                    logger.warning(
                        f"Clients only in {list_embeddings[0].name}: {len(only_in_reference)} clients"
                    )
                if only_in_current:
                    logger.warning(
                        f"Clients only in {embedding.name}: {len(only_in_current)} clients"
                    )
                return False

        logger.info("All client sets are identical!")
        return True

    @staticmethod
    def concat_embeddings(list_embeddings: list[Embedding]):
        arr_embeddings = np.concatenate(
            [embedding.arr_embeddings for embedding in list_embeddings], axis=-1
        )
        return arr_embeddings

    def svd(self, n_components: int, device: str):
        embeddings = torch.from_numpy(self.arr_embeddings).to(
            dtype=torch.float32, device=device
        )
        with torch.no_grad():
            U, S, _ = torch.linalg.svd(embeddings, full_matrices=False)
            reduced = U[:, :n_components] @ torch.diag(S[:n_components])
        self.arr_embeddings = reduced.cpu().numpy().astype(np.float16)
