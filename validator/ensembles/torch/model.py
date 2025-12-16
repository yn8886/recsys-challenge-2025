from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class StackingModelOutputs:
    embedding: torch.Tensor
    churn_logits: Optional[torch.Tensor] = None,
    churn_cart_logits: Optional[torch.Tensor] = None,
    propensity_sku_logits: Optional[torch.Tensor] = None,
    propensity_category_logits: Optional[torch.Tensor] = None,
    cart_sku_logits: Optional[torch.Tensor] = None,
    cart_category_logits: Optional[torch.Tensor] = None


class StackingModel(nn.Module):
    def __init__(
        self,
        tasks: list[str],
        input_dim: int,
        embedding_dim: int,
        hidden_size_thin: int,
        hidden_size_wide: int,
        num_propensity_sku: Optional[int] = None,
        num_propensity_category: Optional[int] = None,
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, embedding_dim)
        self.gelu = nn.GELU()

        self.input_projection = nn.Linear(embedding_dim, hidden_size_thin)
        self.ln_input = nn.LayerNorm(normalized_shape=hidden_size_thin)

        self.layernorms = nn.ModuleList(
            [nn.LayerNorm(normalized_shape=hidden_size_thin) for _ in range(3)]
        )
        self.bottlenecks = nn.ModuleList(
            [
                BottleneckBlock(thin_dim=hidden_size_thin, wide_dim=hidden_size_wide)
                for _ in range(3)
            ]
        )

        self.ln_output = nn.LayerNorm(normalized_shape=hidden_size_thin)

        self.tasks = tasks

        if "churn" in tasks:
            self.churn_head = nn.Linear(hidden_size_thin, 1)

        if "churn_cart" in tasks:
            self.churn_cart_head = nn.Linear(hidden_size_thin, 1)

        if "propensity_sku" in tasks:
            assert num_propensity_sku is not None
            self.propensity_sku_head = nn.Linear(hidden_size_thin, num_propensity_sku)

        if "propensity_category" in tasks:
            assert num_propensity_category is not None
            self.propensity_category_head = nn.Linear(
                hidden_size_thin, num_propensity_category
            )

        if "cart_sku" in tasks:
            assert num_propensity_sku is not None
            self.cart_sku_head = nn.Linear(hidden_size_thin, num_propensity_sku)

        if "cart_category" in tasks:
            assert num_propensity_category is not None
            self.cart_category_head = nn.Linear(
                hidden_size_thin, num_propensity_category
            )

        logger.info(f"StackingModel initialized with tasks: {self.tasks}")

    def forward(self, input_embedding: torch.Tensor) -> StackingModelOutputs:
        embedding = self.linear(input_embedding)
        embedding = self.gelu(embedding)

        x = self.input_projection(embedding)
        x = self.ln_input(x)
        for layernorm, bottleneck in zip(self.layernorms, self.bottlenecks):
            x = x + bottleneck(layernorm(x))
        x = self.ln_output(x)

        if "churn" in self.tasks:
            churn_logits = self.churn_head(x)
        else:
            churn_logits = None

        if "churn_cart" in self.tasks:
            churn_cart_logits = self.churn_cart_head(x)
        else:
            churn_cart_logits = None

        if "propensity_sku" in self.tasks:
            propensity_sku_logits = self.propensity_sku_head(x)
        else:
            propensity_sku_logits = None

        if "propensity_category" in self.tasks:
            propensity_category_logits = self.propensity_category_head(x)
        else:
            propensity_category_logits = None

        if "cart_sku" in self.tasks:
            cart_sku_logits = self.cart_sku_head(x)
        else:
            cart_sku_logits = None

        if "cart_category" in self.tasks:
            cart_category_logits = self.cart_category_head(x)
        else:
            cart_category_logits = None

        return StackingModelOutputs(
            embedding=embedding,
            churn_logits=churn_logits,
            churn_cart_logits=churn_cart_logits,
            propensity_sku_logits=propensity_sku_logits,
            propensity_category_logits=propensity_category_logits,
            cart_sku_logits=cart_sku_logits,
            cart_category_logits=cart_category_logits,
        )


class BottleneckBlock(nn.Module):
    """
    Inverted Bottleneck.
    Taken from "Scaling MLPs: A Tale of Inductive Bias" https://arxiv.org/pdf/2306.13575.pdf.
    The idea is to first expand the input to a wider hidden size, then apply a nonlinearity,
    and finally project back to the original dimension.
    """

    def __init__(self, thin_dim: int, wide_dim: int):
        super().__init__()
        self.l1 = nn.Linear(thin_dim, wide_dim)
        self.l2 = nn.Linear(wide_dim, thin_dim)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x
