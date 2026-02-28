import torch
import torch.nn as nn

class MultiLayerPerceptoron(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        super().__init__()
        layers = list()
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PLE(nn.Module):
    def __init__(
        self,
        user_emb_dim,
        num_shared_experts,
        num_task_experts,
        num_tasks,
        expert_hidden_dims,
        expert_output_dim,
        task_tower_hidden_dims,
    ):
        super().__init__()

        self.user_emb_dim = user_emb_dim
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.num_tasks = num_tasks

        self.shared_expert_1 = nn.ModuleList(
            [
                MultiLayerPerceptoron(
                    user_emb_dim, expert_hidden_dims, expert_output_dim, dropout=0.01
                )
                for i in range(num_shared_experts)
            ]
        )

        self.task_expert_1 = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        MultiLayerPerceptoron(
                            user_emb_dim,
                            expert_hidden_dims,
                            expert_output_dim,
                            dropout=0.01,
                        )
                        for i in range(num_task_experts)
                    ]
                )
                for j in range(self.num_tasks)
            ]
        )

        self.gate_1 = nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        user_emb_dim,
                        (
                            self.num_shared_experts + self.num_task_experts
                            if i < self.num_tasks
                            else self.num_shared_experts
                            + self.num_tasks * self.num_task_experts
                        ),
                    ),
                    torch.nn.Softmax(dim=1),
                )
                for i in range(self.num_tasks + 1)
            ]
        )

        self.shared_expert_2 = nn.ModuleList(
            [
                MultiLayerPerceptoron(
                    expert_output_dim,
                    expert_hidden_dims,
                    expert_output_dim,
                    dropout=0.01,
                )
                for i in range(num_shared_experts)
            ]
        )

        self.task_expert_2 = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        MultiLayerPerceptoron(
                            expert_output_dim,
                            expert_hidden_dims,
                            expert_output_dim,
                            dropout=0.01,
                        )
                        for i in range(num_task_experts)
                    ]
                )
                for j in range(self.num_tasks)
            ]
        )
        self.gate_2 = nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        expert_output_dim,
                        self.num_shared_experts + self.num_task_experts,
                    ),
                    torch.nn.Softmax(dim=1),
                )
                for i in range(self.num_tasks)
            ]
        )
        self.churn_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=1, dropout=0.1
        )
        self.add_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=1, dropout=0.1
        )
        self.buy_sku_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=100, dropout=0.1
        )
        self.buy_cat_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=100, dropout=0.1
        )
        self.buy_price_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=100, dropout=0.1
        )
        self.add_sku_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=100, dropout=0.1
        )
        self.add_cat_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=100, dropout=0.1
        )
        self.add_price_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=100, dropout=0.1
        )

    def forward(self, user_emb):
        # [(B, NUM_EXPERTS), ...]
        shared_expert_outputs = torch.cat(
            [
                self.shared_expert_1[i](user_emb).unsqueeze(1)
                for i in range(self.num_shared_experts)
            ],
            dim=1,
        )

        concat_feats = []
        all_task_expert_outputs = []
        for i in range(self.num_tasks):
            task_expert_outputs = torch.cat(
                [
                    self.task_expert_1[i][j](user_emb).unsqueeze(1)
                    for j in range(self.num_task_experts)
                ],
                dim=1,
            )
            concat_feats.append(
                torch.cat([shared_expert_outputs, task_expert_outputs], dim=1)
            )
            all_task_expert_outputs.append(task_expert_outputs)

        all_task_expert_outputs.append(shared_expert_outputs)
        concat_feats.append(torch.cat(all_task_expert_outputs, dim=1))

        gate_value = [self.gate_1[i](user_emb) for i in range(self.num_tasks + 1)]

        task_feats = [
            torch.bmm(gate_value[i].unsqueeze(1), concat_feats[i]).squeeze(1)
            for i in range(self.num_tasks + 1)
        ]

        shared_experts_outputs = torch.cat(
            [
                self.shared_expert_2[i](task_feats[-1]).unsqueeze(1)
                for i in range(self.num_shared_experts)
            ],
            dim=1,
        )
        expert_outputs = []
        for i in range(self.num_tasks):
            task_expert_outputs = torch.cat(
                [
                    self.task_expert_2[i][j](task_feats[i]).unsqueeze(1)
                    for j in range(self.num_task_experts)
                ],
                dim=1,
            )
            expert_outputs.append(
                torch.cat([shared_experts_outputs, task_expert_outputs], dim=1)
            )
        gate_value = [self.gate_2[i](task_feats[i]) for i in range(self.num_tasks)]
        tower_inputs = [
            torch.bmm(gate_value[i].unsqueeze(1), expert_outputs[i]).squeeze(1)
            for i in range(self.num_tasks)
        ]

        churn_logits = self.churn_tower(tower_inputs[0])
        buy_sku_logits = self.buy_sku_tower(tower_inputs[1])
        buy_cat_logits = self.buy_cat_tower(tower_inputs[2])

        return (
            churn_logits,
            buy_sku_logits,
            buy_cat_logits,
        )