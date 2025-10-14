import torch
from torch import nn
import torch.nn.functional as tF


class DynamicNN(nn.Module):
    def __init__(self, config):
        super(DynamicNN, self).__init__()
        self.layers = nn.ModuleList()
        input_size = config["input_size"]
        output_size = config["output_size"]
        self.dropout_rate = config.get("dropout_rate", 0.5)

        first_layer = nn.Linear(input_size, config["hidden_layer"][0])
        self.layers.append(first_layer)
        for i in range(1, config["n_layers"] + 1):
            in_features = config["neurons"][i - 1]
            out_features = (
                output_size if i == config["n_layers"] else config["hidden_layer"][i]
            )
            self.layers.append(nn.Linear(in_features, out_features))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = tF.relu(x)
            if self.dropout_rate > 0 and i > 1 and i < self.layers.__len__() - 1:
                x = tF.dropout(x, p=self.dropout_rate, training=self.training)

        return x
