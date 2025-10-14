import ray
from ray import train, tune
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn.functional as tF
import torch.optim as optim
from torch import nn
import polars as pl
from torchmetrics import MatthewsCorrCoef

from semi_supervised_learning.utils.dynamic_nn import DynamicNN
from semi_supervised_learning.utils.data_generator import (
    generate_labeled_unlabeled,
    generate_train_test,
)
from semi_supervised_learning.utils.utils import strong_augmentation, weak_augmentation


def train_freematch(
    config,
    # labeled_train_data,
    # unlabeled_train_data,
    # labeled_val_data,
    # unlabeled_val_data,
):
    # datasets
    labeled_train_data = ray.train.get_dataset_shard("labeled_train_data")
    unlabeled_train_data = ray.train.get_dataset_shard("unlabeled_train_data")
    # Hyperparameters
    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 0.001)
    weight_decay = config.get("weight_decay", 1e-4)
    num_epochs = config.get("num_epochs", 50)
    num_classes = config.get("num_classes", 2)

    # freematch hyperparameters
    w_u = config.get("w_u", 1.0)
    w_f = config.get("w_f", 0.5)
    ema_decay = config.get("ema_decay", 0.999)

    # freematch initialization
    global_threshold = torch.tensor(1.0 / num_classes)
    class_probabilities = torch.ones(num_classes) / num_classes
    class_histogram = torch.ones(num_classes) / num_classes

    model = DynamicNN(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = (
        nn.CrossEntropyLoss() if not config["binary"] else nn.BCEWithLogitsLoss()
    )

    model.train()
    label_train_loader = labeled_train_data.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float32
    )
    unlabel_train_loader = unlabeled_train_data.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float32
    )
    label_val_loader = labeled_val_data.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float32
    )
    unlabel_val_loader = unlabeled_val_data.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float32
    )

    for epoch in range(num_epochs):
        matt = MatthewsCorrcoef(num_classes=num_classes).to(device)
        running_loss = torch.tensor(0.0).to(device)
        running_total = torch.tensor(0.0).to(device)

        model.train()
        for label_batch, unlabeled_batch in zip(
            label_train_loader, unlabel_train_loader
        ):
            features_l, label_l = label_batch
            (features_u, _), _ = unlabeled_batch

            optimizer.zero_grad()
            # supervised loss
            logits_l = model(features_l)
            loss_l = criterion(logits_l, label_l)
            matt.update(logits_l, label_l)

            # pseudo-labeling
            with torch.no_grad():
                weak_features = weak_augmentation(features_u)
                logits_u_w = model(weak_features)
                pseudo_label_probs = tF.softmax(logits_u_w, dim=-1)
                pseudo_max_probs, pseudo_labels = torch.max(pseudo_label_probs, dim=-1)

            # self-adaptive thresholding
            new_global_threshold = (ema_decay * global_threshold) + (
                1 - ema_decay
            ) * pseudo_max_probs.mean()
            global_threshold.copy_(new_global_threshold)

            new_class_probabilities = ema_decay * class_probabilities
            for c in range(num_classes):
                class_c_mask = pseudo_labels == c
                if class_c_mask.sum() > 0:
                    class_c_prob = pseudo_max_probs[class_c_mask, c].mean()
                    new_class_probabilities[c] += (1 - ema_decay) * class_c_prob
            class_probabilities.copy_(new_class_probabilities)

            normalized_class_probabilities = class_probabilities / torch.max(
                class_probabilities
            )
            final_thresholds = global_threshold * normalized_class_probabilities

            thresholds_for_samples = final_thresholds[pseudo_labels]
            mask = pseudo_max_probs.ge(thresholds_for_samples)

            # unsupervised loss
            strong_features = strong_augmentation(features_u)
            logits_u_s = model(strong_features)
            unsupervised_loss = criterion(logits_u_s, pseudo_labels)
            loss_u = (unsupervised_loss * mask).mean()

            # Self-adaptive Feairness
            current_pseudo_hist = torch.bincount(
                pseudo_labels, minlength=num_classes
            ).float()
            new_class_histogram = (ema_decay * class_histogram) + (
                1 - ema_decay
            ) * current_pseudo_hist
            class_histogram.copy_(new_class_histogram)

            p_norm = class_probabilities / class_probabilities.sum()
            h_norm = class_histogram / class_histogram.sum()

            loss_f = -torch.sum(p_norm * torch.log(h_norm + 1e-8))

            total_loss = loss_l + w_u * loss_u + w_f * loss_f
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.detach()
            running_total += features_l.size(0)

        epoch_loss = running_loss / running_total
        train.report(
            {
                "epoch": epoch,
                "loss": epoch_loss.item(),
                "matthews_corrcoef": matt.compute().item(),
            }
        )
        matt.reset()


if __name__ == "__main__":
    ray.init(num_gpus=1)

    dataset_train = ray.data.from_arrow(
        pl.read_parquet("./data/synthetic_dataset.parquet").to_arrow()
    )
    dataset_label, dataset_unlabel = generate_labeled_unlabeled(dataset_train)

    config = {
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "num_epochs": 50,
        "num_classes": 2,
        "w_u": 1.0,
        "w_f": 0.5,
        "ema_decay": 0.999,
        "binary": True,
        "input_dim": 20,
        "hidden_layers": [64, 32],
        "dropout_rate": 0.5,
    }

    scaling_config = ScalingConfig(num_workers=1, use_gpu=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_freematch,
        train_loop_config=config,
        dataset={
            "labeled_train_data": dataset_label,
            "unlabeled_train_data": dataset_unlabel,
        },
        scaling_config=scaling_config,
    )

    result = trainer.fit()
