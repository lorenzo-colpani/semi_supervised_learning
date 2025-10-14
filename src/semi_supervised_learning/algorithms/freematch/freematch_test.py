from semi_supervised_learning.utils.dynamic_nn import DynamicNN
import torch
from torch.utils.data import DataLoader
import polars as pl
from torchmetrics import MatthewsCorrCoef
from torchmetrics.classification import BinaryAccuracy

# Define the path to your saved model file
model_path = "/home/lorenzo/ray_results/my_model/my_train_run/TorchTrainer_f98c7_00000_0_2025-10-15_01-17-38/checkpoint_000049/model.pth"
config = {
    "batch_size": 64,
    "input_size": 250,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "num_epochs": 50,
    "num_classes": 2,
    "w_u": 1.0,
    "w_f": 0.5,
    "ema_decay": 0.999,
    "binary": True,
    "hidden_layers": [128, 64, 32],
    "n_layers": 3,
    "dropout_rate": 0.5,
    "output_size": 1,
}  # Instantiate your model
model = DynamicNN(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Load the entire state dictionary
checkpoint_state = torch.load(model_path)
model.load_state_dict(checkpoint_state)

# Set the model to evaluation mode
model.eval()

# load dataset
dataset = pl.read_parquet("./data/synthetic_dataset.parquet")
dataset = dataset.to_torch(
    label=pl.col("label").cast(pl.Int32),
    features=pl.col("features").cast(pl.Array(pl.Float32, 250)),
    return_type="dataset",
)
inference_dataloader = DataLoader(
    dataset, batch_size=1048, shuffle=False, num_workers=8
)

matt = MatthewsCorrCoef("binary", num_classes=2).to(device)
accuracy = BinaryAccuracy(threshold=0.5).to(device)
for features, label in inference_dataloader:
    features = features.to(device)
    labels = label.unsqueeze(1).float().to(device)
    output = model(features)
    matt.update(output, labels)
    accuracy.update(output, labels)
print("Matthews correlation coefficient:", matt.compute().item())
print("Accuracy:", accuracy.compute().item())

