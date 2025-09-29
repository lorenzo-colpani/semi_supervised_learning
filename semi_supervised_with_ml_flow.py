import copy
from itertools import cycle

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.pruners import MedianPruner
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from optuna.integration import MLflowCallback

mlflow.set_tracking_uri("http://localhost:8080")


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def add_gaussian_noise(data_tensor, std=0.1):
    """Adds Gaussian noise to a PyTorch tensor."""
    noise = torch.randn_like(data_tensor) * std
    return data_tensor + noise


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        return x


def load_data(train_batch: int):
    iris = load_iris()
    # shuffle dataset before split
    np.random.seed(42)
    indices = np.random.permutation(len(iris.data))
    iris.data = iris.data[indices]
    iris.target = iris.target[indices]
    X = iris.data
    y = iris.target

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    full_dataset = TensorDataset(X_tensor, y_tensor)

    # split with label and unlabel data. Assume 20% labeled data
    label_data, unlabel_data = train_test_split(
        full_dataset, test_size=0.8, random_state=42
    )

    # Split data into training and testing sets
    train_data, test_data = train_test_split(label_data, test_size=0.2, random_state=42)

    train_data, validation_data = train_test_split(
        train_data, test_size=0.2, random_state=42
    )

    batch_size = 1024

    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    unlabeled_loader = DataLoader(unlabel_data, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader, unlabeled_loader


mlflow.set_tracking_uri("http://localhost:8080")
experiment_id = get_or_create_experiment("mean_teacher_iris_experiment")
mlflow.set_experiment("mean_teacher_iris_experiment")

# Apply the decorator to your objective function


def objective_normal(trial):
    params = {
        "batch_size_train": trial.suggest_categorical(
            "batch_size_train", [16, 32, 64, 128, 256, 512, 1024]
        ),
        "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 50, 500),
    }
    train_loader, val_loader, _, _ = load_data(params["batch_size_train"])
    input_size = 4
    output_size = 3
    model = SimpleNN(input_size, output_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    for epoch in range(params["num_epochs"]):
        # Training
        model.train()
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        train_accuracy = correct_train / total_train

        # Evaluate on validation set
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_outputs = model(val_inputs)
                _, predicted_val = torch.max(val_outputs.data, 1)
                total_val += val_labels.size(0)
                correct_val += (predicted_val == val_labels).sum().item()
        validation_accuracy = correct_val / total_val
        trial.report(validation_accuracy, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    mlflow.log_params(params)
    mlflow.log_metric("train accuracy", train_accuracy)
    # mlflow.log_metric('validation accuracy', validation_accuracy)

    return validation_accuracy


def update_teacher_weights(student_model, teacher_model, alpha=0.99):
    for teacher_param, student_param in zip(
        teacher_model.parameters(), student_model.parameters()
    ):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)


def objective_mean_teacher(trial):
    with mlflow.start_run(nested=True):
        params = {
            "batch_size_train": trial.suggest_categorical(
                "batch_size_train", [16, 32, 64, 128, 256, 512, 1024]
            ),
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "num_epochs": trial.suggest_int("num_epochs", 50, 500),
            "alpha": trial.suggest_float("alpha", 0.80, 0.999),
            "lambda_u": trial.suggest_float("alpha", 1, 10),
        }
        input_size = 4
        output_size = 3
        student_model = SimpleNN(input_size, output_size)
        # Instantiate the teacher model with the same architecture
        teacher_model = copy.deepcopy(student_model)

        # The teacher model should not have its gradients calculated
        for param in teacher_model.parameters():
            param.requires_grad = False
        supervised_loss_fn = nn.CrossEntropyLoss()
        consistency_loss_fn = nn.MSELoss()
        train_loader, val_loader, _, unlabeled_loader = load_data(
            params["batch_size_train"]
        )
        optimizer = optim.Adam(student_model.parameters(), lr=params["lr"])

        for epoch in range(params["num_epochs"]):
            student_model.train()
            for (labeled_data, labeled_labels), (unlabeled_data, _) in zip(
                cycle(train_loader), unlabeled_loader
            ):
                # Supervised loss
                labeled_outputs = student_model(labeled_data)
                supervised_loss = supervised_loss_fn(labeled_outputs, labeled_labels)

                # 1. Weakly augmented view for the TEACHER
                unlabeled_data_teacher = add_gaussian_noise(unlabeled_data, std=0.05)

                # 2. Strongly augmented view for the STUDENT
                unlabeled_data_student = add_gaussian_noise(unlabeled_data, std=0.15)

                # Consistency loss
                with torch.no_grad():
                    teacher_outputs = teacher_model(unlabeled_data_teacher)
                student_outputs_unlabeled = student_model(unlabeled_data_student)
                consistency_loss = consistency_loss_fn(
                    student_outputs_unlabeled, teacher_outputs
                )

                total_loss = (1 - params["lambda_u"]) * supervised_loss + params[
                    "lambda_u"
                ] * consistency_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Update teacher model's weights using EMA
                update_teacher_weights(student_model, teacher_model, params["alpha"])
            # Evaluate on validation set
            student_model.eval()
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_outputs = student_model(val_inputs)
                    _, predicted_val = torch.max(val_outputs.data, 1)
                    total_val += val_labels.size(0)
                    correct_val += (predicted_val == val_labels).sum().item()
            validation_accuracy = correct_val / total_val
            trial.report(validation_accuracy, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        # log model parameters and validation accuracy to MLflow:
        mlflow.pytorch.log_model(model=student_model, artifact_path="student_model")

        mlflow.log_params(params)
        mlflow.log_metric("validation accuracy", validation_accuracy)


# Set the current active MLflow experiment

# override Optuna's default logging to ERROR only
# optuna.logging.set_verbosity(optuna.logging.ERROR)

# 1. Define the storage location and a name for your study
storage_name = "sqlite:///my_optuna_study.db"
study_name = "supervised_iris_study"
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
# Create a study object and specify the direction is to maximize accuracy.
study = optuna.create_study(
    direction="maximize",
    pruner=pruner,
    storage=storage_name,
    study_name=study_name,
    load_if_exists=True,
)

# Start the optimization. Optuna will run the objective function 100 times.
with mlflow.start_run(run_name="hyperparameter-sweep") as parent_run:
    parent_run_id = parent_run.info.run_id
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="validation_accuracy",
        mlflow_kwargs={"nested": True, "parent_run_id": parent_run_id},
    )
    decorator_objective_fn = mlflow_callback.track_in_mlflow()
    new_objective_fn = decorator_objective_fn(objective_normal)
    study.optimize(new_objective_fn, n_trials=10, callbacks=[mlflow_callback])

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_accuracy_validation", study.best_value)

    # Log tags
    mlflow.set_tags(
        tags={
            "project": "Mean Teacher",
            "optimizer_engine": "optuna",
            "model_family": "pytorch",
            "feature_set_version": 1,
        }
    )

    artifact_path = "model"

# mlflow.pytorch.log_model(
#     pytorch_model=model,
#     name=artifact_path,
#     input_example=train_data[0],
#     model_format="ubj",
#     metadata={"model_data_version": 1},
# )

# model_uri = mlflow.get_artifac_uri(artifact_path)

