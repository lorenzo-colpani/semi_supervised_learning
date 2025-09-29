import copy
from inspect import signature
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
from mlflow.optuna.storage import MlflowStorage
from mlflow.models import infer_signature

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


experiment_id = get_or_create_experiment("mean_teacher_iris_supervised")
mlflow.set_experiment(experiment_id=experiment_id)
mlflow_storage = MlflowStorage(experiment_id=experiment_id)

# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)

# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (
                abs(winner - study.best_value) / study.best_value
            ) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(
                f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}"
            )


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


# Apply the decorator to your objective function


def objective_normal(trial):
    with mlflow.start_run(
        nested=True,
        run_name=f"trial_{trial.number}",
    ):
        trial.set_user_attr("run_id", mlflow.active_run().info.run_id)
        params = {
            "batch_size_train": trial.suggest_categorical(
                "batch_size_train", [16, 32, 64, 128, 256, 512, 1024]
            ),
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "num_epochs": trial.suggest_int("num_epochs", 50, 500),
        }
        mlflow.log_params(params)
        mlflow.set_tag(
            "mlflow.note.content",
            f"Trial {trial.number}: Hyperparameter optimization run exploring different learning rates, batch sizes and epochs.",
        )
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        train_loader, val_loader, _, _ = load_data(params["batch_size_train"])
        input_size = 4
        output_size = 3
        model = SimpleNN(input_size, output_size)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        model.to(device)
        for epoch in range(params["num_epochs"]):
            # Training
            model.train()
            correct_train = 0
            total_train = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
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
                    val_inputs, val_labels = (
                        val_inputs.to(device),
                        val_labels.to(device),
                    )
                    val_outputs = model(val_inputs)
                    _, predicted_val = torch.max(val_outputs.data, 1)
                    total_val += val_labels.size(0)
                    correct_val += (predicted_val == val_labels).sum().item()
            validation_accuracy = correct_val / total_val
            trial.report(validation_accuracy, epoch)

            if trial.should_prune():
                mlflow.log_metric("train accuracy", train_accuracy)
                mlflow.log_metric("validation accuracy", validation_accuracy)
                mlflow.end_run(status="KILLED")
                raise optuna.TrialPruned()

        mlflow.log_metric("train accuracy", train_accuracy)
        mlflow.log_metric("validation accuracy", validation_accuracy)
        signature = infer_signature(
            inputs.detach().cpu().numpy(), outputs.detach().cpu().numpy()
        )
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            signature=signature,
        )

        return validation_accuracy


def champion_callback(study, trial):
    # Check if the trial is the new champion (best trial overall)
    # This check is safe because it's run *after* the trial is finished
    if trial.number == study.best_trial.number:
        print(f"Trial {trial.number} is the new champion with value: {trial.value}")

        # Get the run ID of the champion trial from its user attributes
        run_id = trial.user_attrs["run_id"]

        # Define the URI for the logged model artifact
        logged_model_uri = f"runs:/{run_id}/model"

        # Register this specific model artifact to the MLflow Model Registry
        mlflow.register_model(model_uri=logged_model_uri, name="SupervisedIrisModel")
        print("Champion model registered to the MLflow Model Registry.")


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
with mlflow.start_run(run_name=study_name):
    study.optimize(objective_normal, n_trials=100, callbacks=[champion_callback])
    mlflow.log_metric("best_accuracy_validation", study.best_value)
    mlflow.set_tags(
        tags={
            "model": "SupervisedIrisModel",
            "dataset": "Iris",
            "task": "classification",
            "framework": "pytorch",
            "training_approach": "supervised",
            "optimizer_engine": "optuna",
            "best_trial_number": study.best_trial.number,
            "best_run_id": study.best_trial.user_attrs["run_id"],
            "mlflow.note.content": "Hyperparameter optimization run exploring different learning rates, batch sizes and epochs for the label dataset.",
        }
    )

    # add params from best_trial number
    run = mlflow.get_run(study.best_trial.user_attrs["run_id"])
    mlflow.log_params(run.data.params)

    artifact_path = "model"

# mlflow.pytorch.log_model(
#     pytorch_model=model,
#     name=artifact_path,
#     input_example=train_data[0],
#     model_format="ubj",
#     metadata={"model_data_version": 1},
# )

# model_uri = mlflow.get_artifac_uri(artifact_path)
