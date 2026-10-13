"""LSTM et GRU pour la prévision de la consommation électrique.

La validation respecte l'ordre temporel :
train -> validation -> test.
Le scaler est ajusté uniquement sur le train pour éviter une fuite de données.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch import nn


# 1. Reproductibilité et chemins --------------------------------------------

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

DATASET_PATH = DATA_DIR / "dataset_final_2015_2024.csv"

for directory in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    directory.mkdir(exist_ok=True)


# 2. Chargement et variables -----------------------------------------------

df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

weekday = df["date"].dt.dayofweek.to_numpy()
features = pd.DataFrame(
    {
        "consommation": df["consommation_moyenne_mw"],
        "temperature": df["temperature_france"],
        "temperature_carre": df["temperature_france"] ** 2,
        "eolien": df["eolien_moyen"],
        "sin_semaine": np.sin(2 * np.pi * weekday / 7),
        "cos_semaine": np.cos(2 * np.pi * weekday / 7),
        "sin_annee": np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25),
        "cos_annee": np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25),
    }
)

values = features.to_numpy(dtype=np.float32)
n_total = len(values)
n_train = int(n_total * 0.80)
n_validation = int(n_total * 0.10)
test_start = n_train + n_validation
window_size = 14


# 3. Normalisation sans fuite d'information --------------------------------

scaler = StandardScaler()
scaler.fit(values[:n_train])
values_scaled = scaler.transform(values).astype(np.float32)


def make_windows(
    scaled_values: np.ndarray,
    target_start: int,
    target_end: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit les fenêtres et conserve l'indice temporel de chaque cible."""
    x_values = []
    y_values = []
    target_indices = []
    first_target = max(window, target_start)

    for index in range(first_target, target_end):
        x_values.append(scaled_values[index - window : index])
        y_values.append(scaled_values[index, 0])
        target_indices.append(index)

    if not x_values:
        raise ValueError("Aucune fenêtre disponible pour la période demandée.")

    return (
        np.asarray(x_values, dtype=np.float32),
        np.asarray(y_values, dtype=np.float32),
        np.asarray(target_indices, dtype=np.int64),
    )


x_train, y_train, _ = make_windows(
    values_scaled,
    target_start=window_size,
    target_end=n_train,
    window=window_size,
)
x_validation, y_validation, _ = make_windows(
    values_scaled,
    target_start=n_train,
    target_end=test_start,
    window=window_size,
)
x_test, y_test, test_indices = make_windows(
    values_scaled,
    target_start=test_start,
    target_end=n_total,
    window=window_size,
)


def tensors(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(x_values),
        torch.from_numpy(y_values).reshape(-1, 1),
    )


x_train_tensor, y_train_tensor = tensors(x_train, y_train)
x_validation_tensor, y_validation_tensor = tensors(
    x_validation,
    y_validation,
)
x_test_tensor, _ = tensors(x_test, y_test)


# 4. Architectures récurrentes ----------------------------------------------


class RecurrentForecaster(nn.Module):
    """Architecture commune avec LSTM ou GRU selon le type demandé."""

    def __init__(
        self,
        recurrent_type: str,
        input_size: int,
        hidden_size: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        recurrent_layer = nn.LSTM if recurrent_type == "LSTM" else nn.GRU
        self.recurrent = recurrent_layer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        recurrent_output, _ = self.recurrent(inputs)
        last_output = recurrent_output[:, -1, :]
        return self.output(self.dropout(last_output))


def inverse_target(scaled_target: np.ndarray, fitted_scaler: StandardScaler) -> np.ndarray:
    """Revient à l'échelle MW en utilisant les paramètres du train."""
    return (
        scaled_target * fitted_scaler.scale_[0]
        + fitted_scaler.mean_[0]
    )


def train_model(
    model_name: str,
    epochs: int = 60,
    patience: int = 8,
) -> tuple[nn.Module, pd.DataFrame]:
    """Entraîne un réseau avec arrêt anticipé sur la validation temporelle."""
    model = RecurrentForecaster(
        recurrent_type=model_name,
        input_size=values_scaled.shape[1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    history = []
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_prediction = model(x_train_tensor)
        train_loss = loss_function(train_prediction, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(x_validation_tensor)
            validation_loss = loss_function(
                validation_prediction,
                y_validation_tensor,
            )

        train_loss_value = float(train_loss.item())
        validation_loss_value = float(validation_loss.item())
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_value,
                "validation_loss": validation_loss_value,
            }
        )

        if validation_loss_value < best_loss:
            best_loss = validation_loss_value
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"Aucun état optimal obtenu pour {model_name}.")

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def evaluate_model(
    model_name: str,
    model: nn.Module,
) -> tuple[dict, np.ndarray]:
    """Évalue un modèle sur tout le test chronologique."""
    model.eval()
    with torch.no_grad():
        scaled_predictions = model(x_test_tensor).numpy().ravel()

    predictions = inverse_target(scaled_predictions, scaler)
    observations = df["consommation_moyenne_mw"].to_numpy()[test_indices]
    errors = observations - predictions
    mse = float(np.mean(errors**2))
    result = {
        "modele": model_name,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(observations, predictions)),
        "MAPE": float(np.mean(np.abs(errors / observations)) * 100),
    }
    return result, predictions


# 5. Entraînement et sauvegarde ---------------------------------------------

all_metrics = []
all_predictions = {}
all_histories = {}

for model_name in ("LSTM", "GRU"):
    model, history = train_model(model_name)
    metrics, predictions = evaluate_model(model_name, model)
    all_metrics.append(metrics)
    all_predictions[model_name] = predictions
    all_histories[model_name] = history
    history.to_csv(
        RESULTS_DIR / f"courbes_perte_{model_name.lower()}.csv",
        index=False,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "window_size": window_size,
            "feature_names": list(features.columns),
            "seed": SEED,
        },
        MODELS_DIR / f"modele_{model_name.lower()}.pt",
    )

metrics_table = pd.DataFrame(all_metrics)
metrics_table.to_csv(
    RESULTS_DIR / "metriques_reseaux_recurrents.csv",
    index=False,
)
print(metrics_table.to_string(index=False))


# 6. Prévisions test et courbes de perte ------------------------------------

predictions_table = pd.DataFrame(
    {
        "date": df["date"].to_numpy()[test_indices],
        "observation": df["consommation_moyenne_mw"].to_numpy()[test_indices],
        "LSTM": all_predictions["LSTM"],
        "GRU": all_predictions["GRU"],
    }
)
predictions_table.to_csv(
    RESULTS_DIR / "previsions_reseaux_recurrents.csv",
    index=False,
)

horizon = 10
horizon_metrics = []
for model_name in ("LSTM", "GRU"):
    observations = predictions_table["observation"].to_numpy()[:horizon]
    predictions = predictions_table[model_name].to_numpy()[:horizon]
    errors = observations - predictions
    mse = float(np.mean(errors**2))
    horizon_metrics.append(
        {
            "modele": model_name,
            "horizon": min(horizon, len(observations)),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "MAE": float(mean_absolute_error(observations, predictions)),
            "MAPE": float(np.mean(np.abs(errors / observations)) * 100),
        }
    )
pd.DataFrame(horizon_metrics).to_csv(
    RESULTS_DIR / "metriques_reseaux_horizon_10.csv",
    index=False,
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for axis, model_name in zip(axes, ("LSTM", "GRU")):
    history = all_histories[model_name]
    axis.plot(history["epoch"], history["train_loss"], label="train")
    axis.plot(
        history["epoch"],
        history["validation_loss"],
        label="validation",
    )
    axis.set_title(f"Perte {model_name}")
    axis.set_xlabel("Époque")
    axis.set_ylabel("MSE normalisée")
    axis.legend()
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "courbes_perte_lstm_gru.png",
    dpi=150,
)
plt.close(fig)

print("Pipeline LSTM/GRU terminée.")
