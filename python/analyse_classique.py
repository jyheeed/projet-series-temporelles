"""Socle Python du mini-projet de séries temporelles.

Cette pipeline reproduit les étapes classiques demandées dans le PDF :
chargement, contrôle, découpage temporel, baseline, SARIMA, SARIMAX,
VAR, métriques et prévisions à horizon 10.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# 1. Chemins -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "dataset_final_2015_2024.csv"


# 2. Chargement et contrôles -------------------------------------------------

df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

required_columns = {
    "date",
    "consommation_moyenne_mw",
    "temperature_france",
    "temperature_france_carre",
    "eolien_moyen",
    "jour_semaine",
}

missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(f"Colonnes manquantes : {sorted(missing_columns)}")

if df["date"].duplicated().any():
    raise ValueError("Des dates dupliquées sont présentes.")

if not df["date"].is_monotonic_increasing:
    raise ValueError("Les dates ne sont pas dans l'ordre croissant.")

if df[list(required_columns)].isna().any().any():
    raise ValueError("Des valeurs manquantes sont présentes dans les variables utilisées.")


# 3. Découpage chronologique 90 % / 10 % ------------------------------------

target_series = df.set_index("date")["consommation_moyenne_mw"].astype(float)
target = target_series.to_numpy()
n_train = int(len(df) * 0.90)
horizon = 10

if len(df) - n_train < horizon:
    raise ValueError("La période test est trop courte pour h=10.")

y_train = target_series.iloc[:n_train]
y_test = target_series.iloc[n_train:]
observations_h10 = y_test[:horizon]


# 4. Fonctions communes ------------------------------------------------------

def metrics(observations: np.ndarray, predictions: np.ndarray, model: str) -> dict:
    """Calcule les quatre métriques imposées par le PDF."""
    errors = observations - predictions
    mse = float(np.mean(errors**2))
    return {
        "modele": model,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(observations, predictions)),
        "MAPE": float(np.mean(np.abs(errors / observations)) * 100),
    }


def annual_fourier(dates: pd.Series, harmonics: int = 5) -> pd.DataFrame:
    """Construit des termes de Fourier annuels sans utiliser le futur de la cible."""
    day_of_year = dates.dt.dayofyear.to_numpy(dtype=float)
    period = 365.25
    values = {}
    for k in range(1, harmonics + 1):
        values[f"S{k}-365"] = np.sin(2 * np.pi * k * day_of_year / period)
        values[f"C{k}-365"] = np.cos(2 * np.pi * k * day_of_year / period)
    return pd.DataFrame(values, index=dates.index)


def build_exog(data: pd.DataFrame) -> pd.DataFrame:
    """Prépare température, calendrier et Fourier pour le SARIMAX."""
    weekdays = pd.get_dummies(
        data["jour_semaine"],
        prefix="jour",
        drop_first=True,
        dtype=float,
    )
    temperature = pd.DataFrame(
        {
            "temperature_france": data["temperature_france"],
            "temperature_france_carre": data["temperature_france"] ** 2,
        },
        index=data.index,
    )
    return pd.concat(
        [temperature, weekdays, annual_fourier(data["date"])],
        axis=1,
    ).astype(float)


# 5. Baseline naïve saisonnière hebdomadaire -------------------------------

naive_h10 = np.resize(y_train[-7:], horizon)


# 6. SARIMA/SARIMAX ----------------------------------------------------------

# L'ordre est explicite et documenté ; il pourra être comparé à auto.arima en R.
sarima_model = SARIMAX(
    y_train,
    order=(2, 1, 2),
    seasonal_order=(0, 0, 0, 7),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

sarima_forecast = sarima_model.get_forecast(steps=horizon)
sarima_h10 = np.asarray(sarima_forecast.predicted_mean)

exog = build_exog(df)
exog.index = df["date"]
exog_train = exog.iloc[:n_train]
exog_test_h10 = exog.iloc[n_train : n_train + horizon]

sarimax_model = SARIMAX(
    y_train,
    exog=exog_train,
    order=(2, 1, 2),
    seasonal_order=(0, 0, 0, 7),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

sarimax_forecast = sarimax_model.get_forecast(
    steps=horizon,
    exog=exog_test_h10,
)
sarimax_h10 = np.asarray(sarimax_forecast.predicted_mean)


# 7. Comparaison des modèles -----------------------------------------------

metrics_h10 = pd.DataFrame(
    [
        metrics(observations_h10, sarima_h10, "SARIMA Python"),
        metrics(observations_h10, naive_h10, "Naïf saisonnier Python"),
        metrics(observations_h10, sarimax_h10, "SARIMAX Python"),
    ]
)

print(metrics_h10.to_string(index=False))
metrics_h10.to_csv(RESULTS_DIR / "metriques_python_horizon_10.csv", index=False)

forecasts_h10 = pd.DataFrame(
    {
        "horizon": np.arange(1, horizon + 1),
        "date": df["date"].iloc[n_train : n_train + horizon].to_numpy(),
        "observation": observations_h10,
        "SARIMA": sarima_h10,
        "naif_saisonnier": naive_h10,
        "SARIMAX": sarimax_h10,
        "SARIMA_lower_95": np.asarray(sarima_forecast.conf_int(alpha=0.05))[:, 0],
        "SARIMA_upper_95": np.asarray(sarima_forecast.conf_int(alpha=0.05))[:, 1],
        "SARIMAX_lower_95": np.asarray(sarimax_forecast.conf_int(alpha=0.05))[:, 0],
        "SARIMAX_upper_95": np.asarray(sarimax_forecast.conf_int(alpha=0.05))[:, 1],
    }
)
forecasts_h10.to_csv(RESULTS_DIR / "previsions_python_horizon_10.csv", index=False)


# 8. VAR classique -----------------------------------------------------------

var_data = df[
    ["consommation_moyenne_mw", "temperature_france", "eolien_moyen"]
].diff().dropna()

var_selection = VAR(var_data).select_order(maxlags=21)
selected_lag = var_selection.aic
if selected_lag is None or selected_lag < 1:
    selected_lag = 1

var_model = VAR(var_data).fit(selected_lag, trend="c")
print(f"Lag VAR sélectionné par AIC : {selected_lag}")

var_diagnostics = pd.DataFrame(
    [
        {
            "test": "Portmanteau",
            "p_value": float(var_model.test_whiteness(nlags=selected_lag + 7).pvalue),
        },
        {
            "test": "Normalité",
            "p_value": float(var_model.test_normality().pvalue),
        },
    ]
)
var_diagnostics.to_csv(
    RESULTS_DIR / "diagnostics_python_var.csv",
    index=False,
)

granger_rows = []
for cause in ["temperature_france", "eolien_moyen"]:
    result = var_model.test_causality(
        caused=[
            column
            for column in var_data.columns
            if column != cause
        ],
        causing=cause,
        kind="f",
    )
    granger_rows.append(
        {
            "cause": cause,
            "statistique": float(result.test_statistic),
            "p_value": float(result.pvalue),
        }
    )

pd.DataFrame(granger_rows).to_csv(
    RESULTS_DIR / "causalite_granger_python.csv",
    index=False,
)

coint_result = coint_johansen(
    df[
        ["consommation_moyenne_mw", "temperature_france", "eolien_moyen"]
    ].dropna(),
    det_order=0,
    k_ar_diff=max(selected_lag - 1, 1),
)
coint_result_table = pd.DataFrame(
    {
        "hypothese": [
            "r = 0",
            "r <= 1",
            "r <= 2",
        ],
        "statistique_trace": coint_result.lr1,
        "valeur_critique_5pct": coint_result.cvt[:, 1],
    }
)
coint_result_table.to_csv(
    RESULTS_DIR / "cointégration_johansen_python.csv",
    index=False,
)

var_forecast = var_model.forecast(
    var_data.to_numpy()[-selected_lag:],
    steps=horizon,
)
pd.DataFrame(
    var_forecast,
    columns=var_data.columns,
).to_csv(RESULTS_DIR / "previsions_python_var_horizon_10.csv", index=False)


# 9. Visualisation minimale de la comparaison --------------------------------

plt.figure(figsize=(11, 5))
plt.plot(observations_h10, label="Observations", marker="o")
plt.plot(sarima_h10, label="SARIMA", marker="o")
plt.plot(naive_h10, label="Naïf saisonnier", marker="o")
plt.plot(sarimax_h10, label="SARIMAX", marker="o")
plt.title("Prévisions Python à horizon 10 jours")
plt.xlabel("Horizon")
plt.ylabel("Consommation moyenne (MW)")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "comparaison_previsions_python_horizon_10.png", dpi=150)
plt.close()

print("Pipeline Python classique terminée.")
