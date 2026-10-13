"""Compare les modèles évalués à l'horizon de 10 jours.

Les fichiers sont produits séparément par les pipelines R, Python classique
et Python LSTM/GRU. Ce script les rassemble sans réestimer de modèle.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def load_metrics(file_name: str, source: str) -> pd.DataFrame:
    """Charge un tableau de métriques et ajoute la source du résultat."""
    table = pd.read_csv(RESULTS_DIR / file_name)
    if "horizon" not in table:
        table["horizon"] = 10
    table.insert(0, "source", source)
    return table


classical_r = load_metrics(
    "metriques_horizon_10.csv",
    "R",
)
classical_python = load_metrics(
    "metriques_python_horizon_10.csv",
    "Python classique",
)
recurrent = load_metrics(
    "metriques_reseaux_horizon_10.csv",
    "Python réseaux",
)

comparison = pd.concat(
    [classical_r, classical_python, recurrent],
    ignore_index=True,
)
comparison = comparison.sort_values("RMSE").reset_index(drop=True)
comparison.insert(0, "rang_RMSE", comparison.index + 1)
comparison.to_csv(
    RESULTS_DIR / "comparaison_finale_horizon_10.csv",
    index=False,
)

print(comparison.to_string(index=False))

plot_data = comparison.sort_values("RMSE", ascending=True)
fig, axis = plt.subplots(figsize=(10, 5))
axis.barh(plot_data["modele"], plot_data["RMSE"])
axis.set_title("Comparaison des RMSE à l'horizon de 10 jours")
axis.set_xlabel("RMSE (MW)")
axis.invert_yaxis()
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "comparaison_rmse_horizon_10.png",
    dpi=150,
)
plt.close(fig)
