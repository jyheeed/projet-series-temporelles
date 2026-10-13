"""Interface Streamlit simple pour présenter les résultats du mini-projet."""

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


st.set_page_config(
    page_title="Prévision de la consommation électrique",
    page_icon="⚡",
    layout="wide",
)

st.title("Prévision de la consommation électrique française")
st.caption("Projet de séries temporelles — résultats R et Python")


@st.cache_data
def load_csv(file_name: str) -> pd.DataFrame:
    """Charge un résultat CSV et signale clairement un fichier absent."""
    path = RESULTS_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Fichier de résultats absent : {path}")
    return pd.read_csv(path)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Charge le dataset journalier final."""
    path = DATA_DIR / "dataset_final_2015_2024.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset absent : {path}")
    return pd.read_csv(path, parse_dates=["date"])


dataset = load_dataset()
comparison = load_csv("comparaison_finale_horizon_10.csv")
forecasts = load_csv("previsions_python_horizon_10.csv")
forecasts_r = load_csv("previsions_horizon_10.csv")
recurrent_forecasts = load_csv("previsions_reseaux_recurrents.csv")


def calculate_regression_metrics(
    observations: pd.Series,
    predictions: pd.Series,
) -> dict[str, float]:
    """Calcule les métriques adaptées à une prévision numérique."""
    errors = observations.to_numpy() - predictions.to_numpy()
    mse = float((errors**2).mean())
    mae = float(abs(errors).mean())
    mape = float((abs(errors / observations.to_numpy())).mean() * 100)
    total_variation = ((observations - observations.mean()) ** 2).sum()
    residual_variation = (observations - predictions) ** 2
    r2 = float(1 - residual_variation.sum() / total_variation)
    return {
        "MSE": mse,
        "RMSE": mse**0.5,
        "MAE": mae,
        "MAPE (%)": mape,
        "R²": r2,
    }


def metrics_from_forecasts(
    forecast_table: pd.DataFrame,
    prediction_columns: dict[str, str],
) -> pd.DataFrame:
    """Construit un tableau de métriques depuis les prévisions sauvegardées."""
    observations = forecast_table["observation"]
    rows = []
    for model_name, column_name in prediction_columns.items():
        row = {"Modèle": model_name}
        row.update(calculate_regression_metrics(observations, forecast_table[column_name]))
        rows.append(row)
    return pd.DataFrame(rows)


python_metrics = metrics_from_forecasts(
    forecasts,
    {
        "SARIMA Python": "SARIMA",
        "Naïf saisonnier Python": "naif_saisonnier",
        "SARIMAX Python": "SARIMAX",
    },
)
r_metrics = metrics_from_forecasts(
    forecasts_r,
    {
        "SARIMA R": "SARIMA",
        "Naïf saisonnier R": "naif_saisonnier",
        "SARIMAX R": "SARIMAX",
    },
)
recurrent_metrics = metrics_from_forecasts(
    recurrent_forecasts.head(10),
    {
        "LSTM": "LSTM",
        "GRU": "GRU",
    },
)
regression_metrics = pd.concat(
    [python_metrics, r_metrics, recurrent_metrics],
    ignore_index=True,
)

overview_tab, data_tab, forecast_tab, model_tab = st.tabs(
    ["Présentation", "Données", "Prévisions", "Comparaison des modèles"]
)

with overview_tab:
    st.header("Objectif")
    st.write(
        "Prévoir la consommation électrique journalière française à partir "
        "de son historique, de la température et de variables calendaires."
    )
    first_date = dataset["date"].min().strftime("%d/%m/%Y")
    last_date = dataset["date"].max().strftime("%d/%m/%Y")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Observations", f"{len(dataset):,}".replace(",", " "))
    metric_columns[1].metric("Période", f"{first_date} - {last_date}")
    metric_columns[2].metric("Horizon évalué", "10 jours")

    st.subheader("Démarche suivie")
    st.markdown(
        """
        1. Préparation des données RTE et météo.
        2. Analyse exploratoire et tests de stationnarité.
        3. Modèles classiques : naïf, SARIMA, SARIMAX et VAR.
        4. Diagnostics : résidus, Granger, Johansen et GARCH.
        5. Réseaux récurrents : LSTM et GRU.
        """
    )

with data_tab:
    st.header("Dataset journalier")
    st.line_chart(
        dataset.set_index("date")["consommation_moyenne_mw"],
        y_label="Consommation moyenne (MW)",
    )
    st.dataframe(
        dataset[
            [
                "date",
                "consommation_moyenne_mw",
                "temperature_france",
                "eolien_moyen",
                "jour_semaine",
            ]
        ].tail(20),
        use_container_width=True,
        hide_index=True,
    )

with forecast_tab:
    st.header("Prévisions à horizon 10 jours")
    st.info(
        "Les valeurs affichées sont les prévisions déjà produites par les "
        "scripts R et Python. L'application ne réentraîne pas les modèles."
    )
    st.dataframe(
        forecasts,
        use_container_width=True,
        hide_index=True,
    )
    st.line_chart(
        forecasts.set_index("date")[
            ["observation", "SARIMA", "naif_saisonnier", "SARIMAX"]
        ],
        y_label="Consommation (MW)",
    )

    st.subheader("Prévisions LSTM et GRU")
    st.dataframe(
        recurrent_forecasts.head(10),
        use_container_width=True,
        hide_index=True,
    )
    st.line_chart(
        recurrent_forecasts.set_index("date")[
            ["observation", "LSTM", "GRU"]
        ].head(10),
        y_label="Consommation (MW)",
    )

with model_tab:
    st.header("Comparaison des modèles")
    st.warning(
        "Ce projet est un problème de régression : l'accuracy et le "
        "F1-score sont des métriques de classification et ne s'appliquent "
        "pas directement à une consommation exprimée en MW."
    )
    st.subheader("Métriques de régression calculées sur les prévisions")
    st.dataframe(
        regression_metrics.style.format(
            {
                "MSE": "{:,.2f}",
                "RMSE": "{:,.2f}",
                "MAE": "{:,.2f}",
                "MAPE (%)": "{:.2f}",
                "R²": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.write("Classement selon le RMSE à l'horizon de 10 jours.")
    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(
        comparison.set_index("modele")["RMSE"],
        y_label="RMSE (MW)",
    )
    best_model = comparison.iloc[0]
    st.success(
        f"Meilleur modèle selon le RMSE : {best_model['modele']} "
        f"({best_model['RMSE']:,.2f} MW).".replace(",", " ")
    )

st.sidebar.header("Figures du projet")
figure_path = FIGURES_DIR / "courbes_perte_lstm_gru.png"
if figure_path.exists():
    st.sidebar.image(
        str(figure_path),
        caption="Courbes de perte LSTM et GRU",
    )
else:
    st.sidebar.info("La figure des courbes de perte n'est pas disponible.")
