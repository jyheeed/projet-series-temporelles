# =============================================================================
# MINI-PROJET SERIES TEMPORELLES - SCRIPT R COMPLET
# Prevision de la consommation electrique journaliere francaise
# Realise par Zouaghi Jihed et Eya Gharam
# Encadre par Ahmed Dhouibi
#
# Ce fichier execute toute la partie R depuis le dataset journalier final :
# controles, exploration, stationnarite, SARIMA, SARIMAX, VAR, Granger,
# Johansen/VECM, ARCH/GARCH, previsions, metriques et traces.
# =============================================================================

rm(list = ls())
options(stringsAsFactors = FALSE, scipen = 999)

# 1. CHEMINS ET TRACE D'EXECUTION --------------------------------------------

dossier <- "C:/Users/pc/Desktop/Data/Processed/R_project"
dossier_data <- file.path(dossier, "data")
dossier_results <- file.path(dossier, "results")
dossier_diagnostics <- file.path(dossier, "diagnostics")
dossier_models <- file.path(dossier, "models")
dossier_figures <- file.path(dossier, "figures")

dir.create(dossier_results, showWarnings = FALSE, recursive = TRUE)
dir.create(dossier_diagnostics, showWarnings = FALSE, recursive = TRUE)
dir.create(dossier_models, showWarnings = FALSE, recursive = TRUE)
dir.create(dossier_figures, showWarnings = FALSE, recursive = TRUE)

fichier_trace <- file.path(dossier_results, "trace_execution_complete.txt")
sink(fichier_trace, split = TRUE)
sink(fichier_trace, type = "message", append = TRUE)

cat("===== TRACE DU SCRIPT R COMPLET =====\n")
cat("Date d'execution :", as.character(Sys.time()), "\n")
cat("Dossier :", dossier, "\n\n")

# 2. PACKAGES ---------------------------------------------------------------

packages_requis <- c(
  "readr", "dplyr", "tidyr", "forecast", "tseries",
  "vars", "urca", "FinTS", "rugarch"
)

packages_manquants <- packages_requis[
  !vapply(packages_requis, requireNamespace, logical(1), quietly = TRUE)
]
if (length(packages_manquants) > 0) {
  stop(
    "Packages manquants : ",
    paste(packages_manquants, collapse = ", "),
    ". Installez-les avant l'execution."
  )
}

library(readr)
library(dplyr)
library(tidyr)
library(forecast)
library(tseries)
library(vars)
library(urca)
library(FinTS)
library(rugarch)

cat("\n===== 1. CHARGEMENT ET CONTROLES =====\n")

# 3. CHARGEMENT ET CONTROLES DES DONNEES ------------------------------------

dataset_final <- readr::read_csv(
  file.path(dossier_data, "dataset_final_2015_2024.csv"),
  show_col_types = FALSE
) |>
  dplyr::arrange(date)

dataset_final$date <- as.Date(dataset_final$date)

colonnes_requises <- c(
  "date", "consommation_moyenne_mw", "temperature_france",
  "temperature_france_carre", "eolien_moyen", "jour_semaine"
)

if (!all(colonnes_requises %in% names(dataset_final))) {
  stop("Une ou plusieurs colonnes requises sont absentes du dataset.")
}
if (anyDuplicated(dataset_final$date) > 0) {
  stop("Le dataset contient des dates dupliquees.")
}
if (!isTRUE(all(diff(dataset_final$date) > 0))) {
  stop("Les dates ne sont pas strictement croissantes.")
}
if (anyNA(dataset_final[, colonnes_requises])) {
  stop("Des valeurs manquantes existent dans les variables modelisees.")
}

cat("Observations :", nrow(dataset_final), "\n")
cat("Colonnes :", ncol(dataset_final), "\n")
cat("Periode :", as.character(min(dataset_final$date)), "a",
    as.character(max(dataset_final$date)), "\n")
cat("Dates dupliquees :", anyDuplicated(dataset_final$date), "\n")
cat("Valeurs manquantes des variables modelisees : 0\n")
print(summary(dataset_final$consommation_moyenne_mw))

write_csv(
  tibble::tibble(
    controle = c("observations", "colonnes", "dates_dupliquees"),
    valeur = c(nrow(dataset_final), ncol(dataset_final),
               anyDuplicated(dataset_final$date))
  ),
  file.path(dossier_diagnostics, "controles_donnees_complets.csv")
)

# 4. ANALYSE EXPLORATOIRE ET FIGURES ----------------------------------------

cat("\n===== 2. ANALYSE EXPLORATOIRE =====\n")

serie_consommation <- dataset_final$consommation_moyenne_mw
serie_ts <- ts(serie_consommation, frequency = 7)

png(file.path(dossier_figures, "consommation_quotidienne_complete.png"),
    width = 1400, height = 700, res = 120)
plot(dataset_final$date, serie_consommation, type = "l",
     col = "steelblue", main = "Consommation electrique quotidienne",
     xlab = "Date", ylab = "Consommation moyenne (MW)")
dev.off()

png(file.path(dossier_figures, "temperature_quotidienne_complete.png"),
    width = 1400, height = 700, res = 120)
plot(dataset_final$date, dataset_final$temperature_france, type = "l",
     col = "darkorange", main = "Temperature moyenne quotidienne",
     xlab = "Date", ylab = "Temperature (degres C)")
dev.off()

png(file.path(dossier_figures, "relation_temperature_consommation_complete.png"),
    width = 900, height = 700, res = 120)
plot(dataset_final$temperature_france, serie_consommation,
     pch = 16, col = rgb(0.1, 0.3, 0.7, 0.35),
     main = "Relation temperature-consommation",
     xlab = "Temperature moyenne (degres C)", ylab = "Consommation (MW)")
abline(lm(serie_consommation ~ dataset_final$temperature_france),
       col = "red", lwd = 2)
dev.off()

png(file.path(dossier_figures, "acf_pacf_complets.png"),
    width = 1400, height = 900, res = 120)
par(mfrow = c(2, 2))
acf(serie_ts, lag.max = 60, main = "ACF - serie brute")
pacf(serie_ts, lag.max = 60, main = "PACF - serie brute")
acf(diff(serie_ts), lag.max = 60, main = "ACF - serie differenciee")
pacf(diff(serie_ts), lag.max = 60, main = "PACF - serie differenciee")
dev.off()
par(mfrow = c(1, 1))

decomposition_stl <- stl(serie_ts, s.window = "periodic")
write_csv(
  as.data.frame(decomposition_stl$time.series),
  file.path(dossier_diagnostics, "composantes_stl_complete.csv")
)
png(file.path(dossier_figures, "decomposition_stl_complete.png"),
    width = 1400, height = 900, res = 120)
plot(decomposition_stl, main = "Decomposition STL - saisonnalite hebdomadaire")
dev.off()

cat("Moyenne :", mean(serie_consommation), "\n")
cat("Ecart-type :", sd(serie_consommation), "\n")
cat("Minimum :", min(serie_consommation), "\n")
cat("Maximum :", max(serie_consommation), "\n")

# 5. STATIONNARITE -----------------------------------------------------------

cat("\n===== 3. TESTS DE STATIONNARITE =====\n")

test_stationnarite <- function(x, nom) {
  adf <- tseries::adf.test(as.numeric(x), alternative = "stationary")
  kpss <- tseries::kpss.test(as.numeric(x), null = "Level")
  tibble::tibble(
    serie = nom,
    adf_statistique = unname(adf$statistic),
    adf_p_value = adf$p.value,
    kpss_statistique = unname(kpss$statistic),
    kpss_p_value = kpss$p.value
  )
}

stationnarite <- dplyr::bind_rows(
  test_stationnarite(serie_ts, "Consommation brute"),
  test_stationnarite(diff(serie_ts), "Consommation differenciee")
)
print(stationnarite)
write_csv(stationnarite,
          file.path(dossier_diagnostics, "stationnarite_complete.csv"))

cat("ndiffs :", forecast::ndiffs(serie_ts), "\n")
cat("nsdiffs :", forecast::nsdiffs(serie_ts), "\n")

# 6. DECOUPAGE CHRONOLOGIQUE ET REGRESSEURS -------------------------------

cat("\n===== 4. DECOUPAGE TEMPOREL =====\n")

horizon <- 10L
n_total <- nrow(dataset_final)
n_train <- floor(0.90 * n_total)
indices_train <- seq_len(n_train)
indices_test <- (n_train + 1):n_total
indices_h10 <- indices_test[seq_len(horizon)]

y_train <- serie_consommation[indices_train]
y_h10 <- serie_consommation[indices_h10]
ts_train <- ts(y_train, frequency = 7)

cat("Taille train :", length(y_train), "\n")
cat("Debut test h=10 :", as.character(dataset_final$date[indices_h10[1]]), "\n")
cat("Fin test h=10 :", as.character(dataset_final$date[indices_h10[horizon]]), "\n")

jours <- factor(dataset_final$jour_semaine)
x_jours <- model.matrix(~ jours)[, -1, drop = FALSE]
colnames(x_jours) <- paste0("jour_", seq_len(ncol(x_jours)))

x_temperature <- cbind(
  temperature_france = dataset_final$temperature_france,
  temperature_france_carre = dataset_final$temperature_france^2
)
x_fourier <- forecast::fourier(ts(serie_consommation, frequency = 365.25), K = 5)
xreg <- as.matrix(cbind(x_temperature, x_jours, x_fourier))
storage.mode(xreg) <- "double"
xreg_train <- xreg[indices_train, , drop = FALSE]
xreg_h10 <- xreg[indices_h10, , drop = FALSE]

# 7. SARIMA, SARIMAX ET BASELINE --------------------------------------------

cat("\n===== 5. ESTIMATION SARIMA / SARIMAX / NAIF =====\n")

modele_sarima <- forecast::auto.arima(
  ts_train, seasonal = TRUE, stepwise = TRUE,
  approximation = TRUE, trace = TRUE
)
cat("\nModele SARIMA selectionne :\n")
print(modele_sarima)

modele_sarimax <- forecast::Arima(
  ts_train, xreg = xreg_train, seasonal = TRUE
)
cat("\nModele SARIMAX estime :\n")
print(modele_sarimax)

prevision_sarima <- forecast::forecast(modele_sarima, h = horizon)
prevision_naif <- forecast::snaive(ts_train, h = horizon)
prevision_sarimax <- forecast::forecast(
  modele_sarimax, xreg = xreg_h10, h = horizon
)

calculer_metriques <- function(observations, predictions, modele) {
  erreurs <- as.numeric(observations) - as.numeric(predictions)
  tibble::tibble(
    modele = modele,
    MSE = mean(erreurs^2),
    RMSE = sqrt(mean(erreurs^2)),
    MAE = mean(abs(erreurs)),
    MAPE = mean(abs(erreurs / observations)) * 100
  )
}

previsions_h10 <- tibble::tibble(
  horizon = seq_len(horizon),
  date = dataset_final$date[indices_h10],
  observation = y_h10,
  SARIMA = as.numeric(prevision_sarima$mean),
  naif_saisonnier = as.numeric(prevision_naif$mean),
  SARIMAX = as.numeric(prevision_sarimax$mean),
  SARIMA_lower_95 = as.numeric(prevision_sarima$lower[, "95%"]),
  SARIMA_upper_95 = as.numeric(prevision_sarima$upper[, "95%"]),
  SARIMAX_lower_95 = as.numeric(prevision_sarimax$lower[, "95%"]),
  SARIMAX_upper_95 = as.numeric(prevision_sarimax$upper[, "95%"])
)

metriques_h10 <- dplyr::bind_rows(
  calculer_metriques(y_h10, previsions_h10$SARIMA, "SARIMA automatique"),
  calculer_metriques(y_h10, previsions_h10$naif_saisonnier,
                     "Naif saisonnier hebdomadaire"),
  calculer_metriques(y_h10, previsions_h10$SARIMAX,
                     "SARIMAX temperature + Fourier")
)
print(metriques_h10)
write_csv(previsions_h10, file.path(dossier_results, "previsions_horizon_10_complete.csv"))
write_csv(metriques_h10, file.path(dossier_results, "metriques_horizon_10_complete.csv"))

png(file.path(dossier_figures, "previsions_modeles_complets.png"),
    width = 1400, height = 800, res = 120)
plot(previsions_h10$date, y_h10, type = "o", pch = 16,
     ylim = range(c(y_h10, previsions_h10$SARIMA,
                    previsions_h10$SARIMAX)),
     main = "Previsions a horizon 10 jours",
     xlab = "Date", ylab = "Consommation (MW)")
lines(previsions_h10$date, previsions_h10$SARIMA, type = "o", col = "red")
lines(previsions_h10$date, previsions_h10$SARIMAX, type = "o", col = "darkgreen")
legend("topleft", legend = c("Observation", "SARIMA", "SARIMAX"),
       col = c("black", "red", "darkgreen"), lty = 1, pch = 16)
dev.off()

# 8. DIAGNOSTICS SARIMAX ----------------------------------------------------

cat("\n===== 6. DIAGNOSTICS DES RESIDUS SARIMAX =====\n")

residus_sarimax <- as.numeric(residuals(modele_sarimax))
residus_sarimax <- residus_sarimax[is.finite(residus_sarimax)]
ljung <- Box.test(residus_sarimax, lag = 14, type = "Ljung-Box")
jb <- tseries::jarque.bera.test(residus_sarimax)
arch <- FinTS::ArchTest(residus_sarimax, lags = 7)

diagnostics_sarimax <- tibble::tibble(
  modele = "SARIMAX temperature + Fourier",
  ljung_box_statistique = unname(ljung$statistic),
  ljung_box_p_value = ljung$p.value,
  jarque_bera_statistique = unname(jb$statistic),
  jarque_bera_p_value = jb$p.value,
  arch_statistique = unname(arch$statistic),
  arch_p_value = arch$p.value
)
print(diagnostics_sarimax)
write_csv(diagnostics_sarimax,
          file.path(dossier_diagnostics, "diagnostics_sarimax_complete.csv"))
write_csv(tibble::tibble(residu = residus_sarimax),
          file.path(dossier_diagnostics, "residus_sarimax_complete.csv"))

png(file.path(dossier_figures, "acf_residus_sarimax_complete.png"),
    width = 1000, height = 700, res = 120)
acf(residus_sarimax, lag.max = 60, main = "ACF des residus SARIMAX")
dev.off()

# 9. VAR, STABILITE, RESIDUS, GRANGER ---------------------------------------

cat("\n===== 7. VAR ET DIAGNOSTICS MULTIVARIES =====\n")

variables_var <- dataset_final |>
  select(consommation_moyenne_mw, temperature_france, eolien_moyen) |>
  drop_na()

serie_var_niveaux <- ts(variables_var, frequency = 7)
serie_var_diff <- diff(serie_var_niveaux)
selection_var <- vars::VARselect(serie_var_diff, lag.max = 21, type = "const")
print(selection_var$selection)

ordre_var <- as.integer(selection_var$selection["AIC(n)"])
if (is.na(ordre_var) || ordre_var < 1) ordre_var <- 1L
modele_var <- vars::VAR(serie_var_diff, p = ordre_var, type = "const")
racines <- vars::roots(modele_var, modulus = TRUE)

diagnostic_var <- tibble::tibble(
  modele = paste0("VAR(", ordre_var, ")"),
  racine_maximale = max(racines),
  stable = max(racines) < 1
)
print(diagnostic_var)

test_serial_var <- vars::serial.test(
  modele_var, lags.pt = max(14, ordre_var + 7),
  type = "PT.asymptotic"
)
test_normalite_var <- vars::normality.test(modele_var)
test_arch_var <- vars::arch.test(modele_var, lags.multi = min(7, ordre_var))

diagnostics_var <- tibble::tibble(
  modele = paste0("VAR(", ordre_var, ")"),
  test = c("Portmanteau", "Normalite", "ARCH multivarie"),
  p_value = c(
    test_serial_var$serial$p.value,
    test_normalite_var$jb.mul$JB$p.value,
    test_arch_var$arch.mul$p.value
  )
)
print(diagnostics_var)

granger_temperature <- vars::causality(
  modele_var, cause = "temperature_france"
)
granger_eolien <- vars::causality(
  modele_var, cause = "eolien_moyen"
)

resultats_granger <- tibble::tibble(
  cause = c("temperature_france", "eolien_moyen"),
  statistique = c(
    unname(granger_temperature$Granger$statistic),
    unname(granger_eolien$Granger$statistic)
  ),
  p_value = c(
    granger_temperature$Granger$p.value,
    granger_eolien$Granger$p.value
  )
)
print(resultats_granger)

write_csv(diagnostic_var,
          file.path(dossier_diagnostics, "diagnostics_var_complete.csv"))
write_csv(diagnostics_var,
          file.path(dossier_diagnostics, "diagnostics_complets_var_complete.csv"))
write_csv(resultats_granger,
          file.path(dossier_diagnostics, "causalite_granger_var_complete.csv"))
saveRDS(modele_var, file.path(dossier_models, "modele_var_complete.rds"))

# 10. JOHANSEN ET VECM -------------------------------------------------------

cat("\n===== 8. JOHANSEN ET VECM =====\n")

selection_johansen <- vars::VARselect(
  serie_var_niveaux, lag.max = 14, type = "const"
)
ordre_johansen <- as.integer(selection_johansen$selection["AIC(n)"])
if (is.na(ordre_johansen) || ordre_johansen < 2) ordre_johansen <- 2L

test_johansen <- urca::ca.jo(
  serie_var_niveaux, type = "trace", ecdet = "const",
  K = ordre_johansen
)
rang_johansen <- sum(
  test_johansen@teststat > test_johansen@cval[, "5pct"]
)
rang_johansen <- min(rang_johansen, ncol(variables_var) - 1L)

resultat_johansen <- tibble::tibble(
  test = "Johansen trace",
  ordre_retard = ordre_johansen,
  rang_estime = rang_johansen
)
print(resultat_johansen)
write_csv(resultat_johansen,
          file.path(dossier_diagnostics, "cointégration_johansen_complete.csv"))
write_csv(
  tibble::tibble(
    hypothese = rownames(test_johansen@cval),
    statistique_trace = as.numeric(test_johansen@teststat),
    valeur_critique_5pct = as.numeric(test_johansen@cval[, "5pct"])
  ),
  file.path(dossier_diagnostics, "statistiques_trace_johansen_complete.csv")
)
saveRDS(test_johansen, file.path(dossier_models, "test_johansen_complete.rds"))

if (rang_johansen > 0) {
  modele_vecm <- vars::vec2var(test_johansen, r = rang_johansen)
  saveRDS(modele_vecm, file.path(dossier_models, "modele_vecm_complete.rds"))
  cat("VECM estime puis converti en VAR avec rang :", rang_johansen, "\n")
} else {
  cat("Aucune cointégration detectee au seuil de 5 pourcent.\n")
}

# 11. DIEBOLD-MARIANO --------------------------------------------------------

cat("\n===== 9. TESTS DE DIEBOLD-MARIANO =====\n")

erreurs_sarima <- y_h10 - previsions_h10$SARIMA
erreurs_naif <- y_h10 - previsions_h10$naif_saisonnier
erreurs_sarimax <- y_h10 - previsions_h10$SARIMAX

dm_sarima <- forecast::dm.test(erreurs_sarimax, erreurs_sarima,
                               h = 1, power = 2)
dm_naif <- forecast::dm.test(erreurs_sarimax, erreurs_naif,
                             h = 1, power = 2)
resultats_dm <- tibble::tibble(
  comparaison = c("SARIMAX vs SARIMA", "SARIMAX vs Naif saisonnier"),
  statistique_dm = c(unname(dm_sarima$statistic),
                     unname(dm_naif$statistic)),
  p_value = c(dm_sarima$p.value, dm_naif$p.value)
)
print(resultats_dm)
write_csv(resultats_dm,
          file.path(dossier_results, "tests_diebold_mariano_complete.csv"))

# 12. GARCH(1,1) SUR LES RESIDUS DU MODELE MOYEN -----------------------------

cat("\n===== 10. GARCH(1,1) =====\n")

specification_garch <- rugarch::ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "std"
)
modele_garch <- rugarch::ugarchfit(
  spec = specification_garch,
  data = residus_sarimax,
  solver = "hybrid"
)
print(modele_garch)

residus_standardises <- as.numeric(
  rugarch::residuals(modele_garch, standardize = TRUE)
)
volatilite <- as.numeric(rugarch::sigma(modele_garch))
test_arch_post <- FinTS::ArchTest(residus_standardises, lags = 7)
test_ljung_post <- Box.test(
  residus_standardises^2, lag = 7, type = "Ljung-Box"
)
diagnostics_post_garch <- tibble::tibble(
  modele = "GARCH(1,1)",
  arch_statistique = unname(test_arch_post$statistic),
  arch_p_value = test_arch_post$p.value,
  ljung_box_carres_statistique = unname(test_ljung_post$statistic),
  ljung_box_carres_p_value = test_ljung_post$p.value
)
print(diagnostics_post_garch)
write_csv(diagnostics_post_garch,
          file.path(dossier_diagnostics, "diagnostics_post_garch_complete.csv"))
write_csv(
  tibble::tibble(
    index = seq_along(residus_standardises),
    residu_standardise = residus_standardises,
    volatilite_conditionnelle = volatilite
  ),
  file.path(dossier_diagnostics, "diagnostics_residus_garch_complete.csv")
)
saveRDS(modele_garch, file.path(dossier_models, "modele_garch_complete.rds"))

png(file.path(dossier_figures, "volatilite_conditionnelle_garch_complete.png"),
    width = 1400, height = 700, res = 120)
plot(volatilite, type = "l", col = "steelblue",
     main = "Volatilite conditionnelle - GARCH(1,1)",
     xlab = "Observation", ylab = "Volatilite")
dev.off()

# 13. SAUVEGARDE DES MODELES ET FIN -----------------------------------------

cat("\n===== 11. SAUVEGARDE FINALE =====\n")
saveRDS(modele_sarima, file.path(dossier_models, "modele_sarima_complete.rds"))
saveRDS(modele_sarimax, file.path(dossier_models, "modele_sarimax_complete.rds"))

cat("Fichiers de resultats et diagnostics ecrits dans :\n")
cat(dossier_results, "\n")
cat(dossier_diagnostics, "\n")
cat(dossier_models, "\n")
cat(dossier_figures, "\n")
cat("\n===== EXECUTION TERMINEE SANS ERREUR =====\n")

sink(type = "message")
sink()
