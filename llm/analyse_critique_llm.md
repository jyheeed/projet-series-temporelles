# Utilisation critique d'un LLM

Cette annexe documente l'utilisation du LLM conformément à la section 5 du
PDF. Le LLM est utilisé comme assistant d'analyse et non comme source des
résultats numériques. Toutes les valeurs numériques importantes sont vérifiées
dans les fichiers produits par R ou Python.

## 1. Statistiques transmises au LLM

La série étudiée est `consommation_moyenne_mw`, sur 3 653 jours entre 2015 et
2024.

| Statistique | Valeur |
|---|---:|
| Moyenne | 52 700,27 MW |
| Variance | 111 425 564,14 MW² |
| Écart-type | 10 555,83 MW |
| Skewness | 0,680 |
| Kurtosis excédentaire | -0,434 |
| Minimum | 35 589,17 MW |
| Maximum | 89 600,08 MW |
| ADF série brute | p = 0,01 |
| KPSS série brute | p = 0,01 |
| ADF après différenciation simple | p = 0,01 |
| KPSS après différenciation simple | p = 0,10 |

Les diagnostics du SARIMAX indiquent une autocorrélation résiduelle, une
non-normalité et un effet ARCH :

- Ljung-Box : p < 0,001 ;
- Jarque-Bera : p < 0,001 ;
- ARCH-LM : p = 5,8 × 10^-78.

Après le GARCH(1,1), les diagnostics de variance deviennent compatibles avec
une variance conditionnelle mieux modélisée :

- ARCH-LM : p = 0,276 ;
- Ljung-Box sur les résidus au carré : p = 0,243.

## 2. Prompt 1 — Hypothèses de modèles

### Prompt utilisé

> Voici les statistiques d'une série de consommation électrique journalière :
> 3 653 observations, moyenne 52 700,27 MW, écart-type 10 555,83 MW,
> skewness 0,680, kurtosis excédentaire -0,434. Le test ADF donne p = 0,01 et
> le test KPSS donne p = 0,01 sur la série brute. Après une différenciation
> simple, ADF donne p = 0,01 et KPSS p = 0,10. Les résidus d'un SARIMAX
> présentent un Ljung-Box significatif, un Jarque-Bera significatif et un
> test ARCH significatif. Propose des hypothèses de modèles à tester, sans
> inventer de résultats et en justifiant chaque proposition.

### Réponse du LLM

La série présente une forte variabilité et une asymétrie positive modérée. Les
tests ADF et KPSS sur la série brute ne donnent pas une conclusion parfaitement
simple, mais la différenciation simple rend les résultats plus compatibles avec
la stationnarité. Il est raisonnable de tester :

1. un modèle naïf saisonnier hebdomadaire comme référence ;
2. un ARIMA/SARIMA après différenciation, avec une saisonnalité hebdomadaire ;
3. un SARIMAX avec température, température au carré, variables calendaires et
   éventuellement des termes de Fourier annuels ;
4. un VAR sur la consommation, la température et une variable de production
   énergétique si les variables sont suffisamment liées ;
5. un GARCH(1,1) sur les résidus ou les écarts du modèle moyen si le test ARCH
   reste significatif.

Les ordres exacts doivent être déterminés à partir de l'ACF/PACF, d'un critère
d'information et des diagnostics de résidus. Les tests de causalité et de
cointégration ne doivent pas être interprétés comme des preuves de causalité
physique.

### Vérification par le projet

Cette réponse est cohérente avec les modèles effectivement utilisés :

- la baseline naïve hebdomadaire a bien été estimée ;
- SARIMA et SARIMAX ont été comparés ;
- le SARIMAX contient la température, son carré, les jours de la semaine et
  les termes de Fourier ;
- un VAR, Granger et Johansen/VECM ont été réalisés ;
- le test ARCH significatif a motivé le GARCH(1,1).

Le LLM ne donne pas d'ordre numérique inventé et renvoie correctement le choix
des ordres aux diagnostics. La conclusion sur la stationnarité reste prudente,
ce qui est nécessaire puisque les tests bruts ont des p-values faibles tous les
deux.

## 3. Prompt 2 — Explication pour un décideur

### Prompt utilisé

> Explique simplement les résultats suivants à un décideur non statisticien :
> le SARIMAX Python a un RMSE de 5 204,24 MW et un MAPE de 6,54 % à horizon
> 10 jours ; le SARIMAX R a un RMSE de 6 046,21 MW et un MAPE de 7,24 % ; le
> GRU a un RMSE de 7 211,18 MW et un MAPE de 7,50 % ; le LSTM a un RMSE de
> 7 747,64 MW et un MAPE de 8,35 %. Explique aussi les limites de cette
> comparaison.

### Réponse du LLM

Sur les dix jours étudiés, le SARIMAX Python est le modèle le plus précis
parmi les modèles comparés. Son erreur moyenne absolue en pourcentage est
d'environ 6,5 %, ce qui signifie que l'écart moyen relatif entre la prévision
et la consommation observée est de cet ordre sur cet horizon. Le SARIMAX R
arrive ensuite, puis le GRU et le LSTM.

Cela ne signifie pas que le SARIMAX sera toujours meilleur. La comparaison ne
porte que sur une fenêtre de dix jours et les implémentations R et Python ne
sont pas strictement identiques. Les résultats des réseaux dépendent aussi de
la fenêtre, de la normalisation, de l'initialisation et des hyperparamètres.
Il faut donc conserver une baseline, utiliser les mêmes observations et
interpréter les écarts avec prudence.

### Vérification par le projet

Les valeurs citées correspondent exactement au tableau
`results/comparaison_finale_horizon_10.csv`. L'explication est acceptable pour
un décideur, mais elle ne doit pas être transformée en promesse opérationnelle :
une erreur moyenne de 6,5 % ne garantit pas une erreur maximale de 6,5 % et ne
remplace pas une analyse des intervalles de prévision.

## 4. Prompt 3 — Recommandation de gestion énergétique

### Prompt utilisé

> À partir d'un modèle de prévision de consommation électrique à dix jours,
> propose des recommandations de gestion pour anticiper les pics de
> consommation. Ne présente pas les prévisions comme certaines. Énumère
> explicitement les risques, les informations manquantes et les précautions
> nécessaires.

### Réponse du LLM

Les prévisions peuvent aider à préparer les ressources de production et à
identifier les journées nécessitant une surveillance renforcée. Il est
possible de comparer chaque prévision à un seuil historique, de préparer des
scénarios de demande et de vérifier les prévisions chaque jour avec les
observations et les nouvelles données météorologiques.

Ces recommandations ne doivent pas être appliquées automatiquement. Les
prévisions dépendent de la qualité des données météo, des changements de
comportement, des événements exceptionnels, des indisponibilités de production
et des erreurs du modèle. Il faut donc utiliser des intervalles de prévision,
prévoir une marge de sécurité et faire valider toute décision par un
responsable du système électrique.

### Vérification par le projet

La réponse est une aide générale, pas une recommandation d'exploitation
réelle. Le projet ne contient pas les contraintes du réseau, les réserves
disponibles, les prix, les indisponibilités de centrales ni les événements
exceptionnels. Le LLM a correctement signalé ces limites ; il serait
incorrect de lui faire produire un ordre de production ou une décision
énergétique automatique.

## 5. Comparaison critique globale

### Convergences

- Le LLM identifie correctement la nécessité d'une baseline.
- Il relie la différenciation aux tests ADF/KPSS sans prétendre que les tests
  donnent exactement la même conclusion.
- Il justifie l'utilisation d'un modèle avec variables exogènes.
- Il reconnaît que le test de Granger est prédictif et non une causalité
  physique.
- Il recommande un modèle de variance après un test ARCH significatif.
- Il rappelle que dix erreurs donnent une comparaison statistiquement limitée.

### Limites et points corrigés par l'analyse experte

- Le LLM ne connaît pas les détails exacts de l'estimation R et Python s'ils ne
  lui sont pas transmis.
- Une proposition de modèle n'est pas une preuve de performance : les métriques
  doivent être calculées par le code.
- Les recommandations énergétiques restent générales et ne remplacent pas les
  contraintes d'exploitation.
- Il ne faut pas confondre MAPE moyen et garantie d'erreur individuelle.
- Il ne faut pas conclure que le GRU ou le LSTM est universellement meilleur :
  le classement dépend de l'horizon et de la période évaluée.

## 6. Conclusion sur l'usage du LLM

Le LLM a été utilisé pour formuler des hypothèses, vulgariser les résultats et
discuter des usages possibles. Il n'a pas servi à calculer les p-values, à
choisir seul les modèles ni à remplacer les diagnostics. Les résultats retenus
dans le projet sont ceux vérifiés par les scripts R/Python et les fichiers CSV.
