# Rapport Académique : Analyse Financière et Machine Learning

---

## Partie 1 — Rendements financiers et risques

### 1.1 Statistiques descriptives

**Objectif** : Résumer les rendements mensuels des portefeuilles et calculer rendement/risque annualisé.

**Étapes** :
1. Conversion des rendements en décimales.
2. Calcul moyenne, écart-type, médiane.
3. Rendement annualisé : \( R_{annuel} = (1+R_{mensuel})^{12} - 1 \)
4. Volatilité annualisée : \( \sigma_{annuel} = \sigma_{mensuel} \times \sqrt{12} \)

**Code Python :**
```python
import numpy as np

# Rendements en % convertis en décimal
rendements_A = np.array([...]) / 100
rendements_B = np.array([...]) / 100

# Fonctions calcul
moyenne_mensuelle = np.mean(rendements_A)
ecarts_mensuel = np.std(rendements_A, ddof=1)
mediane = np.median(rendements_A)
rendement_annuel = (np.prod(1+rendements_A)**(12/len(rendements_A)) - 1)
volatilite_annuelle = np.std(rendements_A, ddof=1)*np.sqrt(12)
```

### 1.2 Visualisation distributions

**Objectif** : Visualiser la répartition des rendements et identifier outliers.

**Code Python :**
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(rendements_A*100, bins=10, alpha=0.6, color='green', label='Portefeuille A')
plt.hist(rendements_B*100, bins=10, alpha=0.6, color='red', label='Portefeuille B')
plt.legend()
plt.show()

sns.boxplot(data=[rendements_A*100, rendements_B*100], palette=['green','red'])
plt.show()
```

### 1.3 Value at Risk (VaR 95%)

**Objectif** : Estimer la perte maximale à 95% de confiance.

**Formule** : \( VaR_{95\%} = \mu - 1.65 \cdot \sigma \)

**Code Python :**
```python
from scipy.stats import shapiro
capital = 500000
var_mensuelle = np.mean(rendements_A) - 1.65*np.std(rendements_A, ddof=1)
var_annuelle = rendement_annuel - 1.65*volatilite_annuelle
perte_annuelle = var_annuelle * capital
stat, p_value = shapiro(rendements_A)
```

### 1.4 Ratio Sharpe

**Formule** : \( Sharpe = \frac{R_{annuel} - r_f}{\sigma_{annuel}} \)

**Code Python :**
```python
rf = 0.03
sharpe_A = (rendement_annuel - rf) / volatilite_annuelle
```

---

## Partie 2 — Probabilités Bayésiennes et matrice de confusion

### 2.1 Bayes manuel

**Formule** :
\( P(Defaut|Retard) = \frac{P(Retard|Defaut) * P(Defaut)}{P(Retard|Defaut) * P(Defaut) + P(Retard|NoDefaut) * P(NoDefaut)} \)

**Code Python :**
```python
prior = 0.05
P_retard_defaut = 0.6
P_retard_no_defaut = 0.1
posterior = (P_retard_defaut*prior)/(P_retard_defaut*prior + P_retard_no_defaut*(1-prior))
```

### 2.2 Mise à jour séquentielle

**Nouvel événement** : découvert >500€, posterior précédente comme prior.

```python
P_decouvert_defaut = 0.5
P_decouvert_no_defaut = 0.05
posterior2 = (P_decouvert_defaut*posterior)/(P_decouvert_defaut*posterior + P_decouvert_no_defaut*(1-posterior))
```

### 2.3 Fonction générique Bayes

```python
def bayes_update(prior, likelihood_pos, likelihood_neg):
    return (likelihood_pos*prior)/(likelihood_pos*prior + likelihood_neg*(1-prior))
```

### 2.4 Matrice confusion

**Precision** = TP / (TP + FP), correspond à P(Defaut|Retard).  

```python
TP = 400
FP = 950
precision = TP / (TP+FP)
```

---

## Partie 3 — KNN pour détection défaut clients

### 3.1 Génération dataset

**Features** : âge, salaire, ancienneté, dettes, crédits, retards, score crédit.  
**Target** : défaut basé sur combinaison de risques.  

```python
import pandas as pd
np.random.seed(42)
df = pd.DataFrame({ ... })
df.describe()
df['defaut'].value_counts()
df.corr()['defaut'].sort_values()
```

**Visualisations** : heatmap + boxplots.

### 3.2 Preprocessing

- Séparation X/y, train/test split (70/30, stratify=y)  
- StandardScaler pour normaliser features  

```python
from sklearn.preprocessing import StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3.3 Recherche K optimal

- Boucle sur différentes valeurs de K  
- 5-fold cross-validation : AUC, Recall, Precision  
- Graphique AUC vs K  

```python
from sklearn.model_selection import cross_val_score
for K in K_list:
    knn = KNeighborsClassifier(n_neighbors=K)
    auc_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='roc_auc')
```

### 3.4 Entraînement modèle final

- KNN avec K optimal, prédiction test set  
- Calcul métriques : Accuracy, Precision, Recall, F1, AUC, Specificity  
- Matrice confusion et classification_report  

### 3.5 Courbe ROC et analyse seuil

- Trace ROC, calcul AUC  
- Seuil optimal via Youden index  
- Test seuils 0.3, 0.5, 0.7 pour Precision/Recall/F1  

### 3.6 Calcul ROI et recommandation

- Gains TP, coûts FP, pertes FN  
- Calcul ROI net pour différents seuils  
- Recommandation : seuil maximisant ROI tout en gardant Recall ≥ 80%.

