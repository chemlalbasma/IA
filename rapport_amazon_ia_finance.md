# chemlal Basma
22006630
![basma](https://github.com/user-attachments/assets/f91ba6db-f165-4de2-b145-c0c8c31b26bf)

# kawtar Frimi 
22005989
![kawtar](https://github.com/user-attachments/assets/601973ea-21f0-41be-b830-6b63c693b1a1)



# 📊 Rapport : Comment Amazon Exploite l'IA dans le Domaine Financier

> **Section 1.2.6 — Cas Réel d'Entreprise : Amazon**
> *Système de Recommandation & Impact Financier*

---

## Table des Matières

1. [Contexte et Problématique](#1-contexte-et-problématique)
2. [Solution ML Déployée](#2-solution-ml-déployée)
3. [Architecture Technique](#3-architecture-technique)
4. [Résultats Financiers](#4-résultats-financiers)
5. [Amélioration Continue](#5-amélioration-continue)
6. [Analyse Critique](#6-analyse-critique)
7. [Conclusion](#7-conclusion)

---

## 1. Contexte et Problématique

Amazon vend plus de **350 millions de produits** sur sa plateforme. La question centrale est :

> *"Comment suggérer les bons produits à chaque client parmi cette immensité, de façon personnalisée et en temps réel ?"*

Sans IA, il serait **humainement impossible** de gérer cette personnalisation à l'échelle de centaines de millions d'utilisateurs simultanément.

---

## 2. Solution ML Déployée

### 2.1 Collecte de Données Massives

Pour chaque utilisateur, Amazon collecte en permanence :

| Type de donnée | Description | Valeur pour le modèle |
|---|---|---|
| **Historique d'achats** | Tous les produits achetés | ⭐⭐⭐⭐⭐ |
| **Produits consultés** | Pages visitées | ⭐⭐⭐⭐⭐ |
| **Temps sur page** | Durée de consultation | ⭐⭐⭐⭐ |
| **Paniers abandonnés** | Produits ajoutés puis retirés | ⭐⭐⭐⭐ |
| **Recherches effectuées** | Mots-clés tapés | ⭐⭐⭐⭐ |
| **Avis laissés** | Notes et commentaires | ⭐⭐⭐ |

### 2.2 Algorithmes ML Utilisés

#### a) Filtrage Collaboratif (Item-to-Item)

C'est l'algorithme phare du système de recommandation d'Amazon. Il calcule la **similarité entre produits** en analysant les comportements d'achat croisés.

```python
# Pseudo-code simplifié
# Si vous achetez un livre de science-fiction :

similarities = {}
for other_item in all_items:
    users_who_bought_both = count_users(book, other_item)
    similarity_score = cosine_similarity(book, other_item)
    similarities[other_item] = similarity_score

# Recommander les items avec les scores les plus élevés
top_recommendations = sorted(similarities)[:10]
```

#### b) Analyse des Patterns d'Achat

Le modèle découvre automatiquement des **associations statistiques** entre produits :

```
Pattern découvert :
Achat(Nintendo Switch) → forte probabilité d'acheter :
├── Jeux Switch            (95%)
├── Manette supplémentaire (70%)
├── Pochette de transport  (60%)
└── Carte SD               (55%)

→ Afficher ces produits dans "Fréquemment achetés ensemble"
```

---

## 3. Architecture Technique

```
┌──────────────────────────────┐
│       DONNÉES CLIENT         │
│  Historique + Comportement   │
│         temps réel           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│        MODÈLES ML (multi-algorithmes)    │
├──────────────────────────────────────────┤
│  • Collaborative Filtering               │
│  • Deep Learning (embeddings)            │
│  • Association Rules Mining              │
│  • Sequential Pattern Mining             │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│    PERSONNALISATION          │
│       EN TEMPS RÉEL          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   RECOMMANDATIONS AFFICHÉES  │
│    (< 100ms de latence)      │
└──────────────────────────────┘
```

### Détail des algorithmes

| Algorithme | Rôle | Performance |
|---|---|---|
| **Collaborative Filtering** | Similarité entre utilisateurs et produits | Très haute précision |
| **Deep Learning (embeddings)** | Représentation vectorielle des produits | Capture les relations complexes |
| **Association Rules Mining** | Découverte de patterns d'achat | "Fréquemment achetés ensemble" |
| **Sequential Pattern Mining** | Analyse de la séquence d'achats dans le temps | Anticipation des besoins futurs |

---

## 4. Résultats Financiers

### 4.1 Chiffres Clés

| Indicateur | Résultat | Impact |
|---|---|---|
| **Part du CA issue des recommandations** | **35%** du chiffre d'affaires total | 💰 Plusieurs dizaines de milliards $ |
| **Augmentation du panier moyen** | **+29%** | 💰 Revenu par client augmenté |
| **Taux de conversion** | **+15%** pour les cliqueurs | 💰 Plus de ventes conclues |
| **Revenus additionnels estimés** | **Plusieurs milliards $/an** | 💰 ROI exceptionnel |

### 4.2 Traduction Financière Concrète

Avec un chiffre d'affaires annuel d'Amazon dépassant **500 milliards de dollars** :

```
35% du CA provenant des recommandations IA
        ↓
≈ 175 milliards de dollars générés par l'IA
        ↓
Sans IA → perte potentielle de 175 Mds$/an
```

> **Conclusion financière :** L'IA de recommandation est le **premier moteur de revenus** d'Amazon, devant la publicité, AWS et la logistique.

---

## 5. Amélioration Continue

Amazon utilise en permanence une stratégie d'**A/B Testing** couplée à du **réentraînement continu** :

```
Version A de l'algorithme  ←→  Version B de l'algorithme
          ↓                              ↓
     Groupe test 1                  Groupe test 2
          ↓                              ↓
    Mesure des KPIs (clics, conversions, revenus)
          ↓
    Déploiement automatique de la meilleure variante
          ↓
    Réentraînement quotidien avec nouvelles données
```

### KPIs surveillés en temps réel

| KPI | Description |
|---|---|
| **CTR** (Click-Through Rate) | Taux de clic sur les recommandations |
| **CVR** (Conversion Rate) | % de clics convertis en achat |
| **AOV** (Average Order Value) | Valeur moyenne du panier |
| **Revenue per session** | Revenu généré par visite |

---

## 6. Analyse Critique

### ✅ Points Forts

- **Scalabilité exceptionnelle** : fonctionne pour des centaines de millions d'utilisateurs simultanément
- **Personnalisation poussée** : chaque utilisateur a une expérience unique
- **Temps réel** : latence < 100ms, invisible pour l'utilisateur
- **ROI démontré** : 35% du CA est un résultat concret et mesurable

### ⚠️ Limites et Risques

| Limite | Description |
|---|---|
| **Bulle de filtre** | L'utilisateur ne voit que ce qu'il connaît déjà |
| **Biais de popularité** | Les produits populaires sont sur-recommandés |
| **Cold start** | Difficulté à recommander pour les nouveaux utilisateurs |
| **Vie privée** | Collecte massive de données personnelles |
| **Manipulation** | Risque de pousser des produits à marge élevée plutôt que pertinents |

---

## 7. Conclusion

Le cas Amazon illustre parfaitement comment l'IA transforme une contrainte (trop de produits à gérer) en **avantage compétitif décisif**.

> Le système de recommandation d'Amazon n'est pas un simple "plus" technologique. C'est **le cœur du modèle économique**, générant plus d'un tiers du chiffre d'affaires total de l'entreprise.

### La formule du succès Amazon

```
Données massives + Algorithmes ML avancés + Amélioration continue
                          =
        35% du CA | +29% panier moyen | +15% conversion
```

Ce modèle est aujourd'hui **copié par toutes les plateformes e-commerce**, de Netflix (recommandation de films) à Spotify (recommandation musicale) en passant par TikTok (recommandation de vidéos).

---

*Rapport rédigé dans le cadre du cours sur le Machine Learning appliqué aux entreprises*
*Section 1.2.6 — Cas Réel d'Entreprise : Amazon*

---

## 📚 Sources et Références

### Articles Scientifiques et Recherches Académiques

| # | Référence | Lien |
|---|---|---|
| [1] | **Linden, G., Smith, B. & York, J.** (2003). *"Amazon.com Recommendations: Item-to-Item Collaborative Filtering"*. IEEE Internet Computing, vol. 7, no. 1, pp. 76–80. | [IEEE Xplore](https://ieeexplore.ieee.org/document/1167344/) |
| [2] | **Smith, B. & Linden, G.** (2017). *"Two Decades of Recommender Systems at Amazon.com"*. IEEE Internet Computing. | [Amazon Science](https://assets.amazon.science/76/9e/7eac89c14a838746e91dde0a5e9f/two-decades-of-recommender-systems-at-amazon.pdf) |
| [3] | **Amazon Science** (2025). *"The History of Amazon's Recommendation Algorithm"*. | [amazon.science](https://www.amazon.science/the-history-of-amazons-recommendation-algorithm) |

### Sources Officielles Amazon

| # | Référence | Lien |
|---|---|---|
| [4] | **Amazon** (2024). *"Amazon's Gen AI Personalizes Product Recommendations"*. About Amazon. | [aboutamazon.com](https://www.aboutamazon.com/news/retail/amazon-generative-ai-product-search-results-and-descriptions) |

### Rapports et Analyses Business

| # | Référence | Lien |
|---|---|---|
| [5] | **Head of AI** (2025). *"How AI Helps Generate 35% of Amazon's Annual Revenue"*. | [headofai.ai](https://headofai.ai/how-ai-helps-generate-35-of-amazons-annual-revenue-200bn/) |
| [6] | **AgentiveAI** (2025). *"How Amazon Uses AI for Smarter Product Recommendations"*. | [agentiveaiq.com](https://agentiveaiq.com/blog/how-amazon-uses-ai-for-smarter-product-recommendations) |
| [7] | **Stratoflow** (2025). *"Amazon Product Recommendation System: How Does Amazon Algorithm Work?"* | [stratoflow.com](https://stratoflow.com/amazon-recommendation-system/) |
| [8] | **Medium / Qadir Ansah-Smith** (2025). *"Building Better Product Experiences with AI: How Netflix and Amazon Mastered Recommender Systems"*. | [medium.com](https://medium.com/@qadir.ansahsmith/inside-the-ai-engines-of-netflix-amazon-what-recommender-systems-reveal-about-product-strategy-0149be339ff5) |
| [9] | **Lineate** (2023). *"3 Ways Amazon Uses AI to Make Product Recommendations"*. | [lineate.com](https://www.lineate.com/blog/3-ways-amazon-uses-ai-to-make-product-recommendations) |
| [10] | **VWO Blog** (2025). *"How Does Amazon & Netflix Personalization Work?"* | [vwo.com](https://vwo.com/blog/deliver-personalized-recommendations-the-amazon-netflix-way/) |

### Données Statistiques

| Donnée | Source |
|---|---|
| **35% du CA généré par les recommandations** | McKinsey & Company — cité par Stratoflow [7] et confirmé par multiple sources [5][6][8] |
| **+29% panier moyen** | Head of AI [5] / AgentiveAI [6] |
| **+15% taux de conversion** | Industry analysis — AgentiveAI [6] |
| **26% du CA e-commerce mondial influencé par l'IA** | Salesforce — cité par AgentiveAI [6] |
| **56% de clients fidélisés** | VWO Blog [10] |

---

> 📌 **Note méthodologique :** Les chiffres financiers (35% du CA, +29% panier moyen) sont issus d'analyses industrielles et de rapports McKinsey largement cités dans la littérature académique et business. Amazon ne publie pas officiellement ces données de manière isolée dans ses rapports annuels.
