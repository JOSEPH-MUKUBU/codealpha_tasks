# 🤖 CodeAlpha Machine Learning Projects

> **Auteur** : Joseph Mukubu Kapoya  
> **Organisation** : CodeAlpha Machine Learning Internship  
> **Période** : Juillet 2026  
> **Dépôt GitHub** : [codealpha_tasks](https://github.com/JOSEPH-MUKUBU/codealpha_tasks)

---

## 📋 Présentation Générale

Ce dépôt regroupe **3 projets complets de Machine Learning et de Deep Learning** réalisés dans le cadre du stage pratique chez **CodeAlpha**. Chaque projet aborde une problématique distincte de l'apprentissage supervisé, allant des modèles prédictifs tabulaires classiques aux réseaux neuronaux profonds de pointe pour l'analyse audio et la vision par ordinateur.

### 🌟 Les Projets inclus :
1.  **💳 Credit Scoring Model** : Modélisation et prédiction de la solvabilité bancaire.
2.  **🎵 Emotion Recognition from Speech** : Reconnaissance et classification des émotions dans la parole.
3.  **✍️ Handwritten Character & Letter Recognition** : Reconnaissance optique de chiffres (MNIST) et de lettres (EMNIST).

---

## 🗂️ Structure détaillée du Dépôt

Voici l'organisation des fichiers au sein du projet :

```text
CodeAlpha_ML_Projects/
│
├── 💳 CodeAlpha_CreditScoring/
│   ├── credit_scoring.py           # Script principal de modélisation
│   ├── credit_scoring_results.png  # Graphique d'évaluation (Courbes ROC)
│   └── README.md                   # Documentation dédiée au Credit Scoring
│
├── 🎵 CodeAlpha_EmotionRecognition/
│   ├── emotion_recognition.py      # Traitement audio et réseaux CNN/LSTM
│   ├── best_model.h5               # Poids optimaux sauvegardés
│   ├── emotion_recognition_results.png # Courbes d'entraînement et matrice de confusion
│   └── README.md                   # Documentation dédiée à la reconnaissance des émotions
│
├── ✍️ CodeAlpha_HandwrittenRecognition/
│   ├── handwritten_recognition.py  # Script de classification de chiffres (MNIST)
│   ├── emnist_letters.py           # Script de classification de lettres (EMNIST)
│   ├── best_model.h5               # Meilleur modèle MNIST sauvegardé
│   ├── emnist_best_model.h5        # Meilleur modèle EMNIST sauvegardé
│   ├── *predictions.png / *results.png # Graphiques d'analyse des prédictions
│   └── README.md                   # Documentation dédiée à la reconnaissance de caractères
│
├── .gitignore                      # Configuration Git pour exclure venv/ et caches
├── requirements.txt                # Dépendances logicielles du projet global
└── README.md                       # Présentation globale (ce fichier)
```

---

## 🛠️ Installation et Configuration

Pour exécuter ces projets localement, veuillez suivre les étapes suivantes :

### 1. Cloner le dépôt
```bash
git clone https://github.com/JOSEPH-MUKUBU/codealpha_tasks.git
cd codealpha_tasks
```

### 2. Configurer l'environnement virtuel Python
Il est vivement conseillé d'utiliser un environnement virtuel isolé pour éviter les conflits de dépendances.

```bash
# Création de l'environnement virtuel
python -m venv venv

# Activation - Windows (Command Prompt ou PowerShell)
venv\Scripts\activate

# Activation - macOS / Linux
source venv/bin/activate
```

### 3. Installer les dépendances
Installez l'ensemble des modules requis en une seule commande :
```bash
pip install -r requirements.txt
```

---

## ⚙️ Synthèse des Modèles, Datasets et Performances

Le tableau ci-dessous synthétise la méthodologie et les approches développées dans chacun des trois projets :

| Projet | Objectif principal | Dataset exploité | Modèles & Algorithmes | Métriques clés évaluées |
| :--- | :--- | :--- | :--- | :--- |
| **Credit Scoring** | Prédire le risque de défaut de paiement d'un emprunteur. | **German Credit Dataset** (UCI Repository) | Régression Logistique, Arbre de Décision, Random Forest. | Accuracy, Précision, Rappel, F1-Score, ROC-AUC. |
| **Emotion Recognition** | Détecter l'état émotionnel (8 classes) dans la voix humaine. | **RAVDESS** (Ryerson Emotional Speech Audio) | CNN 2D (sur spectrogramme MFCC), LSTM Bidirectionnel. | Classification Report, Matrice de Confusion, Val Accuracy. |
| **Handwritten Character** | Classer des caractères manuscrits (chiffres & lettres). | **MNIST** & **EMNIST** (Extended MNIST) | CNN 2D multi-couches (type VGG/LeNet) avec Data Augmentation. | Accuracy, Courbe de Perte, Visualisation des erreurs. |

---

## 🚀 Guide d'Exécution rapide

### 💳 1. Évaluation du Risque de Crédit
Exécutez le script pour comparer les algorithmes de classification classique :
```bash
cd CodeAlpha_CreditScoring
python credit_scoring.py
```
*Le script télécharge le dataset et génère la courbe ROC sous `credit_scoring_results.png`.*

### 🎵 2. Reconnaissance Vocale des Émotions
Pour entraîner les réseaux de neurones CNN et LSTM :
```bash
cd ../CodeAlpha_EmotionRecognition
# (Consultez le README spécifique pour télécharger le vrai dataset RAVDESS)
python emotion_recognition.py
```
*Génère le graphique d'entraînement et la matrice de confusion sous `emotion_recognition_results.png` ainsi que les fichiers `.h5` des modèles.*

### ✍️ 3. Reconnaissance d'Écritures Manuscrits
Pour entraîner les modèles CNN sur les chiffres (MNIST) ou les lettres (EMNIST) :
```bash
cd ../CodeAlpha_HandwrittenRecognition
# Pour classifier les chiffres 0-9 (MNIST)
python handwritten_recognition.py

# Pour classifier les lettres A-Z/a-z (EMNIST)
python emnist_letters.py
```
*Génère les visualisations d'exemples de prédictions et les courbes d'apprentissage associées.*

---

## 📞 Informations & Contact

*   **Organisation** : [CodeAlpha](https://www.codealpha.tech/)
*   **Auteur** : Joseph Mukubu Kapoya
*   **Email de l'Auteur** : joseph.mukubukapoya@polytechnicien.tn
*   **Lien du Stage** : Stage pratique en Machine Learning et Deep Learning.
