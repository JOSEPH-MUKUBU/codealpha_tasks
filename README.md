# CodeAlpha Machine Learning Projects

> **Auteur** : Joseph Mukubu Kapoya  
> **Organisation** : CodeAlpha ML Internship  
> **Date** : Juillet 2026

---

## 📋 Présentation

Ce repository contient **3 projets complets de Machine Learning** réalisés dans le cadre du stage CodeAlpha :

1. **Credit Scoring Model** — Prédiction de la solvabilité bancaire
2. **Emotion Recognition from Speech** — Reconnaissance d'émotions dans la parole
3. **Handwritten Character Recognition** — Reconnaissance de chiffres manuscrits

---

## 🗂️ Structure du Repository

```
CodeAlpha_ML_Projects/
├── CodeAlpha_CreditScoring/
│   ├── credit_scoring.py
│   └── README.md
├── CodeAlpha_EmotionRecognition/
│   ├── emotion_recognition.py
│   └── README.md
├── CodeAlpha_HandwrittenRecognition/
│   ├── handwritten_recognition.py
│   └── README.md
├── requirements.txt
└── README.md (ce fichier)
```

---

## 🚀 Installation

### 1. Cloner le repository
```bash
git clone https://github.com/votre-username/CodeAlpha_ML_Projects.git
cd CodeAlpha_ML_Projects
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## 📦 Dépendances

| Package | Version | Utilisation |
|---------|---------|-------------|
| Python | ≥3.8 | Langage de base |
| NumPy | ≥1.21 | Calculs numériques |
| Pandas | ≥1.3 | Manipulation de données |
| Matplotlib | ≥3.4 | Visualisations |
| Seaborn | ≥0.11 | Visualisations avancées |
| Scikit-learn | ≥1.0 | Modèles classiques (LR, RF, DT) |
| TensorFlow | ≥2.8 | Deep Learning (CNN, LSTM) |
| Keras | ≥2.8 | API haut niveau pour TensorFlow |
| Librosa | ≥0.9 | Traitement audio (MFCC) |

---

## ▶️ Exécution des Projets

### Projet 1 : Credit Scoring
```bash
cd CodeAlpha_CreditScoring
python credit_scoring.py
```
- Dataset téléchargé automatiquement depuis UCI
- Résultats sauvegardés dans `credit_scoring_results.png`

### Projet 2 : Emotion Recognition
```bash
cd CodeAlpha_EmotionRecognition
# Télécharger d'abord le dataset RAVDESS (voir README du projet)
python emotion_recognition.py
```
- Dataset RAVDESS à télécharger manuellement
- Modèles sauvegardés : `model_cnn.h5`, `model_lstm.h5`

### Projet 3 : Handwritten Recognition
```bash
cd CodeAlpha_HandwrittenRecognition
python handwritten_recognition.py
```
- Dataset MNIST téléchargé automatiquement via Keras
- Modèle sauvegardé : `mnist_cnn_model.h5`

---

## 📊 Résumé des Modèles et Résultats Attendus

| Projet | Algorithmes | Métriques Clés | Dataset |
|--------|-------------|----------------|---------|
| Credit Scoring | Logistic Regression, Decision Tree, Random Forest | Accuracy, Precision, Recall, F1, ROC-AUC | German Credit (UCI) |
| Emotion Recognition | CNN, LSTM | Accuracy, Classification Report | RAVDESS |
| Handwritten Recognition | CNN (LeNet-style) | Accuracy, Confusion Matrix | MNIST |

---

## 📹 Soumission du Stage (CodeAlpha)

1. ✅ Publier sur LinkedIn en mentionnant **@CodeAlpha**
2. ✅ Uploader le code sur GitHub : repository nommé `CodeAlpha_ProjectName`
3. ✅ Poster une vidéo explicative sur LinkedIn avec le lien GitHub
4. ✅ Soumettre via le formulaire du groupe WhatsApp

---

## 📞 Contact

- **Website** : www.codealpha.tech
- **WhatsApp** : +91 93365 76683
- **Email** : services@codealpha.tech

---

## 📝 License

Ce projet est réalisé dans le cadre du stage CodeAlpha Machine Learning.
