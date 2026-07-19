# 🎵 Emotion Recognition from Speech — CodeAlpha

Ce projet implémente un système de reconnaissance d'émotions à partir de la parole (Speech Emotion Recognition - SER) en utilisant des réseaux de neurones profonds. Il compare l'efficacité d'un Réseau de Neurones Convolutif (CNN 2D) et d'un Réseau de Neurones Récurrents (LSTM Bidirectionnel) sur le jeu de données RAVDESS.

---

## 📋 Table des matières
1. [Présentation du projet](#-présentation-du-projet)
2. [Dataset (RAVDESS)](#-dataset-ravdess)
3. [Extraction des Features (MFCC)](#-extraction-des-features-mfcc)
4. [Architecture des Modèles](#-architecture-des-modèles)
5. [Installation et Utilisation](#-installation-et-utilisation)
6. [Résultats et Visualisations](#-résultats-et-visualisations)

---

## 📋 Présentation du projet
La reconnaissance des émotions vocales est un domaine complexe à l'intersection du traitement du signal audio et du Deep Learning. Ce projet vise à classifier des signaux audio de parole humaine selon 8 émotions distinctes. Il explore deux approches architecturales majeures en Deep Learning pour traiter des données séquentielles/spectrales : les CNN 2D (convolutions spatiales sur spectrogrammes) et les LSTM (réseaux récurrents adaptés aux séries temporelles).

---

## 📂 Dataset (RAVDESS)
Le projet exploite le dataset de référence **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song).
*   **Contenu :** Fichiers audio au format `.wav` enregistrés par 24 acteurs professionnels (12 hommes, 12 femmes) vocalisant des phrases avec des expressions émotionnelles précises.
*   **Émotions cibles (8 classes) :**
    1.  `neutral` (Neutre)
    2.  `calm` (Calme)
    3.  `happy` (Joyeux)
    4.  `sad` (Triste)
    5.  `angry` (Colère)
    6.  `fearful` (Peur)
    7.  `disgust` (Dégoût)
    8.  `surprised` (Surpris)

*Note : Si les fichiers réels du dataset RAVDESS ne sont pas présents dans le dossier `ravdess_data/`, le script bascule automatiquement en mode démonstration et génère des signaux audio synthétiques modélisés pour l'apprentissage.*

### Comment télécharger le vrai dataset :
Pour tester avec les vraies voix :
1. Téléchargez le pack audio : **"Audio_Speech_Actors_01-24"** depuis [Kaggle (RAVDESS Dataset)](https://www.kaggle.com/datasets/uwrgrizzly/ravdess-emotional-speech-audio).
2. Créez un répertoire `ravdess_data/` à la racine de ce dossier.
3. Extrayez les 24 répertoires d'acteurs (`Actor_01` à `Actor_24`) directement dans `ravdess_data/`.

---

## 🔧 Extraction des Features (MFCC)
Le signal audio brut est complexe. Pour l'apprentissage automatique, nous le convertissons en représentation temps-fréquence :
*   **MFCC (Mel-Frequency Cepstral Coefficients) :** Extraction de 40 coefficients MFCC décrivant l'enveloppe spectrale du signal.
*   **Deltas & Delta-Deltas :** Calcul des dérivées premières et secondes des MFCC pour capturer la dynamique temporelle (vitesse et accélération des changements de fréquences).
*   **Dimension finale :** Pour chaque fichier audio, nous obtenons une matrice de forme `(120, 130)` (120 canaux de features x 130 pas temporels), représentant une durée fixe de 3 secondes à un taux d'échantillonnage de 22050 Hz.
*   **Data Augmentation :** Ajout d'un bruit blanc Gaussien aléatoire sur les formes d'onde audio d'entraînement pour accroître la robustesse des modèles.

---

## 🤖 Architecture des Modèles

Le script construit et compare deux architectures :

### 1. CNN 2D (Convolutional Neural Network)
Ce modèle traite la matrice de features `(120, 130, 1)` comme une image à un canal.
*   **Structure :**
    *   3 blocs de Convolution (2D Conv) + Batch Normalization + Max Pooling.
    *   Dropout régulier (25% à 50%) pour limiter le surapprentissage.
    *   Couche fully-connected (Dense 256) suivie d'une couche Softmax à 8 sorties.
*   **Avantage :** Excellente capacité à capter les motifs fréquentiels locaux et les structures dans le spectrogramme.

### 2. LSTM Bidirectionnel (Long Short-Term Memory)
Ce modèle traite le signal comme une série temporelle pure de dimension `(130, 120)` (130 étapes temporelles avec 120 variables à chaque étape).
*   **Structure :**
    *   2 couches de LSTM Bidirectionnel (BiLSTM) de 128 et 64 unités respectivement.
    *   Mécanisme de régularisation (Batch Normalization et Dropout).
    *   Couches denses de classification.
*   **Avantage :** Prend en compte l'historique complet du signal vocalisé, à la fois du début vers la fin et inversement.

---

## 🚀 Installation et Utilisation

### Dépendances requises
En plus des packages standard, ce projet nécessite `librosa` et `tensorflow`.
```bash
pip install tensorflow librosa resampy soundfile
```

### Exécution
Pour lancer l'extraction de caractéristiques et l'entraînement :
```bash
cd CodeAlpha_EmotionRecognition
python emotion_recognition.py
```

Les callbacks configurés incluent :
*   **Early Stopping :** Arrête l'entraînement si la perte de validation ne diminue plus pendant 10 époques.
*   **ReduceLROnPlateau :** Réduit le taux d'apprentissage de moitié si la perte stagne pour éviter les oscillations.
*   **Model Checkpoint :** Sauvegarde automatiquement la meilleure version du modèle.

---

## 📈 Résultats et Visualisations

À l'issue de l'entraînement, les fichiers suivants sont générés :
*   `best_model.h5` / `model_cnn_final.h5` / `model_lstm_final.h5` : Modèles sérialisés réutilisables pour inférence.
*   **`emotion_recognition_results.png`** : Un graphique combinant les courbes d'apprentissage (perte et précision pour le CNN et le LSTM) et la matrice de confusion finale sur le jeu de test pour visualiser les émotions fréquemment confondues par le modèle (ex: *calm* et *neutral*).
