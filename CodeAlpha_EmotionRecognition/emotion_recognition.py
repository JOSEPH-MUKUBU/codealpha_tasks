"""
CodeAlpha - Emotion Recognition from Speech (VERSION FINALE)
=============================================================
Reconnaissance des émotions avec le VRAI dataset RAVDESS.

AMÉLIORATIONS :
- Features temporelles complètes (MFCC + delta + delta2)
- CNN 2D sur spectrogrammes MFCC
- LSTM Bidirectionnel
- Data augmentation audio
- Affichage clair du mode utilisé
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "ravdess_data/"
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
MAX_LEN = 130

EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# ============================================================
# IMPORTS
# ============================================================
print("=" * 60)
print("🎵 EMOTION RECOGNITION - CodeAlpha")
print("=" * 60)

try:
    import librosa
    import librosa.display
    print("✅ Librosa OK")
except ImportError:
    print("❌ Librosa non installé. Exécutez : pip install librosa resampy soundfile")
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten,
                                         LSTM, Bidirectional, Dropout, BatchNormalization)
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    print(f"✅ TensorFlow {tf.__version__} OK")
except ImportError:
    print("❌ TensorFlow non installé. Exécutez : pip install tensorflow")
    sys.exit(1)

# ============================================================
# ÉTAPE 1 : EXTRACTION DES FEATURES
# ============================================================
print("\n" + "=" * 60)
print("🔧 ÉTAPE 1 : EXTRACTION DES FEATURES MFCC")
print("=" * 60)

def pad_mfcc(mfcc, max_len=MAX_LEN):
    """Pad ou tronque les MFCC à une longueur fixe."""
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def extract_features(file_path, augment=False):
    """Extraction MFCC + delta + delta-delta avec structure temporelle."""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # Data augmentation
        if augment:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        features = pad_mfcc(features, MAX_LEN)

        return features
    except Exception as e:
        return None

def load_ravdess_data(data_path, augment=False):
    """Charge le dataset RAVDESS."""
    features = []
    labels = []

    if not os.path.exists(data_path):
        print(f"❌ Dossier '{data_path}' introuvable!")
        return None, None

    actor_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"📂 {len(actor_dirs)} dossiers d'acteurs trouvés")

    for actor_dir in sorted(actor_dirs):
        actor_path = os.path.join(data_path, actor_dir)
        for file in os.listdir(actor_path):
            if not file.endswith('.wav'):
                continue

            file_path = os.path.join(actor_path, file)

            try:
                emotion_code = file.split('-')[2]
                emotion = EMOTIONS.get(emotion_code)
                if not emotion:
                    continue
            except:
                continue

            feat = extract_features(file_path, augment=augment)
            if feat is not None:
                features.append(feat)
                labels.append(emotion)

    if len(features) == 0:
        return None, None

    return np.array(features), np.array(labels)

# Chargement données RAVDESS
print("\n📥 Chargement du dataset RAVDESS...")
X, y = load_ravdess_data(DATA_PATH, augment=False)

# ============================================================
# VÉRIFICATION DU MODE
# ============================================================
USE_SYNTHETIC = False

if X is None or len(X) == 0:
    USE_SYNTHETIC = True
    print("\n" + "=" * 60)
    print("🔄 MODE DÉMONSTRATION : Dataset synthétique")
    print("=" * 60)
    print("⚠️  Le vrai dataset RAVDESS n'a pas été trouvé ou est vide.")
    print("    Vérifiez que 'ravdess_data/' contient les dossiers Actor_01 à Actor_24")

    np.random.seed(42)
    n_samples = 1440
    n_channels = 120

    X = np.random.randn(n_samples, n_channels, MAX_LEN)
    y = np.random.choice(list(EMOTIONS.values()), n_samples)

    for i, emotion in enumerate(EMOTIONS.values()):
        mask = y == emotion
        pattern = np.sin(np.linspace(0, (i+1)*np.pi, MAX_LEN)) * 0.5
        X[mask] += pattern[np.newaxis, np.newaxis, :] * np.random.randn(n_channels, 1)

# ============================================================
# AFFICHAGE DU MODE UTILISÉ
# ============================================================
print("\n" + "=" * 60)
if USE_SYNTHETIC:
    print("⚠️  MODE : Dataset SYNTHÉTIQUE (démonstration)")
else:
    print("✅ MODE : Dataset RAVDESS RÉEL")
print("=" * 60)

print(f"\n📊 Dataset : {X.shape[0]} échantillons, shape={X.shape}")

unique, counts = np.unique(y, return_counts=True)
print(f"\n📋 Distribution des émotions :")
for emotion, count in zip(unique, counts):
    print(f"   • {emotion:12s}: {count:4d} ({count/len(y)*100:5.1f}%)")

# ============================================================
# ÉTAPE 2 : PRÉTRAITEMENT
# ============================================================
print("\n" + "=" * 60)
print("⚙️ ÉTAPE 2 : PRÉTRAITEMENT")
print("=" * 60)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

print(f"✅ Classes encodées : {le.classes_}")

# Normalisation
X_norm = np.zeros_like(X, dtype=np.float32)
for i in range(X.shape[1]):
    mean = np.mean(X[:, i, :])
    std = np.std(X[:, i, :])
    if std > 0:
        X_norm[:, i, :] = (X[:, i, :] - mean) / std

# Reshape pour CNN 2D
X_cnn = X_norm[..., np.newaxis]

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"✅ Input shape: {X_train.shape[1:]}")

# ============================================================
# ÉTAPE 3 : MODÈLES
# ============================================================
print("\n" + "=" * 60)
print("🤖 ÉTAPE 3 : CONSTRUCTION DES MODÈLES")
print("=" * 60)

input_shape = X_train.shape[1:]

# CNN 2D
print("\n🔧 CNN 2D...")
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model_cnn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# LSTM Bidirectionnel
print("🔧 LSTM Bidirectionnel...")
X_train_lstm = X_train.squeeze(-1).transpose(0, 2, 1)
X_test_lstm = X_test.squeeze(-1).transpose(0, 2, 1)

model_lstm = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=X_train_lstm.shape[1:]),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model_lstm.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Modèles construits")

# ============================================================
# ÉTAPE 4 : ENTRAÎNEMENT
# ============================================================
print("\n" + "=" * 60)
print("🏋️ ÉTAPE 4 : ENTRAÎNEMENT")
print("=" * 60)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\n🔄 Entraînement CNN 2D...")
history_cnn = model_cnn.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n🔄 Entraînement LSTM...")
history_lstm = model_lstm.fit(
    X_train_lstm, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# ÉTAPE 5 : ÉVALUATION
# ============================================================
print("\n" + "=" * 60)
print("📊 ÉTAPE 5 : ÉVALUATION")
print("=" * 60)

# CNN
loss_cnn, acc_cnn = model_cnn.evaluate(X_test, y_test, verbose=0)
y_pred_cnn = np.argmax(model_cnn.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

print(f"\n📋 CNN 2D :")
print(f"   Accuracy : {acc_cnn*100:.2f}%")
print(f"   Loss     : {loss_cnn:.4f}")

# LSTM
loss_lstm, acc_lstm = model_lstm.evaluate(X_test_lstm, y_test, verbose=0)
y_pred_lstm = np.argmax(model_lstm.predict(X_test_lstm, verbose=0), axis=1)

print(f"\n📋 LSTM :")
print(f"   Accuracy : {acc_lstm*100:.2f}%")
print(f"   Loss     : {loss_lstm:.4f}")

print("\n📄 Classification Report (CNN) :")
print(classification_report(y_true, y_pred_cnn, target_names=le.classes_, digits=3))

print("\n📄 Classification Report (LSTM) :")
print(classification_report(y_true, y_pred_lstm, target_names=le.classes_, digits=3))

# ============================================================
# ÉTAPE 6 : VISUALISATIONS
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Emotion Recognition - Résultats', fontsize=16, fontweight='bold')

# CNN
ax1 = axes[0, 0]
ax1.plot(history_cnn.history['accuracy'], label='Train', color='blue')
ax1.plot(history_cnn.history['val_accuracy'], label='Validation', color='orange')
ax1.set_title('CNN - Accuracy')
ax1.set_xlabel('Époch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(history_cnn.history['loss'], label='Train', color='blue')
ax2.plot(history_cnn.history['val_loss'], label='Validation', color='orange')
ax2.set_title('CNN - Loss')
ax2.set_xlabel('Époch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

ax3 = axes[0, 2]
cm_cnn = confusion_matrix(y_true, y_pred_cnn)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax3.set_title('Matrice - CNN')

# LSTM
ax4 = axes[1, 0]
ax4.plot(history_lstm.history['accuracy'], label='Train', color='green')
ax4.plot(history_lstm.history['val_accuracy'], label='Validation', color='red')
ax4.set_title('LSTM - Accuracy')
ax4.set_xlabel('Époch')
ax4.set_ylabel('Accuracy')
ax4.legend()
ax4.grid(alpha=0.3)

ax5 = axes[1, 1]
ax5.plot(history_lstm.history['loss'], label='Train', color='green')
ax5.plot(history_lstm.history['val_loss'], label='Validation', color='red')
ax5.set_title('LSTM - Loss')
ax5.set_xlabel('Époch')
ax5.set_ylabel('Loss')
ax5.legend()
ax5.grid(alpha=0.3)

ax6 = axes[1, 2]
cm_lstm = confusion_matrix(y_true, y_pred_lstm)
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Greens', ax=ax6,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax6.set_title('Matrice - LSTM')

plt.tight_layout()
plt.savefig('emotion_recognition_results.png', dpi=150, bbox_inches='tight')
print("\n🖼️ Résultats sauvegardés : emotion_recognition_results.png")
plt.show()

# ============================================================
# SAUVEGARDE
# ============================================================
model_cnn.save('model_cnn_final.h5')
model_lstm.save('model_lstm_final.h5')

print("\n" + "=" * 60)
print("✅ PROJET EMOTION RECOGNITION TERMINÉ!")
print("=" * 60)
print(f"🏆 CNN 2D  : {acc_cnn*100:.2f}%")
print(f"🏆 LSTM    : {acc_lstm*100:.2f}%")
print("=" * 60)