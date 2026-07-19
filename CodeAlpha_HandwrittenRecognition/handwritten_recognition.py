"""
CodeAlpha - Handwritten Character Recognition
===============================================
Reconnaissance de chiffres manuscrits (MNIST) et extension possible à EMNIST.

Modèle : Convolutional Neural Network (CNN)
Dataset : MNIST (70,000 images de chiffres 0-9, 28x28 pixels)
Extension : EMNIST pour lettres (A-Z, a-z)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                     Dropout, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES MNIST
# ============================================================
print("=" * 60)
print("✍️ HANDWRITTEN CHARACTER RECOGNITION - CodeAlpha")
print("=" * 60)

print("\n📥 Chargement du dataset MNIST...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"✅ Dataset chargé avec succès!")
print(f"   • Images d'entraînement : {X_train.shape}")
print(f"   • Images de test : {X_test.shape}")
print(f"   • Classes : chiffres de 0 à 9")

# ============================================================
# ÉTAPE 2 : EXPLORATION & VISUALISATION
# ============================================================
print("\n" + "=" * 60)
print("🔍 ÉTAPE 2 : EXPLORATION DES DONNÉES")
print("=" * 60)

# Affichage de quelques exemples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Exemples de Chiffres Manuscrits (MNIST)', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    # Trouver une image de chaque chiffre
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(f'Chiffre: {i}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_examples.png', dpi=150, bbox_inches='tight')
print("🖼️ Exemples sauvegardés : mnist_examples.png")
plt.show()

# Distribution des classes
print("\n📊 Distribution des classes :")
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

for digit, count_tr, count_te in zip(unique_train, counts_train, counts_test):
    print(f"   • Chiffre {digit}: {count_tr} train, {count_te} test")

# ============================================================
# ÉTAPE 3 : PRÉTRAITEMENT DES DONNÉES
# ============================================================
print("\n" + "=" * 60)
print("⚙️ ÉTAPE 3 : PRÉTRAITEMENT")
print("=" * 60)

# Normalisation : pixels de [0, 255] -> [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape pour CNN : (n_samples, 28, 28) -> (n_samples, 28, 28, 1)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print(f"✅ Données normalisées et reshapées : {X_train.shape}")

# Encodage one-hot des labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print("✅ Labels encodés (one-hot)")

# ============================================================
# ÉTAPE 4 : DATA AUGMENTATION (OPTIONNELLE)
# ============================================================
print("\n" + "=" * 60)
print("🔄 ÉTAPE 4 : DATA AUGMENTATION")
print("=" * 60)

datagen = ImageDataGenerator(
    rotation_range=10,        # Rotation de ±10 degrés
    zoom_range=0.1,           # Zoom de ±10%
    width_shift_range=0.1,    # Décalage horizontal
    height_shift_range=0.1,   # Décalage vertical
    shear_range=0.1,          # Cisaillement
    fill_mode='nearest'       # Mode de remplissage
)

print("✅ Générateur de data augmentation configuré")
print("   • Rotation: ±10°")
print("   • Zoom: ±10%")
print("   • Décalage: ±10%")

# ============================================================
# ÉTAPE 5 : CONSTRUCTION DU MODÈLE CNN
# ============================================================
print("\n" + "=" * 60)
print("🤖 ÉTAPE 5 : CONSTRUCTION DU MODÈLE CNN")
print("=" * 60)

model = Sequential([
    # Bloc 1 : Convolution + Pooling
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Bloc 2 : Convolution + Pooling
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Bloc 3 : Convolution + Pooling
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.25),

    # Couches fully connected
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes (chiffres 0-9)
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Modèle CNN construit!")
model.summary()

# ============================================================
# ÉTAPE 6 : ENTRAÎNEMENT
# ============================================================
print("\n" + "=" * 60)
print("🏋️ ÉTAPE 6 : ENTRAÎNEMENT")
print("=" * 60)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_model.h5', 
                    monitor='val_accuracy', save_best_only=True, verbose=1)
]

epochs = 30
batch_size = 128

print(f"\n🔄 Lancement de l'entraînement...")
print(f"   • Épochs max : {epochs}")
print(f"   • Batch size : {batch_size}")
print(f"   • Early stopping : patience=10")

history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=batch_size),
    validation_data=(X_test, y_test_cat),
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# ÉTAPE 7 : ÉVALUATION
# ============================================================
print("\n" + "=" * 60)
print("📊 ÉTAPE 7 : ÉVALUATION")
print("=" * 60)

# Évaluation sur le test set
print("\n📋 Évaluation sur le jeu de test :")
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"   • Loss : {loss:.4f}")
print(f"   • Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")

# Prédictions
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\n📄 Classification Report :")
print(classification_report(y_test, y_pred_classes, digits=4))

# ============================================================
# ÉTAPE 8 : VISUALISATIONS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Handwritten Character Recognition - Résultats', fontsize=16, fontweight='bold')

# 1. Courbes d'apprentissage
ax1 = axes[0, 0]
ax1.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
ax1.axhline(y=accuracy, color='green', linestyle='--', label=f'Test Accuracy: {accuracy:.4f}')
ax1.set_title('Accuracy par Époch')
ax1.set_xlabel('Époch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Courbes de loss
ax2 = axes[0, 1]
ax2.plot(history.history['loss'], label='Train Loss', color='red', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', color='purple', linewidth=2)
ax2.set_title('Loss par Époch')
ax2.set_xlabel('Époch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Matrice de confusion
ax3 = axes[1, 0]
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
            xticklabels=range(10), yticklabels=range(10))
ax3.set_title('Matrice de Confusion')
ax3.set_xlabel('Prédiction')
ax3.set_ylabel('Réalité')

# 4. Prédictions avec erreurs
ax4 = axes[1, 1]
# Trouver les prédictions incorrectes
errors = y_test != y_pred_classes
error_indices = np.where(errors)[0]

# Afficher 10 erreurs aléatoires
if len(error_indices) > 0:
    sample_errors = np.random.choice(error_indices, min(10, len(error_indices)), replace=False)
    fig2, axes2 = plt.subplots(2, 5, figsize=(12, 5))
    fig2.suptitle('Exemples de Prédictions Erronées', fontsize=14, fontweight='bold')

    for i, idx in enumerate(sample_errors[:10]):
        ax = axes2[i // 5, i % 5]
        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'Vrai: {y_test[idx]} | Prédit: {y_pred_classes[idx]}', color='red')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_errors.png', dpi=150, bbox_inches='tight')
    print("🖼️ Erreurs de prédiction sauvegardées : prediction_errors.png")
    plt.show()

# Résumé des erreurs dans le subplot principal
ax4.text(0.5, 0.5, f'Prédictions correctes : {np.sum(~errors)}/{len(y_test)}\n'
                    f'Prédictions erronées : {np.sum(errors)}/{len(y_test)}\n'
                    f'Taux d\'erreur : {np.sum(errors)/len(y_test)*100:.2f}%',
         ha='center', va='center', fontsize=12, transform=ax4.transAxes)
ax4.set_title('Résumé des Erreurs')
ax4.axis('off')

plt.tight_layout()
plt.savefig('handwritten_results.png', dpi=150, bbox_inches='tight')
print("🖼️ Résultats sauvegardés : handwritten_results.png")
plt.show()

# ============================================================
# ÉTAPE 9 : PRÉDICTIONS SUR NOUVELLES IMAGES
# ============================================================
print("\n" + "=" * 60)
print("🔮 ÉTAPE 9 : DÉMONSTRATION DE PRÉDICTIONS")
print("=" * 60)

# Sélectionner 10 images aléatoires du test set
random_indices = np.random.choice(len(X_test), 10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Prédictions du Modèle sur Nouvelles Images', fontsize=14, fontweight='bold')

for i, idx in enumerate(random_indices):
    ax = axes[i // 5, i % 5]
    image = X_test[idx].reshape(28, 28)
    prediction = model.predict(X_test[idx:idx+1], verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    ax.imshow(image, cmap='gray')
    color = 'green' if predicted_class == y_test[idx] else 'red'
    ax.set_title(f'Prédit: {predicted_class} ({confidence:.1f}%)\nVrai: {y_test[idx]}', color=color)
    ax.axis('off')

plt.tight_layout()
plt.savefig('predictions_demo.png', dpi=150, bbox_inches='tight')
print(" Démo de prédictions sauvegardée : predictions_demo.png")
plt.show()

# ============================================================
# ÉTAPE 10 : SAUVEGARDE DU MODÈLE
# ============================================================
model.save('mnist_cnn_model.h5')
print("\n Modèle sauvegardé : mnist_cnn_model.h5")

print("\n" + "=" * 60)
print(" PROJET HANDWRITTEN CHARACTER RECOGNITION TERMINÉ!")
print(f"🏆 Accuracy finale : {accuracy*100:.2f}%")
print("=" * 60)
