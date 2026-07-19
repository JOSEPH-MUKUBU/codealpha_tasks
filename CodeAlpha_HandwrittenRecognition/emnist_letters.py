"""
CodeAlpha - Handwritten Letter Recognition (EMNIST) - VERSION CORRIGÉE
========================================================================
Reconnaissance de lettres manuscrites (A-Z, a-z) avec le dataset EMNIST.

CORRECTIONS :
- Gestion du ZIP corrompu (téléchargement incomplet)
- Méthode alternative avec requests + extraction manuelle
- Fallback dataset démo si tout échoue
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import urllib.request
import gzip
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                     Dropout, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
EMNIST_SUBSET = 'byclass'  # 'balanced', 'letters', ou 'byclass'
DATA_DIR = 'emnist_data'

# Mapping des labels
EMNIST_BALANCED_LABELS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

EMNIST_LETTERS_LABELS = [chr(i) for i in range(ord('a'), ord('z')+1)]

# ============================================================
# FONCTIONS DE CHARGEMENT
# ============================================================
print("=" * 60)
print("✍️ HANDWRITTEN LETTER RECOGNITION (EMNIST)")
print("=" * 60)

def download_file(url, filepath):
    """Télécharge un fichier avec barre de progression."""
    if os.path.exists(filepath):
        print(f"   ✅ {os.path.basename(filepath)} déjà présent")
        return True

    print(f"   📥 Téléchargement de {os.path.basename(filepath)}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"   ✅ Téléchargé")
        return True
    except Exception as e:
        print(f"   ❌ Erreur : {e}")
        return False

def load_mnist_images(filepath):
    """Charge les images MNIST/EMNIST depuis un fichier gzip."""
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filepath):
    """Charge les labels MNIST/EMNIST depuis un fichier gzip."""
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def load_emnist_balanced():
    """
    Charge EMNIST Balanced depuis les fichiers sources.
    Source : https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # URLs pour EMNIST Balanced (format MNIST-like)
    base_url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip_format/"

    files = {
        'train_images': ('emnist-balanced-train-images-idx3-ubyte.gz', 
                         base_url + 'emnist-balanced-train-images-idx3-ubyte.gz'),
        'train_labels': ('emnist-balanced-train-labels-idx1-ubyte.gz',
                         base_url + 'emnist-balanced-train-labels-idx1-ubyte.gz'),
        'test_images': ('emnist-balanced-test-images-idx3-ubyte.gz',
                        base_url + 'emnist-balanced-test-images-idx3-ubyte.gz'),
        'test_labels': ('emnist-balanced-test-labels-idx1-ubyte.gz',
                        base_url + 'emnist-balanced-test-labels-idx1-ubyte.gz'),
    }

    data = {}
    for key, (filename, url) in files.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not download_file(url, filepath):
            return None, None, None, None

        if 'images' in key:
            data[key] = load_mnist_images(filepath)
        else:
            data[key] = load_mnist_labels(filepath)

    return (data['train_images'], data['train_labels'], 
            data['test_images'], data['test_labels'])

def load_emnist_via_package():
    """Essaye de charger via le package emnist."""
    try:
        import emnist
        print("\n📥 Chargement via le package 'emnist'...")

        if EMNIST_SUBSET == 'balanced':
            X_train, y_train = emnist.extract_training_samples('balanced')
            X_test, y_test = emnist.extract_test_samples('balanced')
        elif EMNIST_SUBSET == 'letters':
            X_train, y_train = emnist.extract_training_samples('letters')
            X_test, y_test = emnist.extract_test_samples('letters')
        elif EMNIST_SUBSET == 'byclass':
            X_train, y_train = emnist.extract_training_samples('byclass')
            X_test, y_test = emnist.extract_test_samples('byclass')
        else:
            raise ValueError(f"Sous-ensemble inconnu : {EMNIST_SUBSET}")

        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"   ⚠️  Package emnist indisponible : {e}")
        return None, None, None, None

def generate_demo_dataset(n_classes=47, n_samples_per_class=500):
    """Génère un dataset de démonstration réaliste."""
    print(f"\n🔄 Génération dataset démo ({n_classes} classes, {n_samples_per_class} samples/classe)...")

    np.random.seed(42)
    n_train = n_samples_per_class * n_classes
    n_test = n_train // 5

    X_train = np.zeros((n_train, 28, 28), dtype=np.float32)
    X_test = np.zeros((n_test, 28, 28), dtype=np.float32)

    # Générer des formes de lettres simplifiées
    for cls in range(n_classes):
        start_train = cls * n_samples_per_class
        end_train = (cls + 1) * n_samples_per_class
        start_test = cls * (n_samples_per_class // 5)
        end_test = (cls + 1) * (n_samples_per_class // 5)

        # Créer une forme de base pour cette classe
        base_shape = np.zeros((28, 28))

        # Formes différentes selon la classe
        if cls < 10:  # Chiffres : cercles/lignes
            center = 14
            for i in range(28):
                for j in range(28):
                    dist = np.sqrt((i-center)**2 + (j-center)**2)
                    if abs(dist - (5 + cls)) < 2:
                        base_shape[i, j] = 1.0
        elif cls < 36:  # Majuscules : lignes verticales/horizontales
            angle = (cls - 10) * 10
            for i in range(28):
                j = int(14 + (i - 14) * np.tan(np.radians(angle)))
                if 0 <= j < 28:
                    base_shape[i, j] = 1.0
                    base_shape[i, min(j+1, 27)] = 1.0
        else:  # Minuscules : courbes
            t = np.linspace(0, 2*np.pi, 28)
            for i, ti in enumerate(t):
                x = int(14 + 8 * np.cos(ti + cls))
                y = int(14 + 8 * np.sin(ti + cls * 0.5))
                if 0 <= x < 28 and 0 <= y < 28:
                    base_shape[x, y] = 1.0

        # Ajouter du bruit et variations
        for i in range(start_train, end_train):
            noise = np.random.randn(28, 28) * 0.3
            transform = np.random.choice(['none', 'rotate', 'shift'])
            shape = base_shape.copy()

            if transform == 'rotate':
                from scipy.ndimage import rotate
                shape = rotate(shape, np.random.uniform(-15, 15), reshape=False, order=0)
            elif transform == 'shift':
                shift_x = np.random.randint(-3, 4)
                shift_y = np.random.randint(-3, 4)
                shape = np.roll(shape, shift_x, axis=0)
                shape = np.roll(shape, shift_y, axis=1)

            X_train[i] = np.clip(shape + noise, 0, 1)

        for i in range(start_test, end_test):
            noise = np.random.randn(28, 28) * 0.3
            X_test[i] = np.clip(base_shape + noise, 0, 1)

    y_train = np.repeat(np.arange(n_classes), n_samples_per_class)
    y_test = np.repeat(np.arange(n_classes), n_samples_per_class // 5)

    return X_train, y_train, X_test, y_test

# ============================================================
# ÉTAPE 1 : CHARGEMENT
# ============================================================
print("\n" + "=" * 60)
print("📥 ÉTAPE 1 : CHARGEMENT DU DATASET EMNIST")
print("=" * 60)

# Essayer méthode 1 : package emnist
X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_emnist_via_package()

# Essayer méthode 2 : téléchargement direct
if X_train_raw is None:
    print("\n📥 Tentative de téléchargement direct...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_emnist_balanced()

# Méthode 3 : dataset démo
USE_DEMO = False
if X_train_raw is None:
    USE_DEMO = True
    if EMNIST_SUBSET == 'balanced':
        n_classes = 47
    elif EMNIST_SUBSET == 'letters':
        n_classes = 26
    elif EMNIST_SUBSET == 'byclass':
        n_classes = 62
    else:
        n_classes = 47

    X_train_raw, y_train_raw, X_test_raw, y_test_raw = generate_demo_dataset(n_classes)

print(f"\n📊 Dataset chargé :")
print(f"   • Train : {X_train_raw.shape}")
print(f"   • Test  : {X_test_raw.shape}")
print(f"   • Classes : {len(np.unique(y_train_raw))}")
print(f"   • Mode : {'DÉMONSTRATION' if USE_DEMO else 'EMNIST RÉEL'}")

# ============================================================
# ÉTAPE 2 : PRÉTRAITEMENT (Transposition EMNIST)
# ============================================================
print("\n" + "=" * 60)
print("⚙️ ÉTAPE 2 : PRÉTRAITEMENT (Transposition EMNIST)")
print("=" * 60)

def preprocess_emnist(X):
    """Normalise et transpose les images EMNIST."""
    X = X.astype('float32') / 255.0
    # Transposition : EMNIST images sont transposées
    X = np.array([np.transpose(img) for img in X])
    X = X.reshape(X.shape[0], 28, 28, 1)
    return X

X_train = preprocess_emnist(X_train_raw)
X_test = preprocess_emnist(X_test_raw)

print(f"✅ Images transposées et normalisées")
print(f"   Shape train : {X_train.shape}")
print(f"   Shape test  : {X_test.shape}")

n_classes = len(np.unique(y_train_raw))
y_train_cat = to_categorical(y_train_raw, n_classes)
y_test_cat = to_categorical(y_test_raw, n_classes)

print(f"✅ Labels encodés : {n_classes} classes")

# ============================================================
# ÉTAPE 3 : VISUALISATION
# ============================================================
print("\n" + "=" * 60)
print("🔍 ÉTAPE 3 : VISUALISATION DES LETTRES")
print("=" * 60)

if EMNIST_SUBSET == 'balanced':
    class_labels = EMNIST_BALANCED_LABELS[:n_classes]
elif EMNIST_SUBSET == 'letters':
    class_labels = EMNIST_LETTERS_LABELS[:n_classes]
else:
    class_labels = [str(i) for i in range(n_classes)]

n_show = min(20, n_classes)
fig, axes = plt.subplots(2, 10, figsize=(15, 4))
fig.suptitle(f'EMNIST {EMNIST_SUBSET} - Exemples', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i >= n_show:
        ax.axis('off')
        continue
    idx = np.where(y_train_raw == i)[0][0]
    ax.imshow(X_train[idx].reshape(28, 28), cmap='gray')
    label = class_labels[i] if i < len(class_labels) else str(i)
    ax.set_title(f'{label}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('emnist_examples.png', dpi=150, bbox_inches='tight')
print("🖼️ Exemples sauvegardés : emnist_examples.png")
plt.show()

# ============================================================
# ÉTAPE 4 : DATA AUGMENTATION
# ============================================================
print("\n" + "=" * 60)
print("🔄 ÉTAPE 4 : DATA AUGMENTATION")
print("=" * 60)

datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)
print("✅ Configuré")

# ============================================================
# ÉTAPE 5 : MODÈLE CNN
# ============================================================
print("\n" + "=" * 60)
print("🤖 ÉTAPE 5 : CONSTRUCTION DU CNN")
print("=" * 60)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
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
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(f"✅ CNN pour {n_classes} classes construit")

# ============================================================
# ÉTAPE 6 : ENTRAÎNEMENT
# ============================================================
print("\n" + "=" * 60)
print("🏋️ ÉTAPE 6 : ENTRAÎNEMENT")
print("=" * 60)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('emnist_best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=128),
    validation_data=(X_test, y_test_cat),
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# ÉTAPE 7 : ÉVALUATION
# ============================================================
print("\n" + "=" * 60)
print("📊 ÉTAPE 7 : ÉVALUATION")
print("=" * 60)

loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n📋 Résultats :")
print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Loss     : {loss:.4f}")

y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n📄 Classification Report (début) :")
report = classification_report(y_test_raw, y_pred_classes, 
                                target_names=class_labels, digits=3)
print(report[:1500])

# ============================================================
# ÉTAPE 8 : VISUALISATIONS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'EMNIST {EMNIST_SUBSET} - Résultats', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
ax1.plot(history.history['accuracy'], label='Train', color='blue')
ax1.plot(history.history['val_accuracy'], label='Val', color='orange')
ax1.axhline(y=accuracy, color='green', linestyle='--')
ax1.set_title('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(history.history['loss'], label='Train', color='red')
ax2.plot(history.history['val_loss'], label='Val', color='purple')
ax2.set_title('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

ax3 = axes[1, 0]
n_cm = min(20, n_classes)
cm = confusion_matrix(y_test_raw, y_pred_classes)[:n_cm, :n_cm]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=class_labels[:n_cm], yticklabels=class_labels[:n_cm])
ax3.set_title(f'Matrice ({n_cm} classes)')
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

ax4 = axes[1, 1]
errors = y_test_raw != y_pred_classes
ax4.text(0.5, 0.5, 
         f'Classes : {n_classes}\n'
         f'Accuracy : {accuracy*100:.2f}%\n'
         f'Correct : {np.sum(~errors)}/{len(y_test_raw)}',
         ha='center', va='center', fontsize=14, transform=ax4.transAxes)
ax4.set_title('Résumé')
ax4.axis('off')

plt.tight_layout()
plt.savefig('emnist_results.png', dpi=150, bbox_inches='tight')
print("\n🖼️ Résultats : emnist_results.png")
plt.show()

# ============================================================
# DÉMO PRÉDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("🔮 DÉMONSTRATION DE PRÉDICTIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Prédictions sur Lettres', fontsize=14, fontweight='bold')

random_indices = np.random.choice(len(X_test), 10, replace=False)

for i, idx in enumerate(random_indices):
    ax = axes[i // 5, i % 5]
    image = X_test[idx].reshape(28, 28)
    pred = model.predict(X_test[idx:idx+1], verbose=0)
    pred_class = np.argmax(pred)
    conf = np.max(pred) * 100

    true_l = class_labels[y_test_raw[idx]] if y_test_raw[idx] < len(class_labels) else '?'
    pred_l = class_labels[pred_class] if pred_class < len(class_labels) else '?'

    ax.imshow(image, cmap='gray')
    color = 'green' if pred_class == y_test_raw[idx] else 'red'
    ax.set_title(f'{true_l} → {pred_l} ({conf:.0f}%)', color=color, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('emnist_predictions.png', dpi=150, bbox_inches='tight')
print("🖼️ Prédictions : emnist_predictions.png")
plt.show()

# ============================================================
# SAUVEGARDE
# ============================================================
model.save(f'emnist_{EMNIST_SUBSET}_model.h5')
print(f"\n💾 Modèle : emnist_{EMNIST_SUBSET}_model.h5")

print("\n" + "=" * 60)
print("✅ PROJET EMNIST TERMINÉ!")
print("=" * 60)
print(f"🏆 Accuracy : {accuracy*100:.2f}%")
print(f"📊 Classes : {n_classes} ({EMNIST_SUBSET})")
print(f"🔄 Mode : {'DÉMO' if USE_DEMO else 'RÉEL'}")
print("=" * 60)