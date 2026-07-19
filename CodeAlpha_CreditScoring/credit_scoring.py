"""
CodeAlpha - Credit Scoring Model (VERSION CORRIGÉE)
===================================================
Prédiction de la solvabilité avec gestion des variables catégorielles.

Dataset : German Credit Dataset (UCI Machine Learning Repository)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, 
                             confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES
# ============================================================
print("=" * 60)
print("📊 CREDIT SCORING MODEL - CodeAlpha")
print("=" * 60)

# Noms des colonnes du dataset German Credit
columns = [
    'status_checking', 'duration_months', 'credit_history', 'purpose',
    'credit_amount', 'savings_account', 'employment_since', 'installment_rate',
    'personal_status', 'other_debtors', 'residence_since', 'property',
    'age', 'other_installments', 'housing', 'existing_credits',
    'job', 'num_dependents', 'telephone', 'foreign_worker', 'credit_risk'
]

# Essayer de télécharger le dataset depuis UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

USE_REAL_DATASET = False

try:
    # Le dataset original utilise des espaces comme séparateurs
    # et contient des codes catégoriels (A11, A12, etc.)
    df = pd.read_csv(url, sep=' ', header=None, names=columns)

    # Vérifier si les données sont numériques
    if df.select_dtypes(include=['object']).shape[1] > 0:
        print("📥 Dataset UCI téléchargé (avec variables catégorielles)")
        print("   Encodage des variables catégorielles en cours...")

        # Encodage des variables catégorielles
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'credit_risk':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        USE_REAL_DATASET = True
    else:
        print("📥 Dataset UCI téléchargé (toutes les colonnes sont numériques)")
        USE_REAL_DATASET = True

except Exception as e:
    print(f"⚠️  Impossible de télécharger le dataset UCI : {e}")
    print("   Utilisation du dataset de démonstration...")

    # Dataset de démonstration avec toutes les colonnes numériques
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame({
        'status_checking': np.random.choice([1, 2, 3, 4], n_samples),
        'duration_months': np.random.randint(4, 72, n_samples),
        'credit_history': np.random.choice([0, 1, 2, 3, 4], n_samples),
        'purpose': np.random.choice([1, 2, 3, 4, 5, 6, 8, 9, 10], n_samples),
        'credit_amount': np.random.randint(250, 18424, n_samples),
        'savings_account': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'employment_since': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'installment_rate': np.random.choice([1, 2, 3, 4], n_samples),
        'personal_status': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'other_debtors': np.random.choice([1, 2, 3], n_samples),
        'residence_since': np.random.choice([1, 2, 3, 4], n_samples),
        'property': np.random.choice([1, 2, 3, 4], n_samples),
        'age': np.random.randint(19, 75, n_samples),
        'other_installments': np.random.choice([1, 2, 3], n_samples),
        'housing': np.random.choice([1, 2, 3], n_samples),
        'existing_credits': np.random.choice([1, 2, 3], n_samples),
        'job': np.random.choice([1, 2, 3, 4], n_samples),
        'num_dependents': np.random.choice([1, 2], n_samples),
        'telephone': np.random.choice([1, 2], n_samples),
        'foreign_worker': np.random.choice([1, 2], n_samples),
        'credit_risk': np.random.choice([1, 2], n_samples, p=[0.7, 0.3])
    })

print(f"\n📋 Dataset : {df.shape[0]} clients, {df.shape[1]} attributs")
print(f"   Mode : {'Dataset UCI RÉEL' if USE_REAL_DATASET else 'Dataset de DÉMONSTRATION'}")

# ============================================================
# ÉTAPE 2 : EXPLORATION DES DONNÉES (EDA)
# ============================================================
print("\n" + "=" * 60)
print("🔍 ÉTAPE 2 : EXPLORATION DES DONNÉES")
print("=" * 60)

# La cible : 1 = bon crédit, 2 = mauvais crédit
# On convertit en 0/1 pour faciliter (0 = bon, 1 = risqué)
df['credit_risk'] = df['credit_risk'].apply(lambda x: 0 if x == 1 else 1)

print("\n📊 Distribution des classes :")
class_counts = df['credit_risk'].value_counts().sort_index()
for cls, count in class_counts.items():
    label = 'Bon crédit' if cls == 0 else 'Risque de défaut'
    print(f"   • {label} ({cls}) : {count} ({count/len(df)*100:.1f}%)")

# Types de données
print("\n📋 Types des colonnes :")
print(df.dtypes)

# ============================================================
# ÉTAPE 3 : PRÉTRAITEMENT & FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("⚙️ ÉTAPE 3 : PRÉTRAITEMENT & FEATURE ENGINEERING")
print("=" * 60)

# Vérifier qu'il n'y a plus de colonnes non-numériques
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"⚠️  Colonnes non-numériques détectées : {non_numeric}")
    print("   Encodage forcé...")
    for col in non_numeric:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Séparation des features et de la cible
X = df.drop('credit_risk', axis=1)
y = df['credit_risk']

print(f"\n📐 Features : {X.shape[1]} colonnes")
print(f"   Toutes numériques : {X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]}")

# Division train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📐 Division des données :")
print(f"   • Entraînement : {X_train.shape[0]} échantillons")
print(f"   • Test : {X_test.shape[0]} échantillons")

# Standardisation des variables numériques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Standardisation effectuée")

# ============================================================
# ÉTAPE 4 : ENTRAÎNEMENT DES MODÈLES
# ============================================================
print("\n" + "=" * 60)
print("🤖 ÉTAPE 4 : ENTRAÎNEMENT DES MODÈLES")
print("=" * 60)

# Dictionnaire des modèles
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
}

# Stockage des résultats
results = {}

for name, model in models.items():
    print(f"\n🔄 Entraînement : {name}...")

    # Entraînement
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'model': model
    }

    print(f"   ✅ Accuracy : {accuracy:.4f}")
    print(f"   ✅ ROC-AUC : {roc_auc:.4f}")

# ============================================================
# ÉTAPE 5 : ÉVALUATION & COMPARAISON
# ============================================================
print("\n" + "=" * 60)
print("📊 ÉTAPE 5 : RÉSULTATS & COMPARAISON")
print("=" * 60)

# Tableau comparatif
comparison_df = pd.DataFrame({
    'Modèle': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'F1-Score': [results[m]['f1'] for m in results],
    'ROC-AUC': [results[m]['roc_auc'] for m in results]
})

print("\n📋 Tableau comparatif des modèles :")
print(comparison_df.round(4).to_string(index=False))

# ============================================================
# ÉTAPE 6 : VISUALISATIONS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Credit Scoring Model - Évaluation des Modèles', fontsize=16, fontweight='bold')

# 1. Comparaison des métriques
ax1 = axes[0, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

for i, (name, res) in enumerate(results.items()):
    values = [res['accuracy'], res['precision'], res['recall'], res['f1'], res['roc_auc']]
    ax1.bar(x + i*width, values, width, label=name, alpha=0.8)

ax1.set_xlabel('Métriques')
ax1.set_ylabel('Score')
ax1.set_title('Comparaison des Métriques par Modèle')
ax1.set_xticks(x + width)
ax1.set_xticklabels(metrics, rotation=15)
ax1.legend()
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)

# 2. Matrices de confusion
ax2 = axes[0, 1]
best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Modèle']
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, 
            xticklabels=['Bon Crédit', 'Risque'], 
            yticklabels=['Bon Crédit', 'Risque'])
ax2.set_title(f'Matrice de Confusion - {best_model_name}')
ax2.set_xlabel('Prédiction')
ax2.set_ylabel('Réalité')

# 3. Courbes ROC
ax3 = axes[1, 0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax3.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.3f})", linewidth=2)
ax3.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.5)')
ax3.set_xlabel('Taux de Faux Positifs (FPR)')
ax3.set_ylabel('Taux de Vrais Positifs (TPR)')
ax3.set_title('Courbes ROC')
ax3.legend(loc='lower right')
ax3.grid(alpha=0.3)

# 4. Importance des features (Random Forest)
ax4 = axes[1, 1]
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

top_features = feature_importance.tail(10)
ax4.barh(top_features['feature'], top_features['importance'], color='forestgreen', alpha=0.8)
ax4.set_xlabel('Importance')
ax4.set_title('Top 10 Features Importantes (Random Forest)')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('credit_scoring_results.png', dpi=150, bbox_inches='tight')
print("\n🖼️ Graphiques sauvegardés : credit_scoring_results.png")
plt.show()

# ============================================================
# ÉTAPE 7 : RAPPORT DÉTAILLÉ
# ============================================================
print("\n" + "=" * 60)
print("📄 RAPPORT DÉTAILLÉ PAR MODÈLE")
print("=" * 60)

for name, res in results.items():
    print(f"\n{'─' * 50}")
    print(f"📝 {name}")
    print(f"{'─' * 50}")
    print(classification_report(y_test, res['y_pred'], 
                               target_names=['Bon Crédit', 'Risque de Défaut']))

print("\n" + "=" * 60)
print("✅ PROJET CREDIT SCORING TERMINÉ!")
print(f"🏆 Meilleur modèle : {best_model_name}")
print(f"   ROC-AUC : {results[best_model_name]['roc_auc']:.4f}")
print(f"   Dataset : {'UCI RÉEL' if USE_REAL_DATASET else 'DÉMONSTRATION'}")
print("=" * 60)