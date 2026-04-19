# DeepSDF – Analyse et correction du déséquilibre des données

## 📌 Description

Ce projet explore l’apprentissage de représentations implicites 3D à l’aide du modèle **DeepSDF**.  
L’objectif principal est d’identifier et corriger un problème critique lié à la distribution des données d’entraînement, impactant fortement la qualité de reconstruction des surfaces.

Une analyse approfondie du signal SDF a été menée, suivie d’une étape de rééquilibrage des données, puis d’une validation expérimentale.

---

## 🧠 Contexte

Le modèle **DeepSDF** apprend une fonction de distance signée (SDF) :

- SDF < 0 : point à l’intérieur de l’objet  
- SDF > 0 : point à l’extérieur  
- SDF = 0 : surface  

L’apprentissage correct de cette fonction nécessite une distribution équilibrée de points de part et d’autre de la surface.

---

## ⚠️ Problème identifié

L’analyse du dataset initial (`LFPG_sdf_train.parquet`) révèle :

- Absence de points extérieurs  
- Distribution fortement biaisée vers la surface (~91%)  
- Moyenne négative du SDF  

➡️ Conséquence :  
Le modèle ne peut pas apprendre correctement la structure de la fonction SDF, ce qui empêche une reconstruction fidèle.

---

## 🔧 Solution proposée

Un nouveau dataset équilibré (`LFPG_sdf_train_balanced.parquet`) a été généré avec :

- Répartition homogène intérieur / extérieur  
- Moyenne du SDF proche de 0  
- Meilleure couverture de l’espace  

➡️ Objectif : fournir un signal d’apprentissage complet et cohérent.

---

## 📊 Comparaison des datasets

| Dataset | Surface | Intérieur | Extérieur | SDF mean |
|--------|--------|----------|----------|----------|
| Initial | 91.37% | 8.63% | 0.00% | -0.0107 |
| Équilibré | 78.94% | 10.53% | 10.54% | ~0 |

---

## 🧪 Pipeline

1. Analyse statistique du SDF  
2. Identification du déséquilibre  
3. Génération d’un dataset équilibré  
4. Entraînement de DeepSDF  
5. Comparaison des performances  

---

## 📁 Structure du projet

