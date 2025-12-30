# 🎯 Employee Attrition Prediction - Machine Learning Project

> **Projet Data Science S7 | 2024-2025**  
> Analyse comparative de 12 modèles de Machine Learning pour la prédiction du turnover des employés

## 📊 Aperçu du Projet

- **Dataset**: IBM HR Employee Attrition (1,470 employés, 35 colonnes)
- **Objectif**: Prédire si un employé quittera l'entreprise (classification binaire)
- **Meilleur modèle**: Voting Classifier (**87.07%** accuracy)
- **Réduction de features**: 34 → 12 variables clés
- **Interface**: GUI interactive moderne (Tkinter)

## 🏆 Résultats Principaux

| Rang | Modèle | Accuracy | Precision | Recall | F1-Score |
|:---:|---|:---:|:---:|:---:|:---:|
| **1** | **Voting Classifier** | **87.07%** | 76.47% | 27.66% | 40.62% |
| 2 | Logistic Regression | 86.73% | 75.00% | 25.53% | 38.10% |
| 3 | XGBoost | 86.39% | 66.67% | 29.79% | 41.18% |
| 4 | SVM (RBF) | 85.71% | 72.73% | 17.02% | 27.59% |
| 5 | Gradient Descent | 85.03% | 54.29% | 40.43% | 46.34% |
| 6 | Decision Tree | 84.35% | 51.35% | 40.43% | 45.24% |
| 7-10 | K-NN / Naive Bayes / RF | 84.01% | 50.00% | varies | varies |
| 11 | Tuned Random Forest | 84.35% | 53.85% | 14.89% | 23.33% |
| 12 | K-Means* | 84.01% | 0.00% | 0.00% | 0.00% |

*K-Means est non-supervisé, utilisé comme baseline de comparaison

## 📁 Structure du Projet

```
final_submission/
├── 📂 GUI/
│   └── gui_attrition.py              # Application GUI interactive
├── 📂 Graphes/
│   ├── model_comparison.png          # Comparaison des modèles
│   ├── feature_importance_all.png    # Importance des features
│   ├── knn_elbow_method.png          # Méthode du coude K-NN
│   ├── conclusion_*.png              # Graphes de conclusion
│   └── eda_plots/                    # Visualisations EDA
├── 📂 Rapport et présentation/
│   ├── rapport_attrition.pdf         # Rapport PDF compilé
│   ├── rapport_attrition.tex         # Source LaTeX du rapport
│   └── powerpointAttrition.pptx      # Présentation PowerPoint
├── employee_attrition.ipynb          # Notebook Jupyter complet
├── model_comparison_results.csv      # Résultats des modèles
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Dataset IBM HR
└── README.md                         # Ce fichier
```

## 🔍 Insights Clés

### Top 5 Prédicteurs d'Attrition
1. **OverTime** - Indicateur le plus fort (risque ×3)
2. **YearsAtCompany** - Ancienneté critique
3. **MonthlyIncome** - Salaires bas = mobilité accrue
4. **Age** - Jeunes employés plus à risque
5. **DistanceFromHome** - Trajet domicile-travail

### Recommandations RH
- 🚨 Les employés en heures supplémentaires ont un taux d'attrition **3× supérieur**
- 💰 Les employés partis gagnaient **30% de moins** en moyenne
- ⏳ **Les 2 premières années** sont critiques pour la rétention
- 👥 Les moins de 35 ans sont plus à risque

## 🚀 Installation & Exécution

### Prérequis
```bash
Python 3.8+
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Exécuter la GUI
```bash
cd GUI
python gui_attrition.py
```

### Exécuter le Notebook
Ouvrir `employee_attrition.ipynb` dans Jupyter ou Google Colab.

## 🔬 Méthodologie

1. **Chargement & EDA** - Analyse exploratoire complète
2. **Feature Engineering** - Réduction 34→12 features, encodage, normalisation
3. **Train-Test Split** - 80/20 avec stratification
4. **12 Modèles** - Entraînement et évaluation comparative
5. **Cross-Validation** - Validation croisée 5-fold
6. **Grid Search** - Optimisation hyperparamètres Random Forest

## 👥 Auteurs

- **Mehdi BAHI**
- **Mustapha MELLAKI**

## 📄 Licence

MIT License - See [LICENSE](LICENSE) file  
Dataset: [IBM HR Analytics Attrition Dataset (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)