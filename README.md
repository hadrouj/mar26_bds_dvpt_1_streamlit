# 🌍 APD France — Streamlit App

Application Streamlit pour l'exploration et la modélisation de l'Aide Publique au Développement française.

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre à l'adresse `http://localhost:8501`.

## Structure

```
streamlit_app/
├── app.py                                    # Application principale
├── requirements.txt                          # Dépendances Python
├── aide-publique-au-developpement_clean.csv  # Dataset nettoyé (79 025 × 50)
├── model/
│   ├── pipeline.joblib                       # Pipeline scikit-learn entraîné
│   ├── meta.json                             # Métadonnées (colonnes, métriques)
│   ├── dropdowns.json                        # Valeurs pour les menus déroulants
│   ├── benchmark.json                        # Résultats du benchmark
│   └── shap_top20.json                       # Top 20 features SHAP
├── figures/                                  # Figures pré-générées
│   ├── fig_00.png ... fig_09.png             # Figures Rendu 1
│   └── r2_fig_00.png ... r2_fig_07.png       # Figures Rendu 2
└── README.md
```

## Sections

1. **Introduction** — Contexte du projet et résultats clés
2. **Exploration** — Graphiques interactifs Plotly (distribution, géographie, secteurs)
3. **Pre-processing** — Pipeline de 9 étapes avec statistiques ANOVA/Spearman
4. **Modélisation** — Benchmark, SHAP, analyse d'erreurs, classification
5. **Prédiction** — Formulaire interactif pour estimer le montant d'un projet
