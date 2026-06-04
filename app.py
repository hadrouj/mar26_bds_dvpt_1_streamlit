import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, joblib
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="APD France — Exploration & Modélisation", page_icon="🌍", layout="wide")
BASE = Path(__file__).parent

@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "aide-publique-au-developpement_clean.csv", sep=";")
    return df

@st.cache_resource
def load_model():
    return joblib.load(BASE / "model" / "pipeline.joblib")

def load_json(name):
    with open(BASE / "model" / name) as f:
        return json.load(f)

COLORS = ["#2E74B5", "#4472C4", "#1F3864", "#5B9BD5", "#A5C8E1", "#D6E4F0",
          "#FF6B35", "#F7C948", "#45B7AA", "#E74C3C", "#8E44AD", "#27AE60"]

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🌍 APD France")
st.sidebar.markdown("**Aide Publique au Développement**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "Introduction",
    "Exploration des données",
    "Data Viz",
    "Pre-processing",
    "Modélisation Régression",
    "Modélisation Classification",
    "Démo",
    "Conclusion & Perspectives"
])


st.sidebar.markdown("---")
st.sidebar.caption("DataScientest — ML Engineer")
st.sidebar.caption("Dataset : data.gouv.fr / AFD / SNPC-OCDE")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : INTRODUCTION
# ═════════════════════════════════════════════════════════════════════════════
if page == "Introduction":
    st.title("🌍 Aide Publique au Développement - France")
    st.markdown("### Exploration, Pre-processing & Modélisation Machine Learning")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Lignes brutes", "106 519")
    col2.metric("📐 Colonnes brutes", "103")
    col3.metric("📅 Période", "2018 – 2024")
    col4.metric("🌐 Pays bénéficiaires", "≈ 190")

    st.markdown("")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        #### Objectif du projet
        Prédire le **montant d'engagement financier** (K EUR) accordé par la France
        sur un projet d'aide au développement, à partir de ses caractéristiques
        structurelles (agence, secteur, pays, type de financement, ODD…).

        #### Approche
        1. **Exploration** — 10 visualisations statistiquement commentées
        2. **Pre-processing** — Pipeline de 9 étapes (106Kx103 → 79Kx50)
        3. **Modélisation** — Benchmark de 5 modèles, SHAP, analyse d'erreurs
        4. **Prédiction** — Formulaire interactif pour prédire un nouveau projet
        """)
    with c2:
        st.markdown("""
        #### Résultats clés
        | Métrique | Valeur |
        |---|---|
        | **Meilleur modèle** | RandomForest |
        | **RMSE (log)** | 1,321 |
        | **R²** | 0,682 |
        | **MAE (log)** | 0,967 |
        | **Feature #1** | Agence (η²=0,35) |
        """)

    st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : EXPLORATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Exploration des données":
    st.title("📊 Exploration des données")
    df = load_data()

    st.subheader("Distribution de la variable cible")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Engagements (K EUR)", nbins=100, log_y=True,
                            title="Distribution brute (échelle log-Y)",
                            color_discrete_sequence=[COLORS[0]])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="log_engagements", nbins=80,
                            title="Après log(1+x) — skewness = 0,92",
                            color_discrete_sequence=[COLORS[1]])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Médiane", f"{df['Engagements (K EUR)'].median():,.0f} K EUR")
    c2.metric("Moyenne", f"{df['Engagements (K EUR)'].mean():,.0f} K EUR")
    c3.metric("Skewness (log)", f"{df['log_engagements'].skew():.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : DATA VIZ
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Data Viz":

    st.markdown("<a id='top'></a>", unsafe_allow_html=True)
    st.title(":chart_with_upwards_trend: Data Visualizations")
    st.markdown("### 10 visualisations clés pour comprendre les données")

    # Menu sticky
    st.markdown("""
    <style>
    .sticky-menu {
        position: sticky;
        top: 0;
        background-color: white;
        padding: 10px 0px 15px 0px;
        border-bottom: 1px solid #ddd;
        z-index: 999;
    }
    .sticky-menu a {
        text-decoration: none;
        font-weight: 600;
        color: #2E74B5;
    }
    .sticky-menu a:hover {
        color: #1F3864;
    }
    </style>

    <div class="sticky-menu">
        <ol>
            <li><a href="#viz1">Distribution de la variable cible (brute et log-transformée)</a></li>
            <li><a href="#viz2">Évolution temporelle des engagements (total et nombre de projets par année)</a></li>
            <li><a href="#viz3">Top 15 pays bénéficiaires par volume d'engagements</a></li>
            <li><a href="#viz4">Répartition des engagements par région</a></li>
            <li><a href="#viz5">Répartition des engagements par secteur</a></li>
            <li><a href="#viz6">Répartition des engagements par agence</a></li>
            <li><a href="#viz7">Distribution de log(engagements) par type de financement</a></li>
            <li><a href="#viz8">Heatmap de corrélation des variables numériques</a></li>
            <li><a href="#viz9">Boxplots des engagements par catégorie CAD</a></li>
            <li><a href="#viz10">Taux de valeurs manquantes par variable (top 30)</a></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # --- VISU 1 ---
    st.markdown("<a id='viz2'></a>", unsafe_allow_html=True)
    fig = BASE / "figures" / "fig_00.png"
    if fig.exists():
        st.markdown("---")
        st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h3>1. Distribution de la variable cible (brute et log-transformée)</h3>
            <a href='#top' style='font-size:16px;'>↑ Haut de page</a>
        </div>
        """, unsafe_allow_html=True)
        st.image(str(fig))


    # --- VISU 2 ---
    st.markdown("<a id='viz2'></a>", unsafe_allow_html=True)
    fig = BASE / "figures" / "fig_01.png"
    if fig.exists():
        st.markdown("---")
        st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h3>2. Évolution temporelle des engagements (total et nombre de projets par année)</h3>
            <a href='#top' style='font-size:16px;'>↑ Haut de page</a>
        </div>
        """, unsafe_allow_html=True)
        st.image(str(fig))

    # --- VISU 3 ---
    st.markdown("<a id='viz3'></a>", unsafe_allow_html=True)
    fig = BASE / "figures" / "fig_02.png"
    if fig.exists():
        st.markdown("---")
        st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h3>3. Top 15 pays bénéficiaires par volume d'engagements</h3>
            <a href='#top' style='font-size:16px;'>↑ Haut de page</a>
        </div>
        """, unsafe_allow_html=True)
    st.image(str(fig))

    # --- VISU 4 ---
    st.markdown("<a id='viz4'></a>", unsafe_allow_html=True)
    fig = BASE / "figures" / "fig_03.png"
    if fig.exists():
        st.markdown("---")
        st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h3>4. Répartition des engagements par région</h3>
            <a href='#top' style='font-size:16px;'>↑ Haut de page</a>
        </div>
        """, unsafe_allow_html=True)
        st.image(str(fig))

    # --- VISU 5 ---
    st.markdown("<a id='viz5'></a>", unsafe_allow_html=True)
    fig = BASE / "figures" / "fig_04.png"
    if fig.exists():
        st.markdown("---")
        st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h3>5. Répartition des engagements par secteur</h3>
            <a href='#top' style='font-size:16px;'>↑ Haut de page</a>
        </div>
        """, unsafe_allow_html=True)
        st.image(str(fig))

    # --- VISU 6 ---
    st.markdown("<a id='viz6'></a>", unsafe_allow_html=True)
    fig = BASE / "figures" / "fig_08.png"
    if fig.exists():
        st.markdown("---")
        st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h3>6. Répartition des engagements par agence</h3>
            <a href='#top' style='font-size:16px;'>↑ Haut de page</a>
        </div>
        """, unsafe_allow_html=True)
        st.image(str(fig))

    # --- VISU 7 ---
    st.markdown("<a id='viz7'></a>", unsafe_allow_html=True)
    fig = BASE / "figures" / "fig_06.png"
    if fig.exists():
        st.markdown("---")
        st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h3>7. Distribution de log(engagements) par type de financement</h3>
            <a href='#top' style='font-size:16px;'>↑ Haut de page</a>
        </div>
        """, unsafe_allow_html=True)
        st.image(str(fig))

    # --- VISU 8 ---
    st.markdown("<a id='viz8'></a>", unsafe_allow_html=True)
    fig = BASE / "figures" / "fig_07.png"
    if fig.exists():
        st.markdown("---")
        st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h3>8. Heatmap de corrélation des variables numériques</h3>
            <a href='#top' style='font-size:16px;'>↑ Haut de page</a>
        </div>
        """, unsafe_allow_html=True)
        st.image(str(fig))

    # --- VISU 9 ---
    st.markdown("<a id='viz9'></a>", unsafe_allow_html=True)
    fig = BASE / "figures" / "fig_06.png"
    if fig.exists():
        st.markdown("---")
        st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h3>9. Boxplots des engagements par catégorie CAD</h3>
            <a href='#top' style='font-size:16px;'>↑ Haut de page</a>
        </div>
        """, unsafe_allow_html=True)
        st.image(str(fig))

    # --- VISU 10 ---
    st.markdown("<a id='viz10'></a>", unsafe_allow_html=True)
    fig = BASE / "figures" / "fig_05.png"
    if fig.exists():
        st.markdown("---")
        st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h3>10. Taux de valeurs manquantes par variable (top 30)</h3>
            <a href='#top' style='font-size:16px;'>↑ Haut de page</a>
        </div>
        """, unsafe_allow_html=True)
        st.image(str(fig))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : PRE-PROCESSING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Pre-processing":
    st.title("🔧 Pipeline de pre-processing")

    st.markdown("### Vue d'ensemble — 9 étapes")
    pipeline_data = {
        "Étape": ["0 — Initial","1 — Types","2 — Colonnes red.","3 — Cible","4 — Doublons",
                  "5 — NA > 90%","6 — Log cible","7 — Marqueurs","8 — Libellés","9 — Final + ODD"],
        "Action": ["Chargement CSV","Harmonisation types","Suppression redondances","Imputation + nettoyage",
                   "Déduplication","Colonnes creuses","log(1+x)","Correction barème CAD",
                   "Fusion typos/casse","Leakage + 17 ODD_*"],
        "Lignes": ["106 519","106 519","106 519","83 749","79 025","79 025","79 025","79 025","79 025","79 025"],
        "Colonnes": ["103","103","64","64","58","38","39","39","39","50"],
    }
    st.dataframe(pd.DataFrame(pipeline_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Avant / Après")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes", "106 519 → 79 025", delta="-27 494", delta_color="inverse")
    c2.metric("Colonnes", "103 → 50", delta="-53 + 17 ODD", delta_color="inverse")
    c3.metric("Skewness cible", "76,3 → 0,92", delta="-75,4", delta_color="inverse")

    st.markdown("---")
    with st.expander("📋 Détail des étapes clés", expanded=False):
        st.markdown("""
        **Étape 3 — Imputation de la cible** (2 règles métier)
        - *Règle 1* : « Engagement dérivé du versement » → Montant perçu
        - *Règle 2* : Pure don (Montant versé = Équivalent don) → Équivalent don
        - Puis suppression des lignes à cible NA (-15 207) ou ≤ 0 (-7 563)

        **Étape 7 — Marqueurs politiques**
        - Correction des valeurs hors-barème CAD-OCDE (Désertification, SGMNI) → mode
        - Conversion en Int64 nullable (NaN préservé, pas de -1)

        **Étape 9 — Feature engineering ODD**
        - 17 colonnes binaires par multi-hot encoding de « Titre ODD »
        - Suppression : leakage (Montant perçu), variance nulle (Monnaie), doublons sémantiques
        """)

    st.markdown("### Analyses statistiques")
    tab1, tab2 = st.tabs(["ANOVA", "Spearman"])
    with tab1:
        anova_data = pd.DataFrame({
            "Variable": ["Agence","Type financement","Région","Secteur","Catégorie CAD","Bi/Multi"],
            "F (Fisher)": [2513.5, 3516.4, 685.8, 374.8, 1154.4, 1373.4],
            "η² (eta-carré)": [0.351, 0.151, 0.101, 0.091, 0.081, 0.034],
        })
        fig = px.bar(anova_data, x="η² (eta-carré)", y="Variable", orientation='h',
                    title="ANOVA — Part de variance expliquée (η²) par variable",
                    color="η² (eta-carré)", color_continuous_scale="Blues")
        fig.update_layout(height=350, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        sp_data = pd.DataFrame({
            "Marqueur": ["Gouvernance","Environnement","Genre","Atténuation climat",
                         "Adaptation climat","Biodiversité"],
            "ρ Spearman": [-0.105, 0.100, 0.073, 0.071, 0.054, 0.028],
        })
        fig = px.bar(sp_data, x="ρ Spearman", y="Marqueur", orientation='h',
                    title="Corrélations de Spearman — marqueurs vs log(engagements)",
                    color="ρ Spearman", color_continuous_scale="RdBu_r", range_color=[-0.12, 0.12])
        fig.update_layout(height=300, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : MODÉLISATION RÉGRESSION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Modélisation Régression":
    st.title("Modélisation Machine Learning")

    tab1, tab2, tab3 = st.tabs(["Benchmark", "SHAP", "Analyse d'erreurs"])

    with tab1:
        st.subheader("Benchmark — Validation croisée 5-fold")
        bench = load_json("benchmark.json")
        bench_df = pd.DataFrame({"Modèle": bench['models'], "RMSE (log)": bench['rmse'], "R²": bench['r2']})
        bench_df["Meilleur"] = bench_df["RMSE (log)"] == bench_df["RMSE (log)"].min()

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(bench_df, x="Modèle", y="RMSE (log)", color="Meilleur",
                        color_discrete_map={True: COLORS[0], False: COLORS[5]},
                        title="RMSE (log) par modèle — plus bas = meilleur")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(bench_df, x="Modèle", y="R²", color="Meilleur",
                        color_discrete_map={True: COLORS[0], False: COLORS[5]},
                        title="R² par modèle — plus haut = meilleur")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Performances finales — RandomForest optimisé")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE (log)", "1,321", help="Erreur multiplicative ≈ ×3,7 sur le montant brut")
        c2.metric("R²", "0,682", help="68 % de la variance expliquée")
        c3.metric("MAE (log)", "0,967")

        st.success("""
        **Hyperparamètres optimisés (GridSearchCV)** :
        `n_estimators=800`, `max_depth=None`, `min_samples_leaf=1`
        — entraîné sur 63 220 projets, évalué sur 15 805.
        """)

    with tab2:
        st.subheader("Interprétabilité SHAP — Importance des features")
        shap_data = load_json("shap_top20.json")
        shap_df = pd.DataFrame(shap_data)

        fig = px.bar(shap_df, x="importance", y="feature", orientation='h',
                    title="Top 20 features par importance SHAP moyenne |SHAP|",
                    color="importance", color_continuous_scale="Blues")
        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Lecture clé** : l'Agence domine largement (|SHAP| = 0,63). Elle encode à la fois
        *qui* finance et le profil typique des projets. Le classement SHAP converge avec
        le classement ANOVA du Rendu 1, validant la cohérence de la chaîne complète.
        """)

        with st.expander("🔍 Figures SHAP du notebook (bar plot & beeswarm)"):
            c1, c2 = st.columns(2)
            fig_shap_bar = BASE / "figures" / "r2_fig_03.png"
            fig_shap_bee = BASE / "figures" / "r2_fig_04.png"
            if fig_shap_bar.exists():
                c1.image(str(fig_shap_bar), caption="SHAP Bar Plot")
            if fig_shap_bee.exists():
                c2.image(str(fig_shap_bee), caption="SHAP Beeswarm")

    with tab3:
        st.subheader("Analyse d'erreurs")
        c1, c2 = st.columns(2)
        fig_err_seg = BASE / "figures" / "r2_fig_05.png"
        fig_err_dec = BASE / "figures" / "r2_fig_06.png"
        if fig_err_seg.exists():
            c1.image(str(fig_err_seg), caption="Erreur par segment métier")
        if fig_err_dec.exists():
            c2.image(str(fig_err_dec), caption="RMSE par décile de la cible")

        st.markdown("""
        **Profil d'erreur en U** : les déciles intermédiaires (D3–D5) sont les mieux prédits
        (RMSE = 0,73 à 1,04). Les extrêmes sont plus difficiles :
        - **Très petits projets (D0)** : grande variabilité intrinsèque
        - **Très gros projets (D9)** : opérations souveraines exceptionnelles, trop rares

        **Segments les plus mal prédits** : projets Régionaux (+8,9 pp dans le pire décile)
        et PRITS (+15,9 pp) — montants négociés entre plusieurs bailleurs.
        """)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : MODÉLISATION CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Modélisation Classification":
    st.title("Modélisation Classification")

    st.success("**Performances finales**— BaggingClassifier sur le test set (n = 15 805)" \
    "\n\n **Accuracy** = 0,611  |  **F1-score macro** = 0,610  |  **ROC AUC (OvR, weighted)** = 0,847")


    st.markdown("Par classe :")
    st.markdown("1. **Petit (≤Q1)**    : precision 0,57  |  recall 0,70  |  F1 0,62  (n = 3 995)")
    st.markdown("2. **Moyen (Q1-Q2)**  : precision 0,65  |  recall 0,51  |  F1 0,57  (n = 3 907)")
    st.markdown("3. **Grand (Q2-Q3)**  : precision 0,49  |  recall 0,50  |  F1 0,49  (n = 3 939)")
    st.markdown("4. **Très grand (>Q3)**: precision 0,76  |  recall 0,75  |  F1 0,75  (n = 3 964)")

    fig_conf = BASE / "figures" / "r2_fig_07.png"
    if fig_conf.exists():
        st.markdown("---")
        st.subheader("Classification 4-classes — Matrice de confusion & ROC")
        st.image(str(fig_conf), caption="Matrice de confusion + courbes ROC — BaggingClassifier (ROC AUC = 0,847)")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE : DÉMO
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Démo":
    st.title("Démo")
    st.markdown("Renseignez les caractéristiques d'un projet pour obtenir une estimation du montant engagé.")
    st.markdown("---")

    meta = load_json("meta.json")
    dropdowns = load_json("dropdowns.json")
    model = load_model()
    df = load_data()

    with st.form("prediction_form"):
        st.subheader("Caractéristiques du projet")

        col1, col2, col3 = st.columns(3)
        inputs = {}

        # Categorical inputs
        key_cats = ['Agence', 'Type de financement', 'Pays beneficiaire', 'Région',
                    'Secteur', 'Catégorie CAD', 'Bi/Multi.1']
        with col1:
            for c in key_cats[:3]:
                if c in dropdowns:
                    inputs[c] = st.selectbox(c, dropdowns[c], index=0)
        with col2:
            for c in key_cats[3:6]:
                if c in dropdowns:
                    inputs[c] = st.selectbox(c, dropdowns[c], index=0)
        with col3:
            for c in key_cats[6:]:
                if c in dropdowns:
                    inputs[c] = st.selectbox(c, dropdowns[c], index=0)

        with st.expander("➕ Paramètres avancés (optionnel)", expanded=False):
            c1, c2, c3 = st.columns(3)
            advanced_cats = [c for c in meta['cat_low'] + meta['cat_high'] if c not in key_cats and c in dropdowns]
            for i, c in enumerate(advanced_cats[:12]):
                with [c1, c2, c3][i % 3]:
                    inputs[c] = st.selectbox(c, ["(par défaut)"] + dropdowns.get(c, []), index=0)

        submitted = st.form_submit_button("🚀 Prédire le montant", type="primary", use_container_width=True)

    if submitted:
        # Build input dataframe
        row = {}
        for c in meta['feature_cols']:
            if c in inputs and inputs[c] != "(par défaut)":
                row[c] = inputs[c]
            elif c in meta['num_cols']:
                row[c] = float(df[c].median()) if c in df.columns else 0.0
            else:
                row[c] = df[c].mode().iloc[0] if c in df.columns else "Unknown"

        X_new = pd.DataFrame([row])[meta['feature_cols']]

        # Predict
        log_pred = model.predict(X_new)[0]
        montant_keur = np.expm1(log_pred)
        montant_eur = montant_keur * 1000

        st.markdown("---")
        st.subheader("Résultat de la prédiction")

        c1, c2, c3 = st.columns(3)
        c1.metric("📊 log(1+engagement)", f"{log_pred:.2f}")
        if montant_keur < 1000:
            c2.metric("💰 Montant estimé", f"{montant_keur:,.0f} K EUR")
        else:
            c2.metric("💰 Montant estimé", f"{montant_keur/1000:,.1f} M EUR")
        c3.metric("💶 En euros", f"{montant_eur:,.0f} €")

        # Confidence band
        rmse_log = 1.321
        low = np.expm1(log_pred - rmse_log) * 1000
        high = np.expm1(log_pred + rmse_log) * 1000
        st.info(f"📐 **Fourchette ±1 RMSE** : de {low:,.0f} € à {high:,.0f} € "
                f"(le modèle a une RMSE de {rmse_log} en log, soit un facteur ×{np.exp(rmse_log):.1f})")

        # Classification
        q_boundaries = [1.792, 2.852, 5.017]
        if log_pred <= q_boundaries[0]:
            cat = "🟢 Petit (≤ 5 K EUR)"
        elif log_pred <= q_boundaries[1]:
            cat = "🟡 Moyen (5 – 16 K EUR)"
        elif log_pred <= q_boundaries[2]:
            cat = "🟠 Grand (16 – 150 K EUR)"
        else:
            cat = "🔴 Très grand (> 150 K EUR)"
        st.success(f"**Tranche estimée** : {cat}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE : CONCLUSION & PERSPECTIVES
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Conclusion & Perspectives":
    st.title("Conclusion & Perspectives")


    st.markdown("""
    ### Ce que le modèle a appris

    Le montant d'un engagement APD français se détermine avant tout par **qui** finance
    (Agence, η² = 0,35) et **comment** l'argent circule (type de financement, canal de transfert).
    La géographie et le secteur jouent un rôle secondaire mais significatif.
    Les marqueurs politiques (Genre, Climat, ODD) apportent un signal faible mais robuste.
    """)

    st.markdown("""
    ### Ce que le modèle ne capte pas

    Les projets **multilatéraux et régionaux** restent mal prédits : leurs montants dépendent
    de négociations entre bailleurs non observables dans les données structurelles.
    Les **très gros projets** (> 150 M€) sont trop rares pour généraliser.
    """)

    st.markdown("""

    ### Récapitulatif des choix de modélisation

    | Étape | Choix retenu | Justification |
    |---|---|---|
    | Cible | log(1 + Engagements K EUR) | skewness 76 → 0,96, distribution exploitable |
    | Split | 80/20 stratifié par déciles cible + CV 5-fold | hold-out test intact, représentativité garantie |
    | Encodage bas cardinal | One-Hot | interprétable, faible explosion dimensionnelle |
    | Encodage haute cardinal | Target Encoding (out-of-fold) | capture l'information des 190 pays / 70 agences sans explosion dim. |
    | Modèle | RandomForest | meilleure RMSE log sur validation |
    | Métrique principale | RMSE en log | erreur multiplicative sur engagement original |
    | Interprétabilité | SHAP (TreeExplainer) | contributions par feature, compatible avec les arbres |

    ### Performance finale

    - RMSE (log) = **1,321**
    - R² (log) = **0,682**

    ### Pistes d'amélioration
    
    - **Modèles spécialisés par segment** : entraîner un modèle distinct pour les prêts souverains (AFD) vs les dons (Ministère de l'Europe et des Affaires étrangères) — les profils SHAP confirment que les mécanismes d'allocation sont fondamentalement différents.
    - **Estimer des intervalles de confiance** sur les prédictions, plus utiles opérationnellement que la valeur ponctuelle sur ce type de données fortement dispersées.
    - **Enrichissement externe des données** : intégrer des données externes (PIB, IDH, conflits) pour mieux contextualiser les projets.
""")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("*Source : data.gouv.fr*")

