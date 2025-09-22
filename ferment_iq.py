
import os
import logging

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

os.environ.setdefault('MALLOC_ARENA_MAX', '4')

DISABLE_XGBOOST = os.getenv('DISABLE_XGBOOST', '0') == '1'
if DISABLE_XGBOOST:
    logging.info('DISABLE_XGBOOST env var set; XGBoost import will be skipped; using RandomForest fallback.')

import os
from dataclasses import dataclass
import logging
import sys
import argparse
import os

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
    logging.info('Streamlit available: UI mode enabled')
except Exception:
    STREAMLIT_AVAILABLE = False
    logging.info('Streamlit not available: falling back to CLI mode')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

if not DISABLE_XGBOOST:
    try:
        import xgboost as xgb
        HAS_XGBOOST = True
        logging.info('xgboost available — will use XGBRegressor')
    except Exception as e:
        HAS_XGBOOST = False
        logging.exception('xgboost import failed or not present — falling back to RandomForestRegressor')
        from sklearn.ensemble import RandomForestRegressor
else:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestRegressor
    logging.info('xgboost import skipped by DISABLE_XGBOOST - using RandomForestRegressor')

try:
    import shap
    HAS_SHAP = True
    logging.info('shap available — will show SHAP explainability when possible')
except Exception:
    HAS_SHAP = False
    logging.info('shap not available — skipping SHAP plots')

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False


NAME_EFFECTS = {
    'non anti-biotic': +9.0,
    'animal-free': +7.0,
    'non-animal': +6.0,
    'traditionally fermented protein': +5.0,
    'fermentation-derived protein': +2.0,
    'precision-fermentation ingredient': 0.0,
    'precision-fermentation-derived protein': -1.0,
    'microbial fermentation protein': -2.0,
    'synthetic': -3.0,
    'artificial': -3.0,
}

naming_options = [
    'precision-fermentation-derived protein',
    'fermentation-derived protein',
    'precision-fermentation ingredient',
    'microbial fermentation protein',
    'traditionally fermented protein',
    'animal-free',
    'non-animal',
    'non anti-biotic',
    'synthetic',
    'artificial'
]

REGION_WEIGHTS = {
    'India': 8.0,
    'Brazil': 7.0,
    'Germany': 4.0,
    'USA': 3.0,
    'UK': 1.0
}
regions = list(REGION_WEIGHTS.keys())



def _name_effect_from_label(label: str, name_effects: dict = NAME_EFFECTS):
    
    if not isinstance(label, str):
        return 0.0
    L = label.lower()
    matches = [v for k, v in name_effects.items() if k in L]
    if not matches:
        return 0.0
    return float(max(matches))



NAME_EFFECTS = {
    'non anti-biotic': 9.0,
    'animal-free': 7.0,
    'non-animal': 6.0,
    'traditionally fermented protein': 5.0,
    'fermentation-derived protein': 2.0,
    'precision-fermentation ingredient': 0.0,
    'precision-fermentation-derived protein': -1.0,
    'microbial fermentation protein': -2.0,
    'synthetic': -3.0,
    'artificial': -3.0,
}

def generate_synthetic_pf_data(n=2000, random_seed=42):
    
    rng = np.random.default_rng(random_seed)

    process_pool = ['refined','minimally processed','heat-treated','encapsulated']
    function_pool = ['emulsifier','texturizer','protein','fat','flavor enhancer']

    naming_options = list(NAME_EFFECTS.keys())

    rows = []
    for i in range(n):
        name = str(rng.choice(naming_options))
        protein = float(max(0, rng.normal(45, 12))) if rng.random() < 0.7 else float(max(0, rng.normal(12,6)))
        fat = float(max(0, rng.normal(8,4)))
        carbs = float(max(0, rng.normal(6,5)))
        calories = protein*4 + fat*9 + carbs*4 + float(rng.normal(0,10))
        b12 = float(max(0, rng.normal(1.2,0.8))) if protein>30 else float(max(0, rng.normal(0.2,0.3)))
        iron = float(abs(rng.normal(2.5,1.0)))

        region = str(rng.choice(regions))
        region_weight = float(REGION_WEIGHTS.get(region, 0.0))
        process = str(rng.choice(process_pool))
        function = str(rng.choice(function_pool))
        ingredient_complexity = int(rng.integers(1,10))
        disclosure_level = int(rng.integers(0,4))
        endorsement = int(rng.choice([0,1], p=[0.85,0.15]))
        age = int(rng.integers(18,75))
        allergen_flag = int(rng.random() < 0.08)  
        hazard_flag = int(rng.random() < 0.02)

        environment_focus = int(rng.integers(0,4))

        narrative_neg = float(rng.random()*0.3)
        narrative_pos = float(rng.random()*0.3) + (0.1 * (environment_focus/3.0))  

        name_effect = NAME_EFFECTS.get(name, 0.0)

        
        base_trust = 50.0

        protein_weight = 1.2
        b12_weight = 0.8
        fat_weight = -0.3
        carbs_weight = -0.6
        disclosure_bonus = (disclosure_level - 1) * 3.0
        endorsement_bonus = 6.0 if endorsement == 1 else 0.0
        allergen_penalty = -10.0 if allergen_flag == 1 else 0.0
        hazard_penalty = -15.0 if hazard_flag == 1 else 0.0
        env_bonus = 4.0 * environment_focus  

        complexity_penalty = -0.5 * max(0, ingredient_complexity - 5)

        trust_score = base_trust \
            + protein_weight * (protein - 30.0) \
            + b12_weight * (b12 - 0.5) \
            + fat_weight * (fat - 10.0) \
            + carbs_weight * (carbs - 10.0) \
            + disclosure_bonus \
            + endorsement_bonus \
            + name_effect \
            + env_bonus \
            + complexity_penalty \
            + narrative_pos*5.0 - narrative_neg*4.0 \
            + allergen_penalty + hazard_penalty \
            + region_weight

        trust_score = float(np.clip(trust_score + rng.normal(0,5.0), 0.0, 100.0))

        rows.append({
            'id': i,
            'name_text': name,
            'protein': protein,
            'fat': fat,
            'carbs': carbs,
            'calories': calories,
            'b12': b12,
            'iron': iron,
            'process': process,
            'function': function,
            'ingredient_complexity': ingredient_complexity,
            'disclosure_level': disclosure_level,
            'endorsement': endorsement,
            'age': age,
            'allergen_flag': allergen_flag,
            'hazard_flag': hazard_flag,
            'narrative_neg': narrative_neg,
            'narrative_pos': narrative_pos,
            'environment_focus': environment_focus,
            'region': region,
            'region_weight': region_weight,
            'trust_score': trust_score
        })

    return pd.DataFrame(rows)


from dataclasses import dataclass
@dataclass
class PipelineInfo:
    tfidf_name: any
    scaler: any
    cat_columns: list
    name_tfidf_features: int


def fit_feature_pipeline(df, name_max_feats=50):
    
    df = df.copy()
    tfidf_name = TfidfVectorizer(max_features=name_max_feats, ngram_range=(1,2))
    name_tfidf = tfidf_name.fit_transform(df['name_text'].astype(str)).toarray()

    num_cols = [
        'protein','fat','carbs','calories','b12','iron',
        'ingredient_complexity','disclosure_level','endorsement',
        'allergen_flag','hazard_flag','narrative_neg','narrative_pos','age',
        'environment_focus', 'region_weight'
    ]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
    X_num = df[num_cols].fillna(0).astype(float).values

    cat_cols = ['process','function']
    cat_df = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)
    cat_columns = list(cat_df.columns)
    X_cat = cat_df.values if cat_df.shape[1] > 0 else np.zeros((len(df),0))

    X = np.hstack([name_tfidf, X_num, X_cat])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pipeline = PipelineInfo(
        tfidf_name=tfidf_name,
        scaler=scaler,
        cat_columns=cat_columns,
        name_tfidf_features=name_tfidf.shape[1]
    )
    return X_scaled, pipeline


def transform_with_pipeline(df, pipeline: PipelineInfo):
    
    df = df.copy()
    name_tfidf = pipeline.tfidf_name.transform(df['name_text'].astype(str)).toarray()

    num_cols = [
        'protein','fat','carbs','calories','b12','iron',
        'ingredient_complexity','disclosure_level','endorsement',
        'allergen_flag','hazard_flag','narrative_neg','narrative_pos','age',
        'environment_focus', 'region_weight'
    ]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
    X_num = df[num_cols].fillna(0).astype(float).values

    cat_cols = ['process','function']
    cat_df = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)
    if len(pipeline.cat_columns) > 0:
        cat_df = cat_df.reindex(columns=pipeline.cat_columns, fill_value=0)
        X_cat = cat_df.values
    else:
        X_cat = np.zeros((len(df),0))

    X = np.hstack([name_tfidf, X_num, X_cat])
    X_scaled = pipeline.scaler.transform(X)
    return X_scaled

def features_df_from_transformed(X_transformed, pipeline: PipelineInfo):
    return pd.DataFrame(X_transformed, columns=pipeline.feature_names)


def train_model(X, y):
    if HAS_XGBOOST:
        model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=0, verbosity=0, n_jobs=1)
        model.fit(X, y)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
        model.fit(X, y)
    return model


def predict_with_ci(model, X_sample, n_bootstraps=50, noise_sigma=3.0):
    base = model.predict(X_sample)
    preds = np.array([base + np.random.normal(0, noise_sigma, size=base.shape) for _ in range(n_bootstraps)])
    mean = preds.mean(axis=0)
    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)
    return mean, lower, upper


import seaborn as sns

def plot_trust_vs_numeric(df, numeric_col, outpath=None):

    df = df.copy()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.regplot(x=numeric_col, y='trust_score', data=df, ax=ax, scatter_kws={'alpha':0.6})
    ax.set_title(f'Trust score vs {numeric_col}')
    ax.set_ylabel('Trust score')
    ax.set_xlabel(numeric_col)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
    return fig

def plot_trust_vs_endorsement(df, outpath=None):

    df = df.copy()
    fig, ax = plt.subplots(figsize=(5,4))
    sns.boxplot(x='endorsement', y='trust_score', data=df, palette='Set2', ax=ax)
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_title('Trust score vs Endorsement')
    ax.set_xlabel('Endorsement')
    ax.set_ylabel('Trust score')
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
    return fig


def plot_trust_vs_environment(df, outpath=None):

    df = df.copy()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.regplot(x='environment_focus', y='trust_score', data=df, scatter_kws={'alpha':0.6}, ax=ax)
    ax.set_title('Trust score vs Environment Focus')
    ax.set_xlabel('Environment Focus (0 low → 3 high)')
    ax.set_ylabel('Trust score')
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
    return fig


def plot_trust_vs_disclosure(df, outpath=None):
    df = df.copy()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.regplot(x='disclosure_level', y='trust_score', data=df, scatter_kws={'alpha':0.6}, ax=ax)
    ax.set_title('Trust score vs Disclosure Level')
    ax.set_xlabel('Disclosure Level (0 none → 3 full)')
    ax.set_ylabel('Trust score')
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
    return fig

def plot_trust_by_region_bar(df, outpath=None):
    df = df.copy()
    if 'region' not in df.columns:
        raise ValueError("DataFrame must contain a 'region' column")
    avg = df.groupby('region')['trust_score'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=avg.values, y=avg.index, ax=ax)
    ax.set_xlabel('Average Trust score')
    ax.set_ylabel('Region')
    ax.set_title('Average Trust score by Region')
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
    return fig


def run_tests():
    logging.info('Running tests...')
    df = generate_synthetic_pf_data(n=300, random_seed=1)
    X, pipeline = fit_feature_pipeline(df)
    assert X.shape[0] == df.shape[0]
    assert isinstance(pipeline, PipelineInfo)

    X_train, X_test, y_train, y_test = train_test_split(X, df['trust_score'].values, test_size=0.2, random_state=0)
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]
    assert np.all(np.isfinite(preds))

    sample_df = df.sample(3, random_state=0)
    X_sample = transform_with_pipeline(sample_df, pipeline)
    assert X_sample.shape[0] == 3

    mean, lo, hi = predict_with_ci(model, X_sample, n_bootstraps=10)
    assert mean.shape[0] == 3
    assert np.all(lo <= hi)

    logging.info('All tests passed')



def run_cli_demo(save_artifacts=True):
    logging.info('Running CLI demo (no Streamlit). Generating synthetic data and training model...')
    df = generate_synthetic_pf_data(n=1000)
    X, pipeline = fit_feature_pipeline(df)
    y = df['trust_score'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = train_model(X_train, y_train)

    preds_test = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds_test, squared=False)
    r2 = r2_score(y_test, preds_test)

    logging.info(f'Test RMSE: {rmse:.3f}')
    logging.info(f'Test R2  : {r2:.3f}')

    if save_artifacts and HAS_JOBLIB:
        joblib.dump({'model': model, 'pipeline': pipeline}, 'pf_trust_model_no_sensory.joblib')
        logging.info('Saved model+pipeline to pf_trust_model_no_sensory.joblib')
    elif save_artifacts:
        logging.info('joblib not available — skipping model save')

    example = pd.DataFrame([{
        'name_text': 'precision-fermentation-derived protein',
        'protein': 45.0,
        'fat': 10.0,
        'carbs': 5.0,
        'calories': 45.0*4 + 10*9 + 5*4,
        'b12': 1.5,
        'iron': 2.0,
        'function': 'protein',
        'process': 'refined',
        'ingredient_complexity': 4,
        'disclosure_level': 2,
        'endorsement': 1,
        'age': 35,
        'allergen_flag': 0,
        'hazard_flag': 0,
        'narrative_neg': 0.1,
        'narrative_pos': 0.2,
        'environment_focus': 0.8
    }])

    X_ex = transform_with_pipeline(example, pipeline)
    mean, lo, hi = predict_with_ci(model, X_ex, n_bootstraps=200)
    logging.info(f'Example predicted Trust score: {mean[0]:.2f} (95% CI [{lo[0]:.2f}, {hi[0]:.2f}])')



def run_streamlit_app():
    st.set_page_config(page_title='FERMENT-IQ', layout='wide')
    st.title('FERMENT-IQ: Brewing Trust, One Byte at a Time!')

    st.markdown("Explore how labelling, nutrition, transparency, endorsements, safety, and environmental focus shape predicted consumer trust in novel food ingredients. Adjust factors and see real-time trust scores with correlation plots.")

    @st.cache_resource
    def load_train_cached():
        df_local = generate_synthetic_pf_data(n=1800)
        X_local, pipeline_local = fit_feature_pipeline(df_local)
        y_local = df_local['trust_score'].values
        model_local = train_model(X_local, y_local)
        return df_local, model_local, pipeline_local

    df_local, model_local, pipeline_local = load_train_cached()

    st.sidebar.header('Scenario builder')
    st.sidebar.subheader('Labeling (Front and Back of the pack)')
    name_input_multi = st.sidebar.multiselect(
        'Label naming (choose one or more labels shown on pack)',
        options=naming_options,
        default=[naming_options[0]]
    )
    if len(name_input_multi) == 0:
        name_text_for_model = naming_options[0]
    else:
        name_text_for_model = ' '.join(name_input_multi)

    st.sidebar.subheader('Nutrition')
    protein_input = st.sidebar.number_input('Protein (g/100g)', min_value=0.0, max_value=200.0, value=45.0)
    b12_input = st.sidebar.number_input('B12 (µg/100g)', min_value=0.0, max_value=100.0, value=1.5)
    carbs_input = st.sidebar.number_input('Carbs (g/100g)', min_value=0.0, max_value=200.0, value=5.0)
    fat_input = st.sidebar.number_input('Fat (g/100g)', min_value=0.0, max_value=200.0, value=10.0)

    st.sidebar.subheader('Good for the Earth / You')
    disclosure_input = st.sidebar.slider('Disclosure level (0 none -> 3 full)', 0, 3, 2)
    endorsement_input = st.sidebar.checkbox('Has endorsement (e.g., NGO/regulatory)', value=False)
    allergen_input = st.sidebar.checkbox('Allergen flag', value=False)
    hazard_input = st.sidebar.checkbox('Hazard flag', value=False)
    environment_input = st.sidebar.slider('Environment focus (0 none -> 3 high)', 0, 3, 2)

   
    st.sidebar.subheader('Region')
    region_input = st.sidebar.selectbox('Region', options=regions, index=0)

    age_input = st.sidebar.slider('Representative age', 18, 80, 35)
    process_input = st.sidebar.selectbox('Process', options=['refined','minimally processed','heat-treated','encapsulated'])
    function_input = st.sidebar.selectbox('Function', options=['protein','fat','flavor_enhancer','texturizer','emulsifier'])


    st.sidebar.subheader('Affordability (Cost parity)')
    affordability_input = st.sidebar.slider(
        'Affordability (Cost parity)',
        min_value=-10,
        max_value=100,
        value=0,
        step=1,
        format='%d%%'
    )

    if st.sidebar.button('Predict trust score'):
        row = {
            'name_text': name_text_for_model,
            'protein': protein_input,
            'fat': fat_input,
            'carbs': carbs_input,
            'calories': protein_input*4 + fat_input*9 + carbs_input*4,
            'b12': b12_input,
            'iron': 2.0,
            'ingredient_complexity': 4,
            'function': function_input,
            'process': process_input,
            'disclosure_level': disclosure_input,
            'endorsement': 1 if endorsement_input else 0,
            'age': age_input,
            'allergen_flag': 1 if allergen_input else 0,
            'hazard_flag': 1 if hazard_input else 0,
            'narrative_neg': 0.1,
            'narrative_pos': 0.1,
            'environment_focus': environment_input,
            'region': region_input,
            'region_weight': float(REGION_WEIGHTS.get(region_input, 0.0))
        }
        input_df = pd.DataFrame([row])
        X_input = transform_with_pipeline(input_df, pipeline_local)
        mean, lo, hi = predict_with_ci(model_local, X_input.reshape(1, -1), n_bootstraps=200)

        st.metric(label='Predicted Trust & Acceptance (0-100)', value=f'{mean[0]:.1f}', delta=f'{mean[0]-50:+.1f}')
        st.write(f'95% CI: [{lo[0]:.1f}, {hi[0]:.1f}]')

    st.markdown('---')
    X_all, _ = fit_feature_pipeline(df_local)
    X_train, X_test, y_train, y_test = train_test_split(X_all, df_local['trust_score'].values, test_size=0.2, random_state=0)
    model_eval = train_model(X_train, y_train)
    preds = model_eval.predict(X_test)
    #st.write(f'Test RMSE: {mean_squared_error(y_test,preds,squared=False):.2f}  |  Test R2: {r2_score(y_test,preds):.2f}')

    st.subheader('Trust vs Nutrients')
    fig_protein = plot_trust_vs_numeric(df_local, 'protein')
    st.pyplot(fig_protein)

    fig_carbs = plot_trust_vs_numeric(df_local, 'carbs')
    st.pyplot(fig_carbs)

    st.subheader('Trust vs Endorsement')
    fig_endorse = plot_trust_vs_endorsement(df_local)
    st.pyplot(fig_endorse)

    st.subheader('Trust vs Environment Focus')
    fig_env = plot_trust_vs_environment(df_local)
    st.pyplot(fig_env)

    st.subheader('Trust vs Disclosure Level')
    fig_disc = plot_trust_vs_disclosure(df_local)
    st.pyplot(fig_disc)

    st.subheader('Trust vs Region')
    fig_reg_bar = plot_trust_by_region_bar(df_local)
    st.pyplot(fig_reg_bar)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PF Trust & Acceptance starter')
    parser.add_argument('--test', action='store_true', help='Run built-in tests')
    parser.add_argument('--cli', action='store_true', help='Run CLI demo (no UI)')
    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    if STREAMLIT_AVAILABLE and not args.cli:
        run_streamlit_app()
    else:
        if not STREAMLIT_AVAILABLE:
            logging.info('Streamlit not present. Running CLI demo instead. To use the web UI install streamlit and run `streamlit run <file>`')
        run_cli_demo(save_artifacts=True)
