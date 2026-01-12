import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Set page config
st.set_page_config(page_title="School Completion Analysis", layout="wide")

# --- BIG TITLE ---
st.markdown("<h1 style='text-align: center;'>School Completion Prediction Analysis</h1>", unsafe_allow_html=True)
st.divider()

# --- DATA LOADING & CLEANING ---
@st.cache_data
def load_and_clean_data():
    if not os.path.exists("final_data.csv"):
        st.error("File 'final_data.csv' not found. Please ensure it is in your GitHub repository.")
        return pd.DataFrame()
    
    df = pd.read_csv("final_data.csv")
    df = df[df['stage'] != 'post_secondary']
    df = df[df['students'] > 0]
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    return df

df = load_and_clean_data()

if not df.empty:
    # --- MODEL TRAINING ---
    @st.cache_resource
    def train_models(df):
        target = "completion"
        X = df.drop(columns=[target, 'date'], errors='ignore')
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        num_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X_train.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

        model_definitions = {
            "Multiple Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "K-NN": KNeighborsRegressor(n_neighbors=5),
            "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
        }

        models = {}
        results = []

        for name, regressor in model_definitions.items():
            pipe = Pipeline([('prep', preprocessor), ('reg', regressor)])
            pipe.fit(X_train, y_train)
            
            tr_pred = pipe.predict(X_train)
            te_pred = pipe.predict(X_test)
            
            results.append({
                "Model": name,
                "Train R2": r2_score(y_train, tr_pred),
                "Test R2": r2_score(y_test, te_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, te_pred)),
                "MAE": mean_absolute_error(y_test, te_pred),
                "Overfit Gap": r2_score(y_train, tr_pred) - r2_score(y_test, te_pred),
                "Predictions": te_pred.flatten() # Flatten to ensure 1D
            })
            models[name] = {"pipeline": pipe}

        return models, pd.DataFrame(results).set_index("Model"), y_test

    models_dict, results_df, y_test = train_models(df)
    best_model_name = results_df['Test R2'].idxmax()

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Model Performance", "🔮 Prediction"])

    with tab1:
        st.header("Education Completion Storyboard")
        stage_order = ['primary', 'secondary_lower', 'secondary_upper']
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. The Education Journey")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x='stage', y='completion', order=stage_order, hue='stage', palette='Set2', legend=False, ax=ax1)
            st.pyplot(fig1)

            st.subheader("3. Income Impact per Level")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            for s in stage_order:
                subset = df[df['stage'] == s]
                sns.regplot(data=subset, x='income_mean', y='completion', label=s, ax=ax3, scatter_kws={'alpha':0.3})
            ax3.legend()
            st.pyplot(fig3)

        with col2:
            st.subheader("2. Regional Completion Gaps")
            heatmap_data = df.groupby(['state', 'stage'])['completion'].mean().unstack()[stage_order]
            fig2, ax2 = plt.subplots(figsize=(10, 11))
            sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn", center=95, ax=ax2)
            st.pyplot(fig2)

            st.subheader("4. Completion Progress Over Time")
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df, x='year', y='completion', hue='stage', markers=True, ax=ax4)
            st.pyplot(fig4)

    with tab2:
        st.header("Machine Learning Performance Analysis")
        
        # --- Interactive Table Section ---
        comparison_options = ["Show All Models"] + list(results_df.index)
        selected_view = st.selectbox("Select Model to Filter Comparison Table", comparison_options)
        
        # FIX for ArrowInvalid: Drop 'Predictions' (array column) and force metrics to float64
        display_df = results_df.drop(columns=['Predictions']).astype(float)
        
        if selected_view == "Show All Models":
            st.subheader("Model Comparison Metrics")
            st.table(display_df.style.highlight_max(subset=['Test R2'], color='#90ee90').format("{:.4f}"))
        else:
            st.subheader(f"Metrics for {selected_view}")
            st.table(display_df.loc[[selected_view]].style.format("{:.4f}"))

        st.divider()

        # --- 1. RMSE and MAE Comparison ---
        st.subheader("1. Error Metrics: RMSE and MAE Comparison")
        fig_err, ax_err = plt.subplots(figsize=(9, 5))
        models_names = results_df.index
        x_pos = np.arange(len(models_names))
        width = 0.35
        ax_err.bar(x_pos - width/2, results_df['RMSE'], width, label='RMSE', color='#ff7f0e')
        ax_err.bar(x_pos + width/2, results_df['MAE'], width, label='MAE', color='#1f77b4')
        ax_err.set_xlabel('Model')
        ax_err.set_ylabel('Error Value')
        ax_err.set_xticks(x_pos)
        ax_err.set_xticklabels(models_names, rotation=45)
        ax_err.legend()
        ax_err.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_err)

        # --- 2. R2 Score Comparison ---
        st.subheader("2. R² Score Comparison Across Models")
        fig_r2, ax_r2 = plt.subplots(figsize=(8, 5))
        ax_r2.bar(results_df.index, results_df['Test R2'], color='#2ca02c')
        ax_r2.set_ylabel('R² Score')
        ax_r2.set_xlabel('Model')
        plt.xticks(rotation=45)
        ax_r2.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_r2)

        # --- 3. Residual Comparison ---
        st.subheader("3. Residual Comparison (Error Distribution)")
        res_models = ['Multiple Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
        available_res_models = [m for m in res_models if m in results_df.index]
        
        # Ensure residuals are forced to float to avoid dtype mixing
        residuals_data = {}
        for m in available_res_models:
            res = (y_test.values - results_df.loc[m, 'Predictions']).astype(float)
            residuals_data[m] = res
            
        residuals_df_plot = pd.DataFrame(residuals_data)
        
        fig_res, ax_res = plt.subplots(figsize=(9, 5))
        residuals_df_plot.boxplot(ax=ax_res)
        ax_res.axhline(0, linestyle='--', color='red')
        ax_res.set_ylabel('Residuals')
        ax_res.set_title('Residual Comparison Across Models')
        st.pyplot(fig_res)

        # --- 4. Feature Influence ---
        st.subheader("4. Relative Influence of Socioeconomic Factors")
        features_num = ['students', 'income_mean', 'poverty_absolute', 'poverty_hardcore', 'poverty_relative']
        X_inf = df[features_num]
        y_inf = df['completion']
        scaler = StandardScaler()
        X_inf_scaled = scaler.fit_transform(X_inf)
        lr_inf = LinearRegression()
        lr_inf.fit(X_inf_scaled, y_inf)
        
        coef_df = pd.DataFrame({'Feature': features_num, 'Coefficient': lr_inf.coef_})
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=True)
        
        fig_inf, ax_inf = plt.subplots(figsize=(8, 5))
        ax_inf.barh(coef_df['Feature'], coef_df['Abs_Coefficient'], color='#9467bd')
        ax_inf.set_xlabel('Standardized Influence on Completion Rate')
        st.pyplot(fig_inf)

    with tab3:
        st.header("Predict School Completion Rate")
        st.write(f"Powered by: **{best_model_name}**")

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                state_in = st.selectbox("State", sorted(df['state'].unique()))
                stage_in = st.selectbox("Education Level", ['primary', 'secondary_lower', 'secondary_upper'])
            with c2:
                year_in = st.slider("Target Year", 2024, 2030, 2025)
                students_in = st.number_input("Number of Students", min_value=1, value=5000)
            with c3:
                income_in = st.number_input("Mean Income", value=float(df['income_mean'].mean()))
                p_abs = st.number_input("Poverty Absolute (%)", value=float(df['poverty_absolute'].mean()))
            
            c4, c5 = st.columns(2)
            with c4:
                p_hard = st.number_input("Poverty Hardcore (%)", value=float(df['poverty_hardcore'].mean()))
            with c5:
                p_rel = st.number_input("Poverty Relative (%)", value=float(df['poverty_relative'].mean()))

            predict_btn = st.form_submit_button("Generate Prediction")

        if predict_btn:
            input_data = pd.DataFrame({
                'state': [state_in], 'stage': [stage_in], 'students': [students_in],
                'income_mean': [income_in], 'poverty_absolute': [p_abs],
                'poverty_hardcore': [p_hard], 'poverty_relative': [p_rel], 'year': [year_in]
            })

            winner_pipe = models_dict[best_model_name]["pipeline"]
            try:
                prediction_raw = winner_pipe.predict(input_data)
                # Force standard Python float to prevent progress bar errors
                prediction_val = float(prediction_raw[0])
                
                st.success(f"### Predicted Completion Rate: {prediction_val:.2f}%")
                # Normalize progress to 0.0 - 1.0 range
                progress_val = float(min(max(prediction_val / 100, 0.0), 1.0))
                st.progress(progress_val)
            except Exception as e:
                st.error(f"Prediction failed. Error: {e}")