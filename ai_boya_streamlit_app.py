
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args

st.set_page_config(page_title="AI Boya Formül Önerici", layout="centered")

st.title("🎨 AI Destekli Boya Bileşeni Önerici")
st.markdown("Hedef LAB renk değerlerini girin, sistem size en uygun 6 bileşeni ve yüzdelerini önersin.")

# 1. Kullanıcıdan renk girdisi al
l_val = st.number_input("L değeri", min_value=0.0, max_value=100.0, value=90.0)
a_val = st.number_input("a değeri", min_value=-128.0, max_value=127.0, value=-1.0)
b_val = st.number_input("b değeri", min_value=-128.0, max_value=127.0, value=1.0)

uploaded_file = st.file_uploader("🎯 Deney verisi Excel dosyasını yükleyin (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    component_cols = ['Bileşen1', 'Bileşen2', 'Bileşen3', 'Bileşen4', 'Bileşen5', 'Bileşen6']
    amount_cols = ['%miktar1', '%miktar2', '%miktar3', '%miktar4', '%miktar5', '%miktar6']
    target_cols = ['L', 'a', 'b']

    df_full = df[component_cols + amount_cols + target_cols].dropna()

    # Model eğitimi
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    components_encoded = encoder.fit_transform(df_full[component_cols])
    X_full = np.hstack([components_encoded, df_full[amount_cols].values])
    y_full = df_full[target_cols].values

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_full)
    y_scaled = y_scaler.fit_transform(y_full)

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_scaled, y_scaled)

    # Optimizasyon
    target_color = np.array([l_val, a_val, b_val])
    target_color_scaled = y_scaler.transform(target_color.reshape(1, -1))[0]

    space = []
    for i, categories in enumerate(encoder.categories_):
        space.append(Categorical(categories.tolist(), name=f"Bileşen{i+1}"))
    for i in range(6):
        space.append(Real(0.0, 1.0, name=f"%miktar{i+1}"))

    @use_named_args(space)
    def objective(**params):
        components = [params[f"Bileşen{i+1}"] for i in range(6)]
        amounts = [params[f"%miktar{i+1}"] for i in range(6)]
        components_encoded = encoder.transform([components])
        x_input = np.hstack([components_encoded[0], amounts])
        x_scaled = X_scaler.transform([x_input])
        y_pred_scaled = model.predict(x_scaled)[0]
        return np.linalg.norm(y_pred_scaled - target_color_scaled)

    if st.button("🔍 En uygun formülü öner"):
        st.info("AI önerisi hesaplanıyor, lütfen bekleyin...")
        res = gp_minimize(objective, dimensions=space, n_calls=40, random_state=42)

        best_params = res.x
        best_components = best_params[:6]
        best_amounts = best_params[6:]

        x_input = np.hstack([encoder.transform([best_components])[0], best_amounts])
        x_scaled = X_scaler.transform([x_input])
        predicted_scaled = model.predict(x_scaled)
        predicted_lab = y_scaler.inverse_transform(predicted_scaled)[0]

        st.success("✅ AI önerisi tamamlandı:")
        for i in range(6):
            st.write(f"Bileşen{i+1}: `{best_components[i]}`,  %miktar{i+1}: `{best_amounts[i]:.3f}`")

        st.markdown("### 🎨 Tahmini Renk Değeri")
        st.write(f"L: `{predicted_lab[0]:.2f}`, a: `{predicted_lab[1]:.2f}`, b: `{predicted_lab[2]:.2f}`")
