import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

st.set_page_config(page_title="Regresi Berganda", layout="wide")

# ---------- HEADER ----------
st.markdown("## 📊 Aplikasi Analisis Regresi Berganda")
st.markdown("""
Unggah file CSV yang berisi data numerik, lalu pilih variabel dependen dan independennya.  
Aplikasi ini juga menyertakan uji asumsi dan visualisasi hasil regresi.
""")

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Pengaturan")

uploaded_file = st.sidebar.file_uploader("📥 Upload File CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Pratinjau Data")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    dep_var = st.sidebar.selectbox("🎯 Pilih Variabel Dependen (Y)", numeric_cols)
    indep_vars = st.sidebar.multiselect("📈 Pilih Variabel Independen (X)", 
                                        [col for col in numeric_cols if col != dep_var])

    if dep_var and indep_vars:
        X = df[indep_vars]
        y = df[dep_var]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        st.subheader("📋 Hasil Regresi")
        st.text(model.summary())

        # ---------- PLOT RESIDUAL ----------
        st.subheader("📊 Visualisasi Diagnostik")

        fig1, ax1 = plt.subplots()
        sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax1,
                      line_kws={'color': 'red', 'lw': 1})
        ax1.set_xlabel("Fitted Values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residual vs Fitted")
        st.pyplot(fig1)

        # ---------- QQ PLOT ----------
        fig2 = sm.qqplot(model.resid, line='45')
        st.pyplot(fig2.figure)

        # ---------- Histogram Residual ----------
        st.subheader("📈 Histogram Residual")
        fig3, ax3 = plt.subplots()
        sns.histplot(model.resid, kde=True, ax=ax3)
        ax3.set_title("Distribusi Residual")
        st.pyplot(fig3)

        # ---------- Uji Asumsi ----------
        st.subheader("🧪 Uji Asumsi Regresi")

        st.markdown("### 1. Uji Normalitas (Shapiro-Wilk)")
        stat, p = stats.shapiro(model.resid)
        st.write(f"**Statistic** = {stat:.4f}, **p-value** = {p:.4f}")
        if p > 0.05:
            st.success("Residual terdistribusi normal (p > 0.05)")
        else:
            st.error("Residual tidak terdistribusi normal (p ≤ 0.05)")

        st.markdown("### 2. Uji Homoskedastisitas (Plot Visual)")
        st.markdown("Lihat grafik Residual vs Fitted di atas. Jika pola acak, maka homoskedastisitas terpenuhi.")

        st.markdown("### 3. Uji Multikolinearitas (VIF)")
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif_df = pd.DataFrame()
        vif_df["Variabel"] = X.columns
        vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        st.dataframe(vif_df)

    else:
        st.warning("⚠️ Silakan pilih variabel dependen dan minimal satu variabel independen.")
else:
    st.info("💡 Upload file CSV terlebih dahulu untuk memulai.")
