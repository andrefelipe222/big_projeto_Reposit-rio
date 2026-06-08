import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
# ================================
# FUNDO DE VÍDEO
# ================================
st.markdown(
    """
    <video autoplay muted loop id="bgvideo">
        <source src="https://raw.githubusercontent.com/andrefelipe222/big_projeto_Reposit-rio/main/intro.mp4" type="video/mp4">
    </video>
    <style>
    #bgvideo {
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        z-index: -1;
    }
    .stApp {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ================================
# CONFIGURAÇÃO INICIAL
# ================================
st.set_page_config(page_title="Dashboard Autos", layout="wide")
st.title("🚗 Dashboard Interativo - Projeto Autos")

# ================================
# CARREGAR DADOS COM TRATAMENTO DE NaN
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("autos_tratado.csv")

    # Substituir NaN em colunas numéricas pela média
    num_cols = df.select_dtypes(include=['float64','int64']).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    # Substituir NaN em colunas categóricas pela moda
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            moda = df[col].mode()
            if len(moda) > 0:
                df[col] = df[col].fillna(moda[0])
            else:
                df[col] = df[col].fillna("desconhecido")

    return df

df = load_data()

# ================================
# SIDEBAR
# ================================
st.sidebar.title("📌 Navegação")
section = st.sidebar.radio("Escolha a seção:", ["Análise dos Dados", "Bayes", "Machine Learning"])

# ================================
# SEÇÃO 1 - ANÁLISE DOS DADOS
# ================================
if section == "Análise dos Dados":
    st.header("📊 Seção 1 - Análise Exploratória dos Dados")

    tab1, tab2, tab3, tab4 = st.tabs(["Câmbio", "Combustível", "Potência", "Tipo de Veículo"])

    with tab1:
        st.subheader("Distribuição de Tipo de Câmbio")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="gearbox", palette="Set2", ax=ax)
        st.pyplot(fig)

    with tab2:
        st.subheader("Distribuição de Tipo de Combustível (Top 3)")
        top_fuels = df['fuelType'].value_counts().head(3).index
        fig, ax = plt.subplots()
        sns.countplot(data=df[df['fuelType'].isin(top_fuels)], x="fuelType", palette="viridis", ax=ax)
        st.pyplot(fig)

    with tab3:
        st.subheader("Relação entre Potência e Câmbio")
        fig, ax = plt.subplots()
        sns.boxplot(data=df[df['powerPS'] <= 400], x="gearbox", y="powerPS", palette="Set3", ax=ax)
        st.pyplot(fig)

    with tab4:
        st.subheader("Proporção de Câmbio por Tipo de Veículo")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="vehicleType", hue="gearbox", multiple="stack", palette="Set1", shrink=0.8, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ================================
# SEÇÃO 2 - BAYES
# ================================
elif section == "Bayes":
    st.header("🧮 Seção 2 - Classificação Probabilística (Bayes)")

    vehicleType = st.selectbox("Tipo de Veículo", df['vehicleType'].unique())
    brand = st.selectbox("Marca", df['brand'].unique())
    powerPS = st.slider("Potência (PS)", 10, 400, 100)
    year = st.slider("Ano de Registro", 1950, 2016, 2010)
    month = st.slider("Mês de Registro", 1, 12, 6)

    def mostrar_probabilidades(posteriores):
        df_probs = pd.DataFrame.from_dict(posteriores, orient="index", columns=["Probabilidade"])
        df_probs["Probabilidade"] = (df_probs["Probabilidade"] * 100).round(1)
        st.bar_chart(df_probs)
        for classe, prob in df_probs["Probabilidade"].items():
            st.metric(label=f"Classe: {classe}", value=f"{prob:.1f}%")

    # Bayes 1 - fuelType
    st.subheader("🔹 Previsão de Tipo de Combustível")
    alvo = "fuelType"
    priors = df[alvo].value_counts(normalize=True)
    posteriores = {}
    for classe in df[alvo].unique():
        prob = priors[classe]
        for coluna, valor in {"vehicleType": vehicleType, "brand": brand}.items():
            prob *= (df[df[alvo] == classe][coluna] == valor).mean()
        posteriores[str(classe)] = prob
    soma = sum(posteriores.values())
    for classe in posteriores:
        posteriores[classe] /= soma
    mostrar_probabilidades(posteriores)

    # Bayes 2 - gearbox
    st.subheader("🔹 Previsão de Tipo de Câmbio")
    df['powerPS_cat'] = pd.cut(df['powerPS'], bins=[0,100,200,400,1000], labels=['baixo','medio','alto','muito_alto'])
    alvo = "gearbox"
    priors = df[alvo].value_counts(normalize=True)
    posteriores = {}
    for classe in df[alvo].unique():
        prob = priors[classe]
        for coluna, valor in {"vehicleType": vehicleType, "powerPS_cat": 'medio'}.items():
            prob *= (df[df[alvo] == classe][coluna] == valor).mean()
        posteriores[str(classe)] = prob
    soma = sum(posteriores.values())
    for classe in posteriores:
        posteriores[classe] /= soma
    mostrar_probabilidades(posteriores)

    # Bayes 3 - price_cat
    st.subheader("🔹 Previsão de Faixa de Preço")
    df['price_cat'] = pd.cut(df['price'], bins=[0,2000,10000,50000,100000], labels=['baixo','medio','alto','muito_alto'])
    df = df.dropna(subset=['price_cat'])
    alvo = "price_cat"
    priors = df[alvo].value_counts(normalize=True)
    posteriores = {}
    for classe in df[alvo].unique():
        prob = priors[classe]
        for coluna, valor in {"yearOfRegistration": year, "monthOfRegistration": month}.items():
            prob *= (df[df[alvo] == classe][coluna] == valor).mean()
        posteriores[str(classe)] = prob
    soma = sum(posteriores.values())
    for classe in posteriores:
        posteriores[classe] /= soma
    mostrar_probabilidades(posteriores)

# ================================
# SEÇÃO 3 - MACHINE LEARNING
# ================================
elif section == "Machine Learning":
    st.header("🤖 Seção 3 - Predições com ML")

    # Inputs do usuário também aqui
    vehicleType = st.selectbox("Tipo de Veículo", df['vehicleType'].unique())
    brand = st.selectbox("Marca", df['brand'].unique())
    powerPS = st.slider("Potência (PS)", 10, 400, 100)
    year = st.slider("Ano de Registro", 1950, 2016, 2010)
    month = st.slider("Mês de Registro", 1, 12, 6)

    X = df.drop(columns=['gearbox'])
    y = df['gearbox']

    # One-Hot Encoding
    X = pd.get_dummies(X, drop_first=True)

    # Imputação de NaN apenas para numéricos
    num_cols = X.select_dtypes(include=['float64','int64']).columns
    imputer = SimpleImputer(strategy="mean")
    X[num_cols] = imputer.fit_transform(X[num_cols])

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinar modelos
    lr_model = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42).fit(X_train, y_train)

    # Preparar novo dado
    novo_df = pd.DataFrame([{"vehicleType": vehicleType, "brand": brand, "powerPS": powerPS,
                             "yearOfRegistration": year, "monthOfRegistration": month}])
    novo_df = pd.get_dummies(novo_df)
    novo_df = novo_df.reindex(columns=X.columns, fill_value=0)
    novo_df[num_cols] = imputer.transform(novo_df[num_cols])

    pred_lr = lr_model.predict(scaler.transform(novo_df))[0]
    pred_dt = dt_model.predict(novo_df)[0]

    # Mapear classes para nomes originais
    label_map = {0: "Automatico", 1: "Manual"}

    # Mostrar resultados com métricas
    col1, col2 = st.columns(2)
    col1.metric("Regressão Logística", label_map.get(pred_lr, pred_lr))
    col2.metric("Árvore de Decisão", label_map.get(pred_dt, pred_dt))

    # Mostrar probabilidades formatadas
    st.subheader("Probabilidades")
    proba_lr = lr_model.predict_proba(scaler.transform(novo_df))[0]
    df_lr = pd.DataFrame((proba_lr * 100).round(1), index=lr_model.classes_, columns=["Probabilidade (%)"])
    st.write("🔹 Regressão Logística")
    st.bar_chart(df_lr)

    proba_dt = dt_model.predict_proba(novo_df)[0]
    df_dt = pd.DataFrame((proba_dt * 100).round(1), index=dt_model.classes_, columns=["Probabilidade (%)"])
    st.write("🔹 Árvore de Decisão")
    st.bar_chart(df_dt)
