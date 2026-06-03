import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Carregando dados tratados para modelagem...")
# 1. Ler os dados limpos
try:
    df = pd.read_csv('autos_tratado.csv')
except FileNotFoundError:
    print("\n========================================================")
    print("ERRO: O arquivo 'autos_tratado.csv' nao existe!")
    print("Por favor, rode o script 'tratamento_dados.py' primeiro.")
    print("========================================================\n")
    exit()

# 2. Separar variaveis preditoras (X) e a variavel alvo (y)
X = df.drop(columns=['gearbox'])
y = df['gearbox']

# 3. Divisao de Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Padronizacao (Essencial para modelos lineares como a Regressao Logistica)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# MODELO 1: REGRESSAO LOGISTICA
# ==========================================
print("\nTreinando Regressao Logistica...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

print("\n==== RESULTADOS: REGRESSAO LOGISTICA ====")
print(f"Acuracia: {accuracy_score(y_test, y_pred_lr):.4f}")
print("Matriz de Confusao:")
print(confusion_matrix(y_test, y_pred_lr))
print("Relatorio Tecnico:")
print(classification_report(y_test, y_pred_lr, target_names=['automatik', 'manuell']))

# ==========================================
# MODELO 2: ARVORE DE DECISAO
# ==========================================
print("\nTreinando Arvore de Decisao...")
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\n==== RESULTADOS: ARVORE DE DECISAO ====")
print(f"Acuracia: {accuracy_score(y_test, y_pred_dt):.4f}")
print("Matriz de Confusao:")
print(confusion_matrix(y_test, y_pred_dt))
print("Relatorio Tecnico:")
print(classification_report(y_test, y_pred_dt, target_names=['automatik', 'manuell']))
