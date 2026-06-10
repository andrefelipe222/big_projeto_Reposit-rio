# 🚗 Projeto de Análise Estatística e Classificação — Dataset Autos (eBay)

Este repositório contém o desenvolvimento de um pipeline completo de Ciência de Dados para a limpeza, modelagem preditiva e visualização interativa utilizando o dataset de automóveis do eBay Alemanha (*autos.csv*). O objetivo principal é prever a variável alvo **gearbox** (tipo de câmbio: manual ou automático) com base nas características estruturais e comerciais dos veículos.

---

## 📂 Estrutura do Repositório

* `autos.csv`: Base de dados bruta original (removida do rastreamento do Git por tamanho).
* `tratamento_dados.py`: Script responsável pelo carregamento, filtragem de outliers, tratamento de nulos e encoding categórico.
* `autos_tratado.csv`: Base de dados higienizada resultante do tratamento.
* `classificacao.py`: Script de Machine Learning (Regressão Logística e Árvore de Decisão) com extração de métricas de validação.
* `Probabilidade de bayes.ipynb`: Jupyter Notebook contendo a modelagem estatística através do Teorema de Bayes.
* `app.py`: Interface gráfica e dashboard interativo desenvolvido em Streamlit.
* `intro.mp4`: Arquivo de vídeo utilizado como plano de fundo do dashboard interativo.
* `.gitignore`: Arquivo de configuração para ignorar dados pesados e o ambiente virtual (`.venv/`).

---

## 🛠️ 1. Escolha do Dataset e Origem dos Dados
O conjunto de dados utilizado mapeia anúncios de carros usados na plataforma eBay Alemanha. Ele foi obtido diretamente via Kaggle:
🔗 [Kaggle - Uncovering the Factors That Affect Used Car Prices](https://www.kaggle.com/code/amalsalilan/uncovering-the-factors-that-affect-used-car-prices/input?select=autos.csv)

---

## 🧼 2. Tratamento e Limpeza dos Dados (`tratamento_dados.py`)
Para garantir a qualidade estatística das inferências, foram executadas as seguintes etapas de higienização automatizada:
* **Redução de Dimensionalidade:** Remoção de colunas geradas pelo web crawler sem correlação causal com a mecânica do veículo (`index`, `dateCrawled`, `name`, `postalCode`, etc.).
* **Filtro de Outliers Críticos:** Correção de erros humanos de digitação limitando os dados a intervalos comerciais reais:
    * Ano de registro válido: Entre 1950 e 2016.
    * Preço comercial: Entre € 100 e € 150.000.
    * Potência do motor (`powerPS`): Entre 10 e 1.000 CV.
* **Tratamento de Dados Ausentes:** Exclusão de registros com a variável alvo nula e imputação estratégica da categoria `'unspecified'` para variáveis categóricas secundárias.
* **Codificação Categórica (Encoding):** Conversão de strings de texto em identificadores numéricos com *LabelEncoder*, preparando os dados para processamento matemático.

> 📊 **Impacto Quantitativo:** A base bruta foi reduzida de **371.528 linhas** para **306.066 linhas limpas**, atestando o alto nível de ruído inicial do conjunto de dados.

---

## 🤖 3. Modelagem Preditiva e Algoritmos de Classificação

O projeto implementou três abordagens independentes e complementares para a classificação do tipo de câmbio (`gearbox`):

### A. Modelos Supervisionados (`classificacao.py`)
A base tratada foi dividida na proporção clássica de **80% para treinamento** e **20% para teste**. Os atributos preditores numéricos foram normalizados via `StandardScaler` (Z-score) para otimização do modelo linear.
1.  **Regressão Logística:** Modelo linear estruturado para classificação probabilística binomial.
2.  **Árvore de Decisão:** Modelo não-linear baseado em regras de quebras sucessivas por entropia, limitado a uma profundidade máxima de 10 níveis para evitar *overfitting*.

### B. Abordagem Bayesiana (`Probabilidade de bayes.ipynb`)
Implementação do **Teorema de Bayes** para o cálculo probabilístico *a priori* e *a posteriori*, computando como as propriedades combinadas do veículo (ex: ano de registro e mês) afetam a distribuição de probabilidade das classes de transmissão.

---

## 📈 4. Análise Comparativa e Métricas de Desempenho

A avaliação dos modelos gerada no ambiente de testes revelou os seguintes resultados técnicos:

| Métrica | Regressão Logística | Árvore de Decisão | Teorema de Bayes |
| :--- | :---: | :---: | :---: |
| **Acurácia Geral** | 82,12% | **85,87%** | *Abordagem Probabilística* |
| **Recall (Câmbio Automático)** | 33,00% | **54,00%** | Variável Dinâmica |
| **Recall (Câmbio Manual)** | 96,00% | 95,00% | Variável Dinâmica |

**Conclusão Técnico-Científica:** O mercado automotivo alemão mapeado apresenta um desbalanceamento natural severo, com uma quantidade massivamente superior de veículos de câmbio manual. Por essa razão, o classificador linear da Regressão Logística sofreu um forte viés, tendendo a classificar a classe majoritária com facilidade, mas falhando ao capturar apenas 33% dos carros automáticos. A **Árvore de Decisão provou ser o modelo vencedor**, alcançando **85,87% de acurácia global**. Ao mapear interações não-lineares complexas, ela conseguiu mitigar o desbalanceamento dos dados e elevou o *Recall* de veículos automáticos para 54%.

---

## 🖥️ 5. Dashboard Interativo (`app.py`)
Interface web desenvolvida utilizando a biblioteca **Streamlit**. O dashboard integra os modelos treinados para permitir que qualquer usuário simule o comportamento de mercado em tempo real:
* Formulários interativos para inserção de dados do veículo (Marca, Tipo, Potência, Ano).
* Previsão simultânea exibida por meio de cartões de métricas dinâmicas.
* Cálculo e plotagem de gráficos com as probabilidades analíticas das classes.
* Estilização customizada em CSS com fundo em vídeo dinâmico (`intro.mp4`).

---

## ⚙️ Como Executar o Projeto

### 1. Preparação e Execução do Pipeline
Certifique-se de baixar a base de dados `autos.csv` diretamente do Kaggle e colocá-la na pasta raiz do projeto (pois ela está listada no `.gitignore`). 

Depois, abra o seu terminal na pasta do projeto e execute o bloco de comandos abaixo para criar o ambiente virtual, instalar as dependências e rodar todo o pipeline de dados e a interface:

```bash
# 1. Criar e ativar o ambiente virtual (opcional, mas recomendado)
python -m venv .venv
source .venv/bin/activate  # No Linux/Mac
.venv\Scripts\activate     # No Windows

# 2. Instalar todas as bibliotecas necessárias
pip install pandas scikit-learn streamlit seaborn matplotlib

# 3. Executar o script de limpeza (gera o 'autos_tratado.csv')
python tratamento_dados.py

# 4. Executar o script de classificação para ver as métricas de Machine Learning
python classificacao.py

# 5. Entrar na pasta
cd "DashBoard(Autos)"

# 6. Iniciar o dashboard interativo no navegador
streamlit run app.py