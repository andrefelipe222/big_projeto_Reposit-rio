Link do Kaggle dataset; https://www.kaggle.com/code/amalsalilan/uncovering-the-factors-that-affect-used-car-prices/input?select=autos.csv 

4. RELATÓRIO TÉCNICO
1. Descrição do dataset
Origem: Dataset de veículos usados, derivado do autos.csv (com versão tratada autos_tratado.csv).
Domínio: Mercado automotivo, com foco em anúncios de venda de carros.
Número de instâncias: Aproximadamente dezenas de milhares de registros (dependendo da versão filtrada).
Atributos principais: vehicleType, brand, powerPS, yearOfRegistration, monthOfRegistration, fuelType, gearbox, price.
Variável alvo: gearbox (tipo de câmbio: automático ou manual).

2. Justificativa da escolha
O dataset é adequado porque:
Representa um problema real de classificação (prever câmbio, combustível, faixa de preço).
Possui variáveis categóricas e numéricas, permitindo aplicar tanto modelos probabilísticos (Bayes) quanto algoritmos de ML (Logistic Regression, Decision Tree).
É suficientemente grande para análises estatísticas e visualizações robustas.
Está alinhado ao objetivo de construir um dashboard interativo para análise e previsão.

3. Tratamentos aplicados
Imputação de valores ausentes:
Numéricos → substituídos pela média.
Categóricos → substituídos pela moda ou “desconhecido”.
One-Hot Encoding: aplicado para variáveis categóricas antes de treinar modelos.
Padronização: StandardScaler usado para regressão logística.
Criação de variáveis derivadas:
powerPS_cat (faixas de potência).
price_cat (faixas de preço).
Visualização: gráficos de distribuição, boxplots e histogramas para explorar relações.
Trecho de código ilustrativo:
num_cols = df.select_dtypes(include=['float64','int64']).columns
imputer = SimpleImputer(strategy="mean")
X[num_cols] = imputer.fit_transform(X[num_cols])

4. Insights da análise exploratória
Distribuição de câmbio: predominância de veículos manuais.
Combustível: gasolina e diesel são os mais comuns, mas há diversidade (até 7 tipos).
Potência: veículos automáticos tendem a ter maior potência média.
Tipo de veículo: SUVs e sedans apresentam maior proporção de câmbio automático.
Visualizações confirmaram padrões claros entre potência e câmbio, além da concentração de preços em faixas médias.

5. Análise probabilística (Bayes)
Aplicamos o Teorema de Bayes para calcular probabilidades condicionais:
Exemplo: 
𝑃(fuelType∣vehicleType,brand).
Probabilidades foram normalizadas e exibidas em gráficos de barras e métricas percentuais (ex: 70,0%).
Isso permitiu interpretar a chance de determinado combustível ou câmbio dado o perfil do veículo.

6. Resultados da classificação
Modelos usados:
Regressão Logística (com padronização).
Árvore de Decisão (profundidade máxima = 10).
Predições: exibidas em métricas (automatik ou manuell).
Probabilidades: mostradas em gráficos percentuais (ex: 49,2%, 70,0%).
Comparação com Bayes:
Bayes fornece uma visão probabilística simples, mas menos precisa.
ML entrega classificações mais robustas e probabilidades calibradas.

7. Conclusões
O projeto demonstrou como EDA, Bayes e ML podem ser integrados em um dashboard interativo.
Aprendizados: importância da imputação de dados, impacto da padronização, e diferenças entre abordagens probabilísticas e modelos supervisionados.
Limitações: dataset pode conter ruídos (valores extremos de preço/potência), e a acurácia depende da qualidade dos atributos disponíveis.
Extensões futuras: incluir métricas de desempenho (acurácia, ROC), explorar outros algoritmos (Random Forest, XGBoost) e expandir para previsão de preços.

5. DECLARAÇÃO DE USO DE IA GENERATIVA
O que foi feito com IA:
Apoio na escrita de código em Python (tratamento de dados, implementação de Bayes, ML e Streamlit).
Sugestões de melhorias visuais (uso de métricas, gráficos, vídeo de fundo).
Estruturação do relatório técnico e explicações conceituais.

Objetivo de aprendizado:
Compreender melhor conceitos de classificação probabilística e machine learning.
Aprender boas práticas de organização de dashboards interativos.
Desenvolver habilidades de documentação técnica.
Dificuldades que justificaram o uso:
Ajustes de código para lidar com valores ausentes e encoding.
Formatação de probabilidades em percentuais legíveis.
Integração de vídeo como fundo no Streamlit.
Estruturação clara e completa do relatório técnico.
