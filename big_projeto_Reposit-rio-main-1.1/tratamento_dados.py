import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("Iniciando o tratamento dos dados...")

# 1. Carregar o dataset original
try:
    df = pd.read_csv('autos.csv')
except FileNotFoundError:
    print("ERRO: O arquivo 'autos.csv' nao foi encontrado na mesma pasta!")
    exit()

# 2. Remocao de Colunas Irrelevantes (Reducao de dimensionalidade)
colunas_para_remover = ['index', 'dateCrawled', 'name', 'seller', 'offerType', 
                        'abtest', 'dateCreated', 'nrOfPictures', 'postalCode', 'lastSeen']
df = df.drop(columns=colunas_para_remover, errors='ignore')

# 3. Tratamento de Outliers Graves (Dados inconsistentes/erros de digitacao)
df = df[(df['yearOfRegistration'] >= 1950) & (df['yearOfRegistration'] <= 2016)]
df = df[(df['price'] >= 100) & (df['price'] <= 150000)]
df = df[(df['powerPS'] >= 10) & (df['powerPS'] <= 1000)]

# 4. Tratamento de Valores Ausentes (Missing Values)
df = df.dropna(subset=['gearbox'])
df['vehicleType'] = df['vehicleType'].fillna('unspecified')
df['model'] = df['model'].fillna('unspecified')
df['fuelType'] = df['fuelType'].fillna('unspecified')
df['notRepairedDamage'] = df['notRepairedDamage'].fillna('unspecified')

# 5. Engenharia de Atributos & Codificacao (Encoding)
colunas_categoricas = ['vehicleType', 'model', 'fuelType', 'brand', 'notRepairedDamage']
le = LabelEncoder()
for col in colunas_categoricas:
    df[col] = le.fit_transform(df[col].astype(str))

# Transformando o alvo (gearbox): manuell -> 1, automatik -> 0
df['gearbox'] = le.fit_transform(df['gearbox'])

# 6. Salvar o arquivo limpo
df.to_csv('autos_tratado.csv', index=False)
print("Sucesso! O arquivo 'autos_tratado.csv' foi gerado com sucesso.")
