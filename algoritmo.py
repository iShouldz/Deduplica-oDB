import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import time

"""
Inicia timer para obter o tempo de execução
"""
start_time = time.time()

"""Inicializa as bases de dados de produtos da Google e da Amazon"""
df_amazon = pd.read_csv('amazon_cleaned.csv', encoding='utf-8')
df_google = pd.read_csv('../2/google_cleaned.csv', encoding='utf-8')


"""
Blocagem simples: Blocagem pela primeira letra do nome do produto. 
O processo é aplicado para cada uma das duas bases. É definida como atributo chave do processo a coluna "name". 
A letra é extraída é atribuída  a uma nova coluna. Essa coluna serve como base para grupar os registros dos blocos.
"""
df_amazon['block_key'] = df_amazon['name'].apply(lambda x: x[0].upper() if pd.notnull(x) else '')
df_google['block_key'] = df_google['name'].apply(lambda x: x[0].upper() if pd.notnull(x) else '')


"""
Comparação de registros dentro dos blocos: Tem por função realizar a comparação dentro de cada registro do bloco, 
de cada uma das bases. A similaridade entre o nome dos produtos é calculada usando a fuzz.radio, que fornece
uma implementação de pontos de similaridade baseado na distância de Levenshtein.
É definido o limiar, que serve para definir o quão similares são os nomes dos produtos, o valor do limiar é 
definido em 50, que é considerado baixo e permite certa variação entre os nomes. A abordagem em 50 é pensada 
para evitar minimiza a inclusão de falsos positivos. 
"""
def compare_blocks(df1, df2):
    potential_duplicates = []
    for block_key in df1['block_key'].unique():
        block_amazon = df1[df1['block_key'] == block_key]
        block_google = df2[df2['block_key'] == block_key]
        for _, row_amazon in block_amazon.iterrows():
            for _, row_google in block_google.iterrows():
                # Calcula a similaridade
                similarity = fuzz.ratio(row_amazon['name'], row_google['name'])
                if similarity > 50:
                    potential_duplicates.append((row_amazon['id'], row_google['id'], similarity))
    return pd.DataFrame(potential_duplicates, columns=['Amazon_ID', 'Google_ID', 'Similarity'])

duplicates = compare_blocks(df_amazon, df_google)

"""
Define a preparação para classificação de registros potencialmente duplicados, baseando-se na similaridade 
dos nomes.
"""
duplicates['is_duplicate'] = duplicates['Similarity'].apply(lambda x: 1 if x > 80 else 0)

X = duplicates[['Similarity']]
y = duplicates['is_duplicate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar um classificador
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Avaliar o modelo
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

# Salvar os potenciais duplicados para análise futura
duplicates.to_csv('duplicadasEncontradas.csv', index=False)

end_time = time.time()
execution_time = end_time - start_time

print(f"Tempo de execução do algoritmo: {execution_time:.2f} segundos.")