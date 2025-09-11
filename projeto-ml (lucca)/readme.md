🎬 Projeto de Machine Learning – Classificador de Filmes: Sucesso ou Fracasso
Objetivo

Criar um modelo de classificação binária para prever se um filme será um sucesso (nota ≥ 7.0) com base em variáveis como orçamento, receita, número de votos, duração e gênero.

 Estrutura do Projeto
1. Abordar o problema e analisar

Problema: prever se um filme será um sucesso ou fracasso.

Tipo: classificação binária → 0 = fracasso (nota < 7.0), 1 = sucesso (nota ≥ 7.0).

Desafios:

Diferenças entre gêneros, orçamentos e épocas de lançamento.

Filmes com pouco público podem enviesar a nota média.

Valores ausentes e formatos diferentes de dados.

2. Obter os dados

Fonte: TMDB 5000 Movie Dataset (tmdb_5000_movies.csv).

Leitura via: pandas.read_csv()

Colunas principais: title, budget, revenue, vote_average, vote_count, release_date, runtime, genres, production_companies.

3. Explorar os dados

Verificação de tamanho (df.shape) e tipos (df.info()).

Estatísticas descritivas (df.describe()).

Frequência de gêneros, número de votos, distribuição de receitas e orçamentos.

Identificação de valores ausentes.

4. Tratamento dos dados

Transformação de datas (release_date) → ano de lançamento (year).

Conversão de genres (lista de dicionários) → gênero principal (main_genre).

Criação da variável alvo sucesso:

df['sucesso'] = (df['vote_average'] >= 7.0).astype(int)

5. Separar base de dados em arrays

Features: ['budget', 'revenue', 'runtime', 'vote_count', 'year', 'main_genre']

Target: sucesso

X = df[features]
y = df['sucesso']

6. Técnicas de pré-processamento

Numéricos: imputação (mediana) + padronização (StandardScaler).

Categóricos (main_genre): imputação (mais frequente) + one-hot encoding.

Implementado com ColumnTransformer.

7. Divisão treino/teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

8. Definição de modelos

Modelos treinados com scikit-learn:

LogisticRegression(max_iter=500)

RandomForestClassifier(n_estimators=200, max_depth=8)

KNeighborsClassifier(n_neighbors=5)

Cada modelo integrado em um pipeline (pré-processamento + modelo).

9. Validação dos modelos

Métricas: Acurácia, Precision, Recall, F1-score.

Validação cruzada com 5 folds (cross_val_score).

Comparação entre treino, teste e cross-validation.

Exemplo esperado de saída:

RandomForest - Acurácia: 0.83
LogisticRegression - Acurácia: 0.68
KNN - Acurácia: 0.72

10. Salvar a solução

O melhor modelo é salvo com joblib:

import joblib
joblib.dump(best_model, "models/best_model_filmes.pkl")

📊 Resultados

O RandomForestClassifier geralmente apresenta o melhor desempenho (~82–84% de acurácia).

Ajustando o limiar de nota (vote_average), é possível controlar o que é considerado sucesso.

O modelo final é salvo em best_model_filmes.pkl para uso futuro.

🚀 Como usar

Suba o dataset no Colab:

from google.colab import files
files.upload()  # selecione tmdb_5000_movies.csv


Execute as 10 etapas do projeto.

Ajuste os parâmetros de interesse (como vote_average ou features adicionais) para melhorar o modelo.

Rode os modelos, compare os resultados e use o melhor modelo salvo.

📌 Resumo

Este projeto cobre o ciclo completo de Machine Learning supervisionado: análise e limpeza de dados, pré-processamento, treinamento, avaliação de modelos e salvamento da solução final. O dataset TMDB foi utilizado para prever se um filme terá sucesso com base em suas características.
