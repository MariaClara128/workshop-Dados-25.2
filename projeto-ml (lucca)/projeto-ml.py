# 🎬 IMPORTAÇÕES
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


data = {
    'titulo': ['Filme A', 'Filme B', 'Filme C', 'Filme D', 'Filme E',
               'Filme F', 'Filme G', 'Filme H', 'Filme I', 'Filme J'],
    'ano': [2020, 2019, 2021, 2018, 2022, 2021, 2020, 2019, 2022, 2021],
    'duracao': [120, 150, 95, 180, 130, 140, 110, 160, 145, 125],
    'genero': ['Ação', 'Drama', 'Comédia', 'Ação', 'Drama',
               'Comédia', 'Drama', 'Ação', 'Comédia', 'Drama'],
    'orcamento': [100, 150, 50, 200, 120, 80, 130, 170, 140, 110],
    'receita': [300, 180, 70, 400, 250, 90, 200, 220, 210, 150],
    'nota': [7.5, 6.8, 7.2, 5.5, 8.0, 6.0, 7.0, 7.8, 7.1, 6.5],
    'votos': [1500, 2000, 500, 4000, 2500, 1200, 1800, 2200, 2100, 1600]
}

df = pd.DataFrame(data)
print(df.head())

# Criar variável alvo: sucesso (nota >= 7.0)
df['sucesso'] = (df['nota'] >= 7.0).astype(int) 

 # Informações gerais
print(df.info())

# Valores únicos por coluna
for col in df.columns:
    print(f"\nColuna: {col}")
    print(df[col].value_counts().head(5))

# Estatísticas descritivas
print(df.describe())

features = ['ano', 'duracao', 'genero', 'orcamento', 'receita', 'votos']
X = df[features]
y = df['sucesso']

print(X.head())
print(y.head())

# Colunas numéricas e categóricas
num_features = ['ano', 'duracao', 'orcamento', 'receita', 'votos']
cat_features = ['genero']

# Pipeline numérico
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline categórico
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar pipelines
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(X_train.shape, X_test.shape)

models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

pipelines = {name: Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
]) for name, model in models.items()}

# Treinar cada modelo
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    print(f"{name} treinado com sucesso.")

results = {}
for name, pipe in pipelines.items():
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} - Acurácia: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    print(f"{name} - CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    results[name] = acc

best_model_name = max(results, key=results.get)
best_model = pipelines[best_model_name]
print("\nMelhor modelo:", best_model_name)

joblib.dump(best_model, "best_model_filmes.pkl")
print("Modelo salvo em best_model_filmes.pkl")

novos_filmes = pd.DataFrame({
    'ano': [2023, 2022],
    'duracao': [130, 150],
    'genero': ['Ação', 'Drama'],
    'orcamento': [120, 180],
    'receita': [250, 300],
    'votos': [2000, 2200]
})

previsoes = best_model.predict(novos_filmes)
print("\nPrevisões de sucesso dos novos filmes:", previsoes)
