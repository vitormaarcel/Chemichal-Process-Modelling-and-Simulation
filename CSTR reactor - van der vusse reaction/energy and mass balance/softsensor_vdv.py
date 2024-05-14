import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Dados
dados = pd.read_excel('dados_simulacao_vdv.xlsx')

# Dados de Entrada
X = dados.drop(labels='cb(t)', axis=1)

# Dados de Saída
y = dados['cb(t)']

# Separação dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# Normalização
# Dados de entrada
scalerInput = StandardScaler()
scalerInput.fit(X_train)
X_train = scalerInput.transform(X_train)
X_test = scalerInput.transform(X_test)

# Dados de saída
scalerOutput = StandardScaler()
scalerOutput.fit(y_train)
y_train = scalerOutput.transform(y_train)
y_test = scalerOutput.transform(y_test)

# Construção e Treinamendo de Rede
mlp = MLPRegressor(hidden_layer_sizes=(10, ), activation='tanh',
                   solver='lbfgs', max_iter=200, tol=1e-4)
mlp.fit(X_train, y_train.ravel())

# Resultados
y_pred = mlp.predict(X_test)

# Métricas
score = mlp.score(X_test, y_test)
RMSE = mean_squared_error(y_test, y_pred, squared=False)

# Gráfico de Paridade
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_pred)], [min(y_test), max(y_pred)], 'k')
plt.suptitle(f'Score: {round(score, 4)}\n'
             f'RMSE: {round(RMSE, 4)}')
plt.plot()
plt.xlabel('Target')
plt.ylabel('Predict')
plt.show()
