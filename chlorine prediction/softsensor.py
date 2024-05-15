# Importando Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Entrada
X = pd.read_csv('Softsensor\entrada.csv')

# Saída
y = pd.read_csv('Softsensor\saida.csv')

# Separação e normalização dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

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

# Construção e Treinamento de Rede
mlp = MLPRegressor(hidden_layer_sizes=(10, ), activation='tanh',
                   solver='lbfgs', max_iter=5000, tol=10e-5)
mlp.fit(X_train, y_train.ravel())

# Resultados
y_pred = mlp.predict(X_test)

# Métricas
score = mlp.score(X_test, y_test)
RMSE = mean_squared_error(y_test, y_pred, squared=False)
print('Score:{:.4f}\n RMSE:{:.4f}'.format(score, RMSE))

# Resultado Gráfico

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k')
plt.plot()
plt.xlabel('Target')
plt.ylabel('Predict')
plt.show()
