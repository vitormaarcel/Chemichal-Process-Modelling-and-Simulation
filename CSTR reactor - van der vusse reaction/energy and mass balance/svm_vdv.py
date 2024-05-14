# Importando bibliotecas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Dados
dados = pd.read_excel('dados_simulacao_vdv.xlsx')

# Dados de Entrada
X = dados.drop(labels='cb(t)', axis=1)

# Dados de Saída
y = dados['cb(t)']

# Separação dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# Normalização
# Dados de entrada
scalerInput = StandardScaler()
X_train = scalerInput.fit_transform(X_train)
X_test = scalerInput.fit_transform(X_test)

# Dados de saída
scalerOutput = StandardScaler()
y_train = scalerOutput.fit_transform(y_train)
y_test = scalerOutput.fit_transform(y_test)


# Construção da rede
rbf = SVR(kernel='rbf', gamma='scale', tol=0.001, C=1.0, epsilon=0.1)

linear = SVR(kernel='linear', tol=0.001, C=1.0, epsilon=0.1)

poly = SVR(kernel='poly', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0,
           epsilon=0.1)

sigmoid = SVR(kernel='sigmoid', gamma='scale', coef0=0.0, tol=0.001, C=1.0,
              epsilon=0.1)

# Treinamento da rede
rbf.fit(X_train, y_train.ravel())
linear.fit(X_train, y_train.ravel())
poly.fit(X_train, y_train.ravel())
sigmoid.fit(X_train, y_train.ravel())

# Resultados
y_rbf = rbf.predict(X_train)
y_linear = linear.predict(X_train)
y_poly = poly.predict(X_train)
y_sigmoid = sigmoid.predict(X_train)

# Métricas
score_rbf = rbf.score(X_test, y_test)
score_linear = linear.score(X_test, y_test)
score_poly = poly.score(X_test, y_test)
score_sigmoid = sigmoid.score(X_test, y_test)

RMSE_rbf = mean_squared_error(y_train, y_rbf, squared=False)
RMSE_linear = mean_squared_error(y_train, y_linear, squared=False)
RMSE_poly = mean_squared_error(y_train, y_poly, squared=False)
RMSE_sigmoid = mean_squared_error(y_train, y_sigmoid, squared=False)

# Gráfico
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 6))

ax1.set_title(f'TREINO RBF\nScore={round(score_rbf, 3)}\n'
              f'RMSE={round(RMSE_rbf, 3)}')
ax1.scatter(y_train, y_rbf)
ax1.plot([min(y_train), max(y_rbf)], [min(y_train), max(y_rbf)], 'k')
ax1.set_xlabel('Target')
ax1.set_ylabel('Predict')

ax2.set_title(f'TREINO LINEAR\nScore={round(score_linear, 3)}\n'
              f'RMSE={round(RMSE_linear, 3)}')
ax2.scatter(y_train, y_linear)
ax2.plot([min(y_train), max(y_linear)], [min(y_train), max(y_linear)], 'k')
ax2.set_xlabel('Target')
ax2.set_ylabel('Predict')

ax3.set_title(f'TREINO POLY\nScore={round(score_poly, 3)}\n'
              f'RMSE={round(RMSE_poly, 3)}')
ax3.scatter(y_train, y_poly)
ax3.plot([min(y_train), max(y_poly)], [min(y_train), max(y_poly)], 'k')
ax3.set_xlabel('Target')
ax3.set_ylabel('Predict')

ax4.set_title(f'TREINO SIGMOID\nScore={round(score_sigmoid, 3)}\n'
              f'RMSE={round(RMSE_sigmoid, 3)}')
ax4.scatter(y_train, y_sigmoid)
ax4.plot([min(y_train), max(y_sigmoid)], [min(y_train), max(y_sigmoid)], 'k')
ax4.set_xlabel('Target')
ax4.set_ylabel('Predict')
plt.show()
