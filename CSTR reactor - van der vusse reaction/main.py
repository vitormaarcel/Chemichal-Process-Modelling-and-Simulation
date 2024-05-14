# Importando bibliotecas

import numpy as np
import matplotlib.pyplot as plt

# Constantes e Condições Iniciais

area_reator = 5  # [m^2]
constante_valvula = 7/3600  # [m^2.5/s]
vazao_alimentacao = 10/3600  # [m^3/s]
constante_reacao1 = 10  # [1/s]
constante_reacao2 = 1  # [1/s]
constante_reacao3 = 0.5  # [1/mol/s]
concentracao_inicial_a = 0.6  # [mol/m^3]
concentracao_inicial_b = 0  # [mol/m^3]
concentracao_inicial_c = 0  # [mol/m^3]
concentracao_inicial_d = 0  # [mol/m^3]
altura_inicial = 0.5  # [m]

# Parâmetros (ODE's) - Variações ocorrem em função da variação de tempo [dt]


def VariacaoConcentracaoA(concentracao_a): return (concentracao_inicial_a - concentracao_a) / \
    tempo_residencia - taxa_consumo_reacao1 - 2 * taxa_consumo_reacao3


def VariacaoConcentracaoB(concentracao_b): return (concentracao_inicial_b - concentracao_b) / \
    tempo_residencia + taxa_consumo_reacao1 - taxa_consumo_reacao2


def VariacaoConcentracaoC(concentracao_c): return (concentracao_inicial_c - concentracao_c) / \
    tempo_residencia + taxa_consumo_reacao2


def VariacaoConcentracaoD(concentracao_d): return (concentracao_inicial_d - concentracao_d) / \
    tempo_residencia + taxa_consumo_reacao3


def VariacaoAltura(altura): return (vazao_alimentacao -
                                    constante_valvula*np.sqrt(altura))/area_reator


# Método Iterativo

passo = 0.001  # Comprimento do Passo
numero_iteracoes = 6000  # Número de Iterações
tempo = np.arange(0, numero_iteracoes + 1, 1)

# Armazenamento (Numerical Grid)

altura = np.zeros(len(tempo))
concentracao_a = np.zeros(len(tempo))
concentracao_b = np.zeros(len(tempo))
concentracao_c = np.zeros(len(tempo))
concentracao_d = np.zeros(len(tempo))
concentracao_a[0] = concentracao_inicial_a
concentracao_b[0] = concentracao_inicial_b
concentracao_c[0] = concentracao_inicial_c
concentracao_d[0] = concentracao_inicial_d
altura[0] = altura_inicial

for i in range(0, len(tempo) - 1):
    tempo_residencia = area_reator*altura[i]/vazao_alimentacao
    taxa_consumo_reacao1 = constante_reacao1*concentracao_a[i]
    taxa_consumo_reacao2 = constante_reacao2*concentracao_b[i]
    taxa_consumo_reacao3 = constante_reacao3*concentracao_a[i]**2
    concentracao_a[i + 1] = concentracao_a[i] + passo * \
        VariacaoConcentracaoA(concentracao_a[i])
    concentracao_b[i + 1] = concentracao_b[i] + passo * \
        VariacaoConcentracaoB(concentracao_b[i])
    concentracao_c[i + 1] = concentracao_c[i] + passo * \
        VariacaoConcentracaoC(concentracao_c[i])
    concentracao_d[i + 1] = concentracao_d[i] + passo * \
        VariacaoConcentracaoD(concentracao_d[i])
    altura[i + 1] = altura[i] + passo * VariacaoAltura(altura[i])

concentracao_final_b = round(concentracao_b[numero_iteracoes], 3)
concentracao_final_c = round(concentracao_c[numero_iteracoes], 3)
print(f' Concentração final de B = {concentracao_final_b} \n',
      f'Concentração final de C = {concentracao_final_c}')

# Exibição dos Resultados

plt.figure(figsize=(8, 6))
plt.plot(tempo, concentracao_a, color='black',
         label='concentracao_a [kmol/m^3]')
plt.plot(tempo, concentracao_b, color='red', label='concentracao_b [kmol/m^3]')
plt.plot(tempo, concentracao_c, color='blue',
         label='concentracao_c [kmol/m^3]')
plt.plot(tempo, concentracao_d, color='green',
         label='concentracao_d [kmol/m^3]')
plt.legend(loc='center right')
plt.xlabel('Número de iterações')
plt.ylabel('Variação da Concentração')
plt.show()
