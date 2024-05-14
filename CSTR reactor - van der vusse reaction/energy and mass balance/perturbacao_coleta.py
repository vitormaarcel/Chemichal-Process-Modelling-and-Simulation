import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint

#  Parâmetros
volume_reator = 0.01  # m3
densidade_reag = 934.2  # kg/m3
cp_reag = 3.01  # kJ/kg.K; Capacidade Calorifica
massa_refri = 5  # kg
cp_refri = 2.0  # kJ/kg.K
area_jaqueta = 0.215  # m2; Area da Superfície
coeficiente_calor = 67.2  # kJ/min.m2.K; Coeficiente de Transferencia de Calor
prefator_reacao1 = 2.145e+10  # 1/min; Fator Pre-exponencial
prefator_reacao2 = 2.145e+10  # 1/min
prefator_reacao3 = 1.5072e+8  # 1/min.kmol
ativacao_reacao1 = 9758.3  # K; Valor da Razao entre Energia de Ativacao e R
ativacao_reacao2 = 9758.3  # K
ativacao_reacao3 = 8560  # K
entalpia_reacao1 = -4200  # kJ/kmol
entalpia_reacao2 = 11000  # kJ/kmol
entalpia_reacao3 = 41850  # kJ/kmol
conc_inicial_a = 5.1  # kmol/m3; Concentração Inicial de A
temp_inicial_reag = 387.05  # K; Temperatura Inicial do Componente A
vazao_reagente = 0.002365  # m3/min
remocao_calor_refri = -18.56  # kJ/min; taxa de remoção de calor do refrigerante

#  ODE's


def VarConcA(conc_a): return \
    (vazao_reagente/volume_reator)*(conc_inicial_a - conc_a) - \
    constante_reacao1*conc_a - constante_reacao3*(conc_a**2)


def VarConcB(conc_a, conc_b): return \
    (-vazao_reagente / volume_reator) * conc_b + \
    constante_reacao1 * conc_a - constante_reacao2 * conc_b


def VarTempReag(temp_reag, temp_refri): return \
    (vazao_reagente / volume_reator) * (temp_inicial_reag - temp_reag) - \
    (calor_reacao / (densidade_reag * cp_reag)) + \
    ((area_jaqueta * coeficiente_calor) / (volume_reator * densidade_reag * cp_reag)) * \
    (temp_refri - temp_reag)


def VarTempRefri(temp_reag, temp_refri): return \
    (1 / (massa_refri * cp_refri)) * \
    (remocao_calor_refri + area_jaqueta *
     coeficiente_calor * (temp_reag - temp_refri))


# Número de Iterações, Passo e Tempo de Simulação
passo = 0.001  # min
numero_iteracoes = 318001
tempo_simulacao = np.arange(0, numero_iteracoes + 1, 1)

# Constante de Tempo e Tempo de Amostragem
constante_tempo = 191.13  # s
tempo_perturbacao = int(constante_tempo/0.06)  # Número da iteração
tempo_amostragem = 0.1*constante_tempo
amostras = int(tempo_amostragem/0.06)  # Número da iteração

# Armazenamento (Numerical Grid)
conc_a = np.zeros(len(tempo_simulacao))
conc_b = np.zeros(len(tempo_simulacao))
temp_reag = np.zeros(len(tempo_simulacao))
temp_refri = np.zeros(len(tempo_simulacao))

# Valores Iniciais
conc_a[0] = 3.08
conc_b[0] = 0.924
temp_reag[0] = 374.2
temp_refri[0] = 372.92

# Listas dos Dados
vazoes = [0.002365]
vazoes_perturbadas = []
concb_t2 = []
concb_t1 = []
concb_t = []

# Método de Euler, Perturbação e Coleta de Dados
contador = 0

for i in range(0, len(tempo_simulacao) - 1):
    constante_reacao1 = prefator_reacao1 * \
        np.exp(-ativacao_reacao1/temp_reag[i])
    constante_reacao2 = prefator_reacao2 * \
        np.exp(-ativacao_reacao2/temp_reag[i])
    constante_reacao3 = prefator_reacao3 * \
        np.exp(-ativacao_reacao3/temp_reag[i])
    calor_reacao = entalpia_reacao1 * constante_reacao1 * conc_a[i] + \
        entalpia_reacao2 * constante_reacao2 * conc_b[i] + \
        entalpia_reacao3 * constante_reacao3 * (conc_a[i]**2)
    conc_a[i+1] = conc_a[i] + passo * VarConcA(conc_a[i])
    conc_b[i+1] = conc_b[i] + passo * VarConcB(conc_a[i], conc_b[i])
    temp_reag[i+1] = temp_reag[i] + passo * \
        VarTempReag(temp_reag[i], temp_refri[i])
    temp_refri[i+1] = temp_refri[i] + passo * \
        VarTempRefri(temp_reag[i], temp_refri[i])
    if (i % tempo_perturbacao) == 0:
        vazao_reagente = 0.002365
        porcentagem = randint(-10, 10)
        vazao_reagente_perturbada = vazao_reagente*(1 + porcentagem/100)
        for i in range(8):
            vazoes_perturbadas.append(vazao_reagente_perturbada)
        vazao_reagente = vazao_reagente_perturbada
    else:
        pass
    if (i % amostras) == 0:
        contador += 1
        if 1 <= contador <= 8:
            concb_t2.append(conc_b[i])
        if 2 <= contador <= 9:
            concb_t1.append(conc_b[i])
        if 3 <= contador <= 10:
            concb_t.append(conc_b[i])
        if contador == 10:
            contador = 0
    vazoes.append(vazao_reagente)

# Geração da Tabela de Dados
df = pd.DataFrame(
    {
        'vazoes': vazoes_perturbadas,
        'cb(t-2)': concb_t2,
        'cb(t-1)': concb_t1,
        'cb(t)': concb_t,
    }
)
df.to_excel('dados_simulacao_vdv.xlsx')

# Gráficos: Variação da Vazão e da Concentração de B
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Variação de Cb e Vazões Perturbadas')
ax1.plot(tempo_simulacao, conc_b)
ax2.plot(tempo_simulacao, vazoes)
plt.show()
