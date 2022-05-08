# Importar Bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# Dados de Entrada
pressao = 20  # pressão do vapor saturado [bar]
t_sat = 486  # temperatura de saturação do varpor [K]
emissividade = 0.8
pi = np.pi
d_tubo = 0.2  # diâmetro inicial, sem isolante [m]
sigma = 5.67e-8  # Stefan-Boltzman const. [W/m2K4]
t_int = t_sat  # temperatura da superfície interna [K]
t_inf = 298  # temperatura do ar na sala [K]
t_viz = t_inf  # tempertura da vizinhança do tubo [K]
h_ext = 20  # coeficiente de transferência de calor externo [W/m2K]

# Sem isolante
# Definindo as funções de calor por radiação e por convecção


def calor_rad(d, t):
    calor_rad = emissividade*pi*d*sigma*(t**4-t_inf**4)
    return calor_rad


def calor_conv(d, t):
    calor_conv = h_ext*pi*d*(t-t_viz)
    return calor_conv


# Cálculo da perda de calor sem isolamento
calor_sem_isolante = round(
    calor_rad(d_tubo, t_int) + calor_conv(d_tubo, t_int), 2)
print(f'Perda de calor sem isolamento térmico = {calor_sem_isolante} W/m')

# Com isolante
espessura_isolante = 50e-3  # [m]
d_novo = d_tubo + 2*espessura_isolante
k = 0.058  # constante resistiva do isolante [W/mK]

# Função do calor por condução


def calor_cond(d, t):
    calor_cond = 2*pi*k*(t_int-t)/np.log(d/d_tubo)
    return calor_cond


# Método iterativo para encontrar a temperatura exterior
# Função objetivo e sua derivada


def funcao_objetivo(d, t):
    fo = calor_cond(d, t) - (calor_conv(d, t) + calor_rad(d, t))
    return fo


def der_funcao_objetivo(d, t):
    dq_cond = -2*np.pi*k/np.log(d/d_tubo)
    dq_conv = h_ext*pi*d
    dq_rad = 4*emissividade*pi*d*sigma*(t**3)

    dfo = dq_cond - (dq_conv + dq_rad)
    return dfo


# Método da Newton-Rphson
erro = 1
tol = 1e-3  # tolerância
t_ext = 298  # temperatura externa da camada isolante; valor palpite [K]
iteracao = 0

while erro >= tol:
    iteracao += 1
    t_nova = t_ext - funcao_objetivo(d_novo, t_ext) / \
        der_funcao_objetivo(d_novo, t_ext)
    erro = abs(t_nova-t_ext)
    t_ext = t_nova

calor_com_isolante = round(calor_cond(d_novo, t_nova), 2)
print(f'Perda de calor com isolamento térmico = {calor_com_isolante} W/m')

# Variação da temperatura final em função da espessura do isolante
variacao_espessura = []
variacao_temperatura = []
iteracoes = 100

for i in range(iteracoes):
    espessura_isolante += 50e-4
    d_novo = d_tubo + 2*espessura_isolante
    t_nova = t_ext - funcao_objetivo(d_novo, t_ext) / \
        der_funcao_objetivo(d_novo, t_ext)
    erro = abs(t_nova-t_ext)
    t_ext = t_nova
    variacao_espessura.append(espessura_isolante)
    variacao_temperatura.append(t_ext)

# Exibindo o resultado
plt.plot(variacao_espessura, variacao_temperatura)
plt.title('Variação da tempertura da superfície externa \nem função da espessura da camada isolante')
plt.xlabel('Variação da Espessura [m]')
plt.ylabel('Variação da Temperatura da Superfície Externa [K]')
plt.show()
