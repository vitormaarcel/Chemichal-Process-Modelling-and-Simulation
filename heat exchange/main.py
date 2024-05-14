# Importar Bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# Dados de Entrada
pressao = 20  # pressão do vapor saturado [bar]
# temperatura de saturação do varpor [cte_resistiva_isolante]
temperatura_saturacao = 486
emissividade = 0.8
pi = np.pi
diametro_tubo = 0.2  # diâmetro inicial, sem isolante [m]
sigma = 5.67e-8  # Stefan-Boltzman const. [W/m2cte_resistiva_isolante4]
# temperatura da superfície interna [cte_resistiva_isolante]
temperatura_interna = temperatura_saturacao
# temperatura do ar na sala [cte_resistiva_isolante]
temperatura_inferior = 298
# tempertura da vizinhança do tubo [cte_resistiva_isolante]
temperatura_vizinhanca = temperatura_inferior
# coeficiente de transferência de calor externo [W/m2cte_resistiva_isolante]
h_externo = 20

# Sem isolante
# Definindo as funções de calor por radiação e por convecção


def calor_radicao(d, t):
    calor_radicao = emissividade*pi*d*sigma*(t**4-temperatura_inferior**4)
    return calor_radicao


def calor_conveccao(d, t):
    calor_conveccao = h_externo*pi*d*(t-temperatura_vizinhanca)
    return calor_conveccao


# Cálculo da perda de calor sem isolamento
calor_sem_isolante = round(
    calor_radicao(diametro_tubo, temperatura_interna) + calor_conveccao(diametro_tubo, temperatura_interna), 2)
print(f'Perda de calor sem isolamento térmico = {calor_sem_isolante} W/m')

# Com isolante
espessura_isolante = 50e-3  # [m]
diametro_novo = diametro_tubo + 2*espessura_isolante
# constante resistiva do isolante [W/mcte_resistiva_isolante]
cte_resistiva_isolante = 0.058

# Função do calor por condução


def calor_cond(d, t):
    calor_cond = 2*pi*cte_resistiva_isolante * \
        (temperatura_interna-t)/np.log(d/diametro_tubo)
    return calor_cond


# Método iterativo para encontrar a temperatura exterior
# Função objetivo e sua derivada


def funcao_objetivo(d, t):
    fo = calor_cond(d, t) - (calor_conveccao(d, t) + calor_radicao(d, t))
    return fo


def der_funcao_objetivo(d, t):
    dq_cond = -2*np.pi*cte_resistiva_isolante/np.log(d/diametro_tubo)
    dq_conv = h_externo*pi*d
    dq_rad = 4*emissividade*pi*d*sigma*(t**3)

    dfo = dq_cond - (dq_conv + dq_rad)
    return dfo


# Método da Newton-Raphson
erro = 1
tolerancia = 1e-3  # toleranciaerância
# temperatura externa da camada isolante; valor palpite [cte_resistiva_isolante]
t_ext = 298
iteracao = 0

while erro >= tolerancia:
    iteracao += 1
    t_nova = t_ext - funcao_objetivo(diametro_novo, t_ext) / \
        der_funcao_objetivo(diametro_novo, t_ext)
    erro = abs(t_nova-t_ext)
    t_ext = t_nova

calor_com_isolante = round(calor_cond(diametro_novo, t_nova), 2)
print(f'Perda de calor com isolamento térmico = {calor_com_isolante} W/m')

# Variação da temperatura final em função da espessura do isolante
variacao_espessura = []
variacao_temperatura = []
iteracoes = 100

for i in range(iteracoes):
    espessura_isolante += 50e-4
    diametro_novo = diametro_tubo + 2*espessura_isolante
    t_nova = t_ext - funcao_objetivo(diametro_novo, t_ext) / \
        der_funcao_objetivo(diametro_novo, t_ext)
    erro = abs(t_nova-t_ext)
    t_ext = t_nova
    variacao_espessura.append(espessura_isolante)
    variacao_temperatura.append(t_ext)

# Exibindo o resultado
plt.plot(variacao_espessura, variacao_temperatura)
plt.title('Variação da tempertura da superfície externa \nem função da espessura da camada isolante')
plt.xlabel('Variação da Espessura [m]')
plt.ylabel(
    'Variação da Temperatura da Superfície Externa [cte_resistiva_isolante]')
plt.show()
