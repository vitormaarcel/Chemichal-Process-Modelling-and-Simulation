import math
import numpy as np

pi = math.pi
emissividade = 0.8
sigma = 5.67e-8  # Stefan-Boltzman const. [W/m2cte_resistiva_isolante4]


def calor_radicao(d, t, temperatura_inferior):
    calor_radicao = emissividade*pi*d*sigma*(t**4-temperatura_inferior**4)
    return calor_radicao


def calor_conveccao(d, t, h_externo, temperatura_vizinhanca):
    calor_conveccao = h_externo*pi*d*(t-temperatura_vizinhanca)
    return calor_conveccao


def calor_cond(d, t, cte_resistiva_isolante, temperatura_interna, diametro_tubo):
    calor_cond = 2*pi*cte_resistiva_isolante * \
        (temperatura_interna-t)/np.log(d/diametro_tubo)
    return calor_cond
