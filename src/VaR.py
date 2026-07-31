import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats

def retorno_carteira(retornos, pesos):
    return np.asarray(retornos) @ pesos

def medidas_kD(ret_cart, k=5):
    ret_cart = np.asarray(ret_cart)
    n_blocos = len(ret_cart) // k
    ret_truncado = ret_cart[:n_blocos * k]
    mu_kD = k * np.mean(ret_cart)
    sigma_kD = math.sqrt(k * np.var(ret_cart))
    retorno_kD = ret_truncado.reshape(n_blocos, k).sum(axis=1)
    return mu_kD, sigma_kD, retorno_kD

def VaR(mu, sigma, retorno, retorno_kD, pesos, k = 5):
    def Param(mu, sigma):
        z_95 = scipy.stats.norm.ppf(0.05)
        return -1 * (mu + sigma * z_95)

    def Hist(retorno_kD):
        return -np.quantile(retorno_kD, 0.05)

    def Boot(retorno):
        retorno = np.asarray(retorno)
        B = 5000
        idx = np.random.choice(np.arange(len(retorno)), size=(B, k), replace=True)
        ret_b = retorno[idx]
        ret_cart_b = retorno_carteira(ret_b, pesos)
        b_kD = ret_cart_b.sum(axis=1)
        return -np.quantile(b_kD, 0.05)

    def testeLR(retorno_kD, VaR_95):
        viol = retorno_kD <= -VaR_95
        n, x = len(retorno_kD), np.sum(viol)
        log_L0 = (n - x) * np.log(0.95) + x * np.log(0.05)
        log_L1 = (n - x) * np.log(1 - x/n) + x * np.log(x/n) if x > 0 else n * np.log(1 - x/n)
        LR = -2 * (log_L0 - log_L1)
        return scipy.stats.chi2.sf(LR, 1)

    VaR_param = Param(mu, sigma)
    VaR_hist = Hist(retorno_kD)
    VaR_boot = Boot(retorno)

    results = {
        "Métodos": ["Paramétrico", "Histórico", "Monte Carlo"],
        "VaR": [VaR_param, VaR_hist, VaR_boot],
        "p-valor": [testeLR(retorno_kD, VaR_param),
                    testeLR(retorno_kD, VaR_hist),
                    testeLR(retorno_kD, VaR_boot)]
    }
    return results
        


