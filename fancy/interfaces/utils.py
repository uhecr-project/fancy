import pickle
import os
import numpy as np


def get_nucleartable():
    '''Read dictionary of nuclear tables'''
    nuc_table = {
        "p": (1, 1),
        "H": (1, 1),
        "He": (4, 2),
        "Li": (7, 3),
        "C": (12, 6),
        "N": (14, 7),
        "O": (16, 8),
        "Si": (28, 14),
        "Fe": (56, 26),
    }

    return nuc_table


"""Integral of Fischer distribution used to evaluate kappa_d"""


def fischer_int(kappa, cos_thetaP):
    """Integral of vMF function over all angles"""
    return (1.0 - np.exp(-kappa * (1 - cos_thetaP))) / (1.0 - np.exp(-2.0 * kappa))


def fischer_int_eq_P(kappa, cos_thetaP, P):
    """Equation to find roots for"""
    return fischer_int(kappa, cos_thetaP) - P


def f_theta_scalar(kappa, P=0.683):
    """Returns costheta"""
    if kappa <= 1e5 and kappa > 1e-3:
        return np.arccos(1 + np.log((1 - P * (1 - np.exp(-2 * kappa)))) / kappa)
    elif kappa > 1e5:
        return np.arccos(1 + np.log(1 - P) / kappa)
    elif kappa <= 1e-3:
        return np.arccos(1 + np.log(1 - 2 * P * kappa) / kappa)


def f_theta(kappa, P=0.683):
    """vectorized version"""
    return np.vectorize(f_theta_scalar)(kappa, P)
