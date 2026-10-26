#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Structured QDelta matrices associated with Runge--Kutta symmetries.
"""

import numpy as np
from numpy.polynomial.legendre import legval

from qmat.qcoeff import QGenerator
from qmat.qcoeff.collocation import Collocation
from qmat.qdelta import QDeltaGenerator, register


@register
class Centroskew(QDeltaGenerator):
    r"""
    Centroskew part of a symmetric Runge--Kutta matrix.

    For a symmetric RK method satisfying

        J Q J + Q = 1 b^T,

    this generator returns

        S = Q - 1/2 1 b^T,

    so that J S J = -S.
    """

    aliases = ["S", "CENTROSKEW"]

    def __init__(self, Q, weights, check_symmetry=False, **kwargs):
        super().__init__(Q=Q)

        self.weights = np.asarray(weights, dtype=float)

        if self.weights.shape != (self.size,):
            raise ValueError(
                f"weights must have shape ({self.size},), "
                f"got {self.weights.shape}"
            )

        if check_symmetry:
            J = np.fliplr(np.eye(self.size))
            one = np.ones(self.size)

            if not np.allclose(
                J @ self.Q @ J + self.Q,
                np.outer(one, self.weights),
            ):
                raise ValueError(
                    "The supplied RK/collocation scheme is not symmetric"
                )

    @staticmethod
    def extractParams(qGen: QGenerator) -> dict:
        assert isinstance(qGen, QGenerator), \
            "qGen parameter must be a QGenerator object"

        return {
            "Q": qGen.Q,
            "weights": qGen.weights,
        }

    def computeQDelta(self, k=None):
        one = np.ones(self.size)

        return self.Q - 0.5 * np.outer(one, self.weights)
    
def _radau_real_scalar(value, name):
    """Convert a real scalar to float, rejecting arrays and nonfinite values."""
    value = np.asarray(value)
    if value.ndim != 0 or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


import numpy as np
from numpy.polynomial.legendre import legval

from qmat.qcoeff import QGenerator
from qmat.qdelta import QDeltaGenerator, register


@register
class RadauRankTwo(QDeltaGenerator):
    r"""
    Radau IIA preconditioner with explicitly supplied alpha and beta.

        QDelta = Q - alpha * 1 b^T - beta * p p^T B,

    where B = diag(weights) and p contains the values of the
    orthonormal shifted Legendre polynomial of degree M-1.

    Assumes Radau IIA coefficients on [0, 1], with M >= 2.
    """

    aliases = ["RADAU-AB"]

    def __init__(self, Q, weights, nodes, alpha, beta, **kwargs):
        super().__init__(Q=np.asarray(Q, dtype=float))

        self.weights = np.asarray(weights, dtype=float)
        self.nodes = np.asarray(nodes, dtype=float)
        self.alpha = float(alpha)
        self.beta = float(beta)

        M = self.size

        if M < 2:
            raise ValueError("RadauRankTwo requires at least two stages")

        if (
            self.Q.shape != (M, M)
            or self.weights.shape != (M,)
            or self.nodes.shape != (M,)
        ):
            raise ValueError("Incompatible shapes for Q, weights and nodes")

        if not np.all(np.isfinite([self.alpha, self.beta])):
            raise ValueError("alpha and beta must be finite real numbers")

        # Derived parameters retained for compatibility with existing scripts.
        self.zeta = 1.0 / (4*M - 2)
        self.r = 2.0 * self.alpha
        self.s = self.beta / self.zeta

        # p_i = sqrt(2M-1) * L_{M-1}(2c_i-1).
        coeffs = np.zeros(M)
        coeffs[-1] = 1.0
        self.p = np.sqrt(2*M - 1) * legval(2*self.nodes - 1, coeffs)

    @staticmethod
    def extractParams(qGen: QGenerator) -> dict:
        return {
            "Q": qGen.Q,
            "weights": qGen.weights,
            "nodes": qGen.nodes,
        }

    def computeQDelta(self, k=None) -> np.ndarray:
        return (
            self.Q
            - self.alpha * np.outer(np.ones(self.size), self.weights)
            - self.beta * np.outer(self.p, self.p * self.weights)
        )