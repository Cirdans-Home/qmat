"""Tests for RadauRankTwo with explicitly supplied alpha and beta."""

import numpy as np
import pytest
from numpy.polynomial.legendre import legvander

from qmat import genQDeltaCoeffs
from qmat.qcoeff.collocation import Collocation
from qmat.qdelta.structured import RadauRankTwo


def radau(M):
    return Collocation(nNodes=M, nodeType="LEGENDRE", quadType="RADAU-RIGHT")


@pytest.mark.parametrize("M", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("r,s", [(0, 0), (1, 1), (0.4, 0.2), (1, -1)])
def test_original_and_transformed_formulas(M, r, s):
    coll = radau(M)
    zeta = 1.0 / (4*M - 2)
    alpha, beta = r/2, zeta*s
    gen = RadauRankTwo(qGen=coll, alpha=alpha, beta=beta)
    QD = gen.getQDelta()

    # Independently construct the orthonormal shifted Legendre basis.
    W = legvander(2*coll.nodes - 1, M-1) * np.sqrt(2*np.arange(M) + 1)
    Winv = W.T * coll.weights[None, :]
    XD = Winv @ coll.Q @ W
    XD[0, 0] -= alpha
    XD[-1, -1] -= beta

    np.testing.assert_allclose(Winv @ W, np.eye(M), atol=2e-12)
    np.testing.assert_allclose(QD, W @ XD @ Winv, rtol=1e-12, atol=2e-12)
    np.testing.assert_allclose(gen.p, W[:, -1], atol=2e-12)

    # The weights in p p^T B must multiply on the right.
    p = W[:, -1]
    expected = (
        coll.Q
        - alpha * np.outer(np.ones(M), coll.weights)
        - beta * np.outer(p, p * coll.weights)
    )
    np.testing.assert_allclose(QD, expected, rtol=1e-12, atol=2e-12)
    assert np.linalg.matrix_rank(coll.Q - QD, tol=1e-11) <= 2

    assert gen.alpha == pytest.approx(alpha)
    assert gen.beta == pytest.approx(beta)
    assert gen.zeta == pytest.approx(zeta)
    assert gen.r == pytest.approx(r)
    assert gen.s == pytest.approx(s)


@pytest.mark.parametrize("M", [2, 4, 6, 8, 12])
def test_even_explicit_optimal_parameters(M):
    coll = radau(M)
    zeta = 1.0 / (4*M - 2)
    a = np.sqrt(2.0) - 1.0
    gen = RadauRankTwo(qGen=coll, alpha=0.5, beta=zeta*a*a)
    K = np.eye(M) - np.linalg.solve(gen.getQDelta(), coll.Q)

    # At this explicit parameter choice, K(K+aI)^2 = 0 in exact arithmetic.
    A = K + a*np.eye(M)
    np.testing.assert_allclose(K @ A @ A, 0, atol=2e-11)

    # Repeated eigenvalues are more sensitive than the polynomial identity.
    rho = np.max(np.abs(np.linalg.eigvals(K)))
    assert rho == pytest.approx(a, abs=2e-6)


@pytest.mark.parametrize("M", [3, 5, 7, 9, 11])
def test_odd_explicit_nilpotent_parameters(M):
    coll = radau(M)
    zeta = 1.0 / (4*M - 2)
    gen = RadauRankTwo(qGen=coll, alpha=0.5, beta=-zeta)
    K = np.eye(M) - np.linalg.solve(gen.getQDelta(), coll.Q)

    # Check nilpotency using powers, not numerically computed eigenvalues.
    np.testing.assert_allclose(K @ K @ K, 0, atol=2e-11)
    assert np.linalg.norm(K @ K, ord=np.inf) > 1e-3


@pytest.mark.parametrize("M", [3, 4, 5, 6])
def test_registry_multiple_sweeps_and_exact_case(M):
    coll = radau(M)
    params = {"alpha": 0.2, "beta": 0.1 / (4*M - 2)}

    QD = genQDeltaCoeffs("RADAU-AB", qGen=coll, **params)
    gen = RadauRankTwo(qGen=coll, **params)
    np.testing.assert_array_equal(QD, gen.getQDelta())

    direct = genQDeltaCoeffs(
        "RADAU-AB", Q=coll.Q, nodes=coll.nodes,
        weights=coll.weights, **params,
    )
    np.testing.assert_array_equal(direct, QD)

    repeated = genQDeltaCoeffs(
        "RADAU-AB", qGen=coll, nSweeps=3, **params,
    )
    assert repeated.shape == (3, M, M)
    for matrix in repeated:
        np.testing.assert_array_equal(matrix, QD)

    exact = genQDeltaCoeffs(
        "RADAU-AB", qGen=coll, alpha=0.0, beta=0.0,
    )
    np.testing.assert_array_equal(exact, coll.Q)


@pytest.mark.parametrize("params", [{}, {"alpha": 0.5}, {"beta": 0.1}])
def test_both_parameters_are_required(params):
    coll = radau(4)
    with pytest.raises(TypeError, match="required positional argument"):
        RadauRankTwo(qGen=coll, **params)
    with pytest.raises(TypeError, match="required positional argument"):
        genQDeltaCoeffs("RADAU-AB", qGen=coll, **params)


@pytest.mark.parametrize("name", ["alpha", "beta"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_parameter(name, value):
    params = {"alpha": 0.5, "beta": 0.1}
    params[name] = value
    with pytest.raises(ValueError, match="finite real"):
        RadauRankTwo(qGen=radau(4), **params)


@pytest.mark.parametrize("name", ["alpha", "beta"])
@pytest.mark.parametrize("value", [None, 1+0j, [0.5]])
def test_parameter_not_convertible_to_float(name, value):
    params = {"alpha": 0.5, "beta": 0.1}
    params[name] = value
    with pytest.raises(TypeError):
        RadauRankTwo(qGen=radau(4), **params)


@pytest.mark.parametrize("alpha,beta", [(0.75, 0.1), (-0.25, -0.2)])
def test_explicit_parameters_are_not_range_restricted(alpha, beta):
    gen = RadauRankTwo(qGen=radau(4), alpha=alpha, beta=beta)
    assert gen.alpha == alpha
    assert gen.beta == beta
    assert np.all(np.isfinite(gen.getQDelta()))


@pytest.mark.parametrize("M", [3, 5, 7])
def test_singular_skew_part_is_allowed(M):
    coll = radau(M)
    QD = genQDeltaCoeffs(
        "RADAU-AB", qGen=coll, alpha=0.5, beta=1.0/(4*M - 2),
    )
    assert np.linalg.matrix_rank(QD, tol=1e-11) == M-1

    # QDelta is singular, but a sweep at z=-1 remains well defined.
    A = np.eye(M) + QD
    u = np.linalg.solve(A, np.ones(M))
    np.testing.assert_allclose(A @ u, np.ones(M), atol=1e-12)


def test_reject_one_stage():
    with pytest.raises(ValueError, match="at least two"):
        RadauRankTwo(qGen=radau(1), alpha=0.5, beta=0.1)


@pytest.mark.parametrize("name", ["Q", "weights", "nodes"])
def test_incompatible_shapes(name):
    coll = radau(4)
    params = {
        "Q": coll.Q,
        "weights": coll.weights,
        "nodes": coll.nodes,
        "alpha": 0.5,
        "beta": 0.1,
    }
    params[name] = params[name][:, :-1] if name == "Q" else params[name][:-1]
    with pytest.raises(ValueError, match="Incompatible shapes"):
        RadauRankTwo(**params)