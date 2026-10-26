import numpy as np
import pytest

from qmat import genQDeltaCoeffs
from qmat.qcoeff.collocation import Collocation


@pytest.mark.parametrize("nNodes", [2, 3, 4, 5, 6])
def testCentroskewGauss(nNodes):
    coll = Collocation(
        nNodes=nNodes,
        nodeType="LEGENDRE",
        quadType="GAUSS",
    )

    S = genQDeltaCoeffs(
        "S",
        qGen=coll,
        check_symmetry=True,
    )

    J = np.fliplr(np.eye(nNodes))
    correction = 0.5 * np.outer(
        np.ones(nNodes),
        coll.weights,
    )

    # Definition
    assert np.allclose(
        S,
        coll.Q - correction,
    )

    # Centroskew property
    assert np.allclose(
        J @ S @ J,
        -S,
    )

    # Rank-one difference Q-S
    assert np.allclose(
        coll.Q - S,
        correction,
    )


@pytest.mark.parametrize(
    "quadType",
    ["RADAU-RIGHT", "RADAU-LEFT"],
)
def testCentroskewRejectsNonSymmetricCollocation(quadType):
    coll = Collocation(
        nNodes=4,
        nodeType="LEGENDRE",
        quadType=quadType,
    )

    with pytest.raises(ValueError):
        genQDeltaCoeffs(
            "S",
            qGen=coll,
            check_symmetry=True,
        )


@pytest.mark.parametrize("nNodes", [2, 4, 6])
def testCentroskewStiffLimit(nNodes):
    coll = Collocation(
        nNodes=nNodes,
        nodeType="LEGENDRE",
        quadType="GAUSS",
    )

    S = genQDeltaCoeffs(
        "S",
        qGen=coll,
        check_symmetry=True,
    )

    Kinf = np.eye(nNodes) - np.linalg.solve(S, coll.Q)

    # The theoretical result is K_inf^2 = 0
    assert np.allclose(
        Kinf @ Kinf,
        0,
        atol=1e-11,
    )
    
@pytest.mark.parametrize("nNodes", [3, 5])
def testCentroskewOddStageSingular(nNodes):
    coll = Collocation(
        nNodes=nNodes,
        nodeType="LEGENDRE",
        quadType="GAUSS",
    )

    S = genQDeltaCoeffs("S", qGen=coll)

    assert np.linalg.matrix_rank(S) < nNodes