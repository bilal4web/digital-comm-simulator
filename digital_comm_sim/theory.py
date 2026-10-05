"""Closed-form bit-error-rate expressions (AWGN, hard decision).

Used expressions (all standard textbook results for Gray-coded constellations
over AWGN):

- BPSK:  Pb = Q(sqrt(2 * Eb/N0))
- QPSK:  Pb = Q(sqrt(2 * Eb/N0))   (Gray-coded QPSK has the same bit-error
  performance as BPSK: each bit sees an equivalent BPSK channel)
- 16-QAM: Pb ~= (3/8) * erfc(sqrt(0.4 * Eb/N0))
  (Gray-coded square 16-QAM approximation, e.g. Proakis / standard
  communications references)
"""

import numpy as np
from scipy.special import erfc


def qfunc(x: np.ndarray | float) -> np.ndarray | float:
    """Gaussian Q-function, Q(x) = 0.5 * erfc(x / sqrt(2))."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def ber_bpsk(eb_n0_db: np.ndarray | float):
    """Theoretical BER of BPSK over AWGN."""
    gamma = 10.0 ** (np.asarray(eb_n0_db, dtype=float) / 10.0)
    return qfunc(np.sqrt(2.0 * gamma))


def ber_qpsk(eb_n0_db: np.ndarray | float):
    """Theoretical BER of Gray-coded QPSK over AWGN (identical to BPSK)."""
    return ber_bpsk(eb_n0_db)


def ber_qam16(eb_n0_db: np.ndarray | float):
    """Theoretical BER approximation of Gray-coded 16-QAM over AWGN."""
    gamma = 10.0 ** (np.asarray(eb_n0_db, dtype=float) / 10.0)
    return (3.0 / 8.0) * erfc(np.sqrt(0.4 * gamma))


THEORY = {"BPSK": ber_bpsk, "QPSK": ber_qpsk, "QAM16": ber_qam16}
