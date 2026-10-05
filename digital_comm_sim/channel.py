"""AWGN channel model.

The channel adds circularly symmetric complex Gaussian noise. The noise
variance is set from the target Eb/N0, assuming unit-average-energy
transmit symbols (E[|s|^2] = 1, as produced by
:mod:`digital_comm_sim.modulation`).

For k bits per symbol:  Eb = Es / k = 1 / k.
With gamma = Eb/N0 (linear):  N0 = Eb / gamma.
Each of the real and imaginary noise components then has variance N0 / 2.
"""

import numpy as np


def awgn(symbols: np.ndarray, eb_n0_db: float, bits_per_symbol: int,
         seed: int | None = None) -> np.ndarray:
    """Pass complex symbols through an AWGN channel.

    Parameters
    ----------
    symbols:
        Transmit symbols, unit average energy.
    eb_n0_db:
        Target Eb/N0 in dB.
    bits_per_symbol:
        Bits carried per symbol (1 for BPSK, 2 for QPSK, 4 for 16-QAM).
    seed:
        Optional seed for the noise generator (reproducibility).
    """
    gamma = 10.0 ** (eb_n0_db / 10.0)          # Eb/N0, linear
    eb = 1.0 / bits_per_symbol                 # unit-energy symbols
    n0 = eb / gamma
    sigma = np.sqrt(n0 / 2.0)                  # std dev per real/imag part
    rng = np.random.default_rng(seed)
    noise = sigma * (rng.standard_normal(symbols.shape)
                     + 1j * rng.standard_normal(symbols.shape))
    return symbols + noise
