"""Pseudo-random bit generation.

All randomness in the simulator flows through :func:`generate_bits`, which is
seedable so every experiment is reproducible.
"""

import numpy as np


def generate_bits(n_bits: int, seed: int | None = None) -> np.ndarray:
    """Generate ``n_bits`` pseudo-random bits (0/1), uniformly distributed.

    Parameters
    ----------
    n_bits:
        Number of bits to generate.
    seed:
        Seed for the random number generator. ``None`` gives non-deterministic
        output; any integer gives a reproducible sequence.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_bits,)`` with dtype ``np.uint8``.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)
