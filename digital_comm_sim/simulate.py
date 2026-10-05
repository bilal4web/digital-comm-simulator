"""Monte Carlo BER simulation over the AWGN channel.

For each Eb/N0 point, batches of random bits are generated, modulated,
passed through AWGN, demodulated, and compared with the transmitted bits.
Simulation at a point stops once ``target_errors`` bit errors have been
observed or ``max_bits`` bits have been simulated, whichever comes first.
"""

import numpy as np

from .bits import generate_bits
from .modulation import BITS_PER_SYMBOL, modulate, demodulate
from .channel import awgn


def simulate_ber(scheme: str, eb_n0_db: float, target_errors: int = 100,
                 max_bits: int = 2_000_000, batch_bits: int = 200_000,
                 seed: int | None = None) -> tuple[float, int, int]:
    """Estimate BER for one scheme at one Eb/N0 point.

    Returns
    -------
    (ber, errors, bits_simulated)
    """
    k = BITS_PER_SYMBOL[scheme]
    batch_bits = (batch_bits // k) * k  # whole symbols per batch
    rng = np.random.default_rng(seed)
    errors = 0
    bits_total = 0
    while errors < target_errors and bits_total < max_bits:
        n = min(batch_bits, max_bits - bits_total)
        n = (n // k) * k
        if n == 0:
            break
        tx_bits = generate_bits(n, seed=int(rng.integers(0, 2 ** 31)))
        tx_syms = modulate(tx_bits, scheme)
        rx_syms = awgn(tx_syms, eb_n0_db, k)
        rx_bits = demodulate(rx_syms, scheme)
        errors += int(np.sum(tx_bits != rx_bits))
        bits_total += n
    ber = errors / bits_total if bits_total else float("nan")
    return ber, errors, bits_total


def sweep(scheme: str, eb_n0_db: np.ndarray, **kwargs) -> dict:
    """Run :func:`simulate_ber` over an array of Eb/N0 points.

    Returns a dict with keys ``eb_n0_db``, ``ber``, ``errors``, ``bits``.
    """
    eb_n0_db = np.asarray(eb_n0_db, dtype=float)
    bers, errs, bits = [], [], []
    for i, snr in enumerate(eb_n0_db):
        ber, e, b = simulate_ber(scheme, float(snr), seed=1234 + i, **kwargs)
        bers.append(ber)
        errs.append(e)
        bits.append(b)
        print(f"  {scheme:5s} Eb/N0={snr:5.1f} dB  BER={ber:.3e} "
              f"(errors={e}, bits={b})", flush=True)
    return {"eb_n0_db": eb_n0_db,
            "ber": np.array(bers),
            "errors": np.array(errs),
            "bits": np.array(bits)}
