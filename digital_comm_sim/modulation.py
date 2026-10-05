"""Modulation and demodulation: BPSK, QPSK, and 16-QAM.

All constellations are Gray-coded and normalized to unit average symbol
energy (E[|s|^2] = 1).

Gray mappings
-------------
BPSK (1 bit/symbol):  0 -> -1,  1 -> +1

QPSK (2 bits/symbol, Gray):
    00 -> ( 1+1j)/sqrt(2)    01 -> (-1+1j)/sqrt(2)
    11 -> (-1-1j)/sqrt(2)    10 -> ( 1-1j)/sqrt(2)

16-QAM (4 bits/symbol): bits (b0, b1) select the in-phase level and
(b2, b3) the quadrature level, each Gray-coded as
    00 -> -3,  01 -> -1,  11 -> +1,  10 -> +3,
then the whole constellation is scaled by 1/sqrt(10) so that the average
symbol energy is 1.

Demodulation is hard-decision minimum-Euclidean-distance detection.
"""

import numpy as np

_QPSK_MAP = {
    (0, 0): (1 + 1j),
    (0, 1): (-1 + 1j),
    (1, 1): (-1 - 1j),
    (1, 0): (1 - 1j),
}
_QPSK_BITS = {v: k for k, v in _QPSK_MAP.items()}

_QAM16_LEVEL = {(0, 0): -3.0, (0, 1): -1.0, (1, 1): 1.0, (1, 0): 3.0}
_QAM16_NORM = np.sqrt(10.0)


def _build_constellation(scheme: str):
    """Return (symbols, bit_labels) for a scheme.

    symbols: complex array of constellation points (unit average energy).
    bit_labels: (M, k) uint8 array, bit_labels[i] are the bits of symbols[i].
    """
    if scheme == "BPSK":
        symbols = np.array([-1.0, 1.0], dtype=complex)
        bit_labels = np.array([[0], [1]], dtype=np.uint8)
    elif scheme == "QPSK":
        pts, labels = [], []
        for bits, pt in _QPSK_MAP.items():
            pts.append(pt / np.sqrt(2.0))
            labels.append(bits)
        symbols = np.array(pts, dtype=complex)
        bit_labels = np.array(labels, dtype=np.uint8)
    elif scheme == "QAM16":
        pts, labels = [], []
        for b0 in (0, 1):
            for b1 in (0, 1):
                for b2 in (0, 1):
                    for b3 in (0, 1):
                        i = _QAM16_LEVEL[(b0, b1)]
                        q = _QAM16_LEVEL[(b2, b3)]
                        pts.append((i + 1j * q) / _QAM16_NORM)
                        labels.append((b0, b1, b2, b3))
        symbols = np.array(pts, dtype=complex)
        bit_labels = np.array(labels, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")
    return symbols, bit_labels


_CONSTELLATIONS = {s: _build_constellation(s) for s in ("BPSK", "QPSK", "QAM16")}

BITS_PER_SYMBOL = {"BPSK": 1, "QPSK": 2, "QAM16": 4}


def get_constellation(scheme: str):
    """Return ``(symbols, bit_labels)`` for ``scheme`` (unit average energy)."""
    return _CONSTELLATIONS[scheme]


def modulate(bits: np.ndarray, scheme: str) -> np.ndarray:
    """Map a bit array to complex constellation symbols (Gray-coded).

    Parameters
    ----------
    bits:
        1-D array of 0/1 values; its length must be a multiple of the number
        of bits per symbol of ``scheme``.
    scheme:
        One of ``"BPSK"``, ``"QPSK"``, ``"QAM16"``.
    """
    symbols, bit_labels = get_constellation(scheme)
    k = BITS_PER_SYMBOL[scheme]
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % k:
        raise ValueError(f"Bit count {bits.size} not a multiple of {k}")
    grouped = bits.reshape(-1, k)
    # index = binary value of the bit group; build lookup table once
    table = np.empty(2 ** k, dtype=complex)
    for sym, label in zip(symbols, bit_labels):
        idx = 0
        for b in label:
            idx = (idx << 1) | int(b)
        table[idx] = sym
    idx = np.zeros(len(grouped), dtype=np.int64)
    for j in range(k):
        idx = (idx << 1) | grouped[:, j]
    return table[idx]


def demodulate(received: np.ndarray, scheme: str) -> np.ndarray:
    """Hard-decision minimum-distance demodulation.

    Returns the most likely bit array (same ordering as :func:`modulate`).
    """
    symbols, bit_labels = get_constellation(scheme)
    received = np.asarray(received, dtype=complex).ravel()
    # (n, M) squared distances, then argmin over constellation points
    dist2 = np.abs(received[:, None] - symbols[None, :]) ** 2
    nearest = np.argmin(dist2, axis=1)
    return bit_labels[nearest].reshape(-1).astype(np.uint8)
