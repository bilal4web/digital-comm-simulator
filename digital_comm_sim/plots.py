"""Plots: BER vs Eb/N0 curves and constellation diagrams."""

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

from .modulation import get_constellation, modulate
from .channel import awgn
from .bits import generate_bits
from .theory import THEORY


def plot_ber(results: dict, eb_fine_db: np.ndarray, path: str) -> None:
    """Plot simulated vs theoretical BER vs Eb/N0 (log scale).

    Parameters
    ----------
    results:
        Mapping scheme -> sweep() result dict.
    eb_fine_db:
        Fine Eb/N0 grid (dB) on which to evaluate the theoretical curves.
    path:
        Output PNG file path.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"BPSK": "tab:blue", "QPSK": "tab:orange", "QAM16": "tab:green"}
    for scheme, res in results.items():
        c = colors.get(scheme, None)
        ax.semilogy(res["eb_n0_db"], res["ber"], "o", color=c, markersize=5,
                    label=f"{scheme} (simulated)")
        ax.semilogy(eb_fine_db, THEORY[scheme](eb_fine_db), "-", color=c,
                    linewidth=1.5, label=f"{scheme} (theory)")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit error rate (BER)")
    ax.set_title("BER vs Eb/N0 over AWGN: simulation vs theory")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_constellations(path: str, eb_n0_db_noisy: float = 8.0,
                        n_symbols: int = 2000, seed: int = 7) -> None:
    """Plot clean and noisy constellation diagrams for each scheme.

    One row per scheme (BPSK, QPSK, 16-QAM); columns are the clean
    constellation and the constellation after the AWGN channel at
    ``eb_n0_db_noisy`` dB.
    """
    schemes = ["BPSK", "QPSK", "QAM16"]
    fig, axes = plt.subplots(len(schemes), 2, figsize=(9, 10))
    for row, scheme in enumerate(schemes):
        symbols, _ = get_constellation(scheme)
        k = int(np.log2(len(symbols)))
        tx_bits = generate_bits(n_symbols * k, seed=seed)
        tx = modulate(tx_bits, scheme)
        rx = awgn(tx, eb_n0_db_noisy, k, seed=seed + 1)

        ax = axes[row, 0]
        ax.scatter(symbols.real, symbols.imag, s=60, c="tab:blue",
                   edgecolors="black", zorder=3)
        ax.set_title(f"{scheme}: clean constellation")
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)

        ax = axes[row, 1]
        ax.scatter(rx.real, rx.imag, s=4, c="tab:red", alpha=0.5)
        ax.scatter(symbols.real, symbols.imag, s=60, c="none",
                   edgecolors="black", linewidths=1.5, zorder=3)
        ax.set_title(f"{scheme}: noisy (Eb/N0 = {eb_n0_db_noisy:.0f} dB)")
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        for a in (axes[row, 0], axes[row, 1]):
            a.set_xlabel("In-phase")
            a.set_ylabel("Quadrature")
    fig.suptitle("Constellation diagrams", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
