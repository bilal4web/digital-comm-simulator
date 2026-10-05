"""Run the full digital communication system simulation.

Produces:
  results/ber_vs_snr.png        - simulated vs theoretical BER vs Eb/N0
  results/constellations.png    - clean and noisy constellation diagrams
  results/verification.txt      - numerical simulation-vs-theory comparison

Usage:
    python3 run_simulation.py
"""

import os
import numpy as np

from digital_comm_sim.simulate import sweep
from digital_comm_sim.plots import plot_ber, plot_constellations
from digital_comm_sim.theory import THEORY

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Per-scheme Monte Carlo budget: simulate until 100 bit errors are seen or
# max_bits is reached. BPSK/QPSK need more bits because their BER is much
# lower at high Eb/N0.
SETTINGS = {
    "BPSK":  {"target_errors": 100, "max_bits": 10_000_000},
    "QPSK":  {"target_errors": 100, "max_bits": 10_000_000},
    "QAM16": {"target_errors": 100, "max_bits": 2_000_000},
}


def verify(results: dict) -> str:
    """Compare simulated BER against theory; return a text report.

    Only points with at least 10 observed errors are used for the deviation
    statistics (fewer errors are dominated by Monte Carlo noise).
    """
    lines = ["Simulation vs theory verification",
             "=================================",
             "Metric: max |log10(sim BER) - log10(theory BER)| over points",
             "with >= 10 observed errors.",
             ""]
    for scheme, res in results.items():
        th = THEORY[scheme](res["eb_n0_db"])
        mask = (res["errors"] >= 10) & (res["ber"] > 0) & (th > 0)
        dev = np.abs(np.log10(res["ber"][mask]) - np.log10(th[mask]))
        lines.append(f"{scheme}: max log10 deviation = {dev.max():.3f} "
                     f"over {mask.sum()} points "
                     f"(worst point: Eb/N0={res['eb_n0_db'][mask][np.argmax(dev)]:.0f} dB)")
        lines.append(f"  points with <10 errors (excluded): "
                     f"{int((~mask).sum())}")
    return "\n".join(lines)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    eb_db = np.arange(0, 11, 1.0)

    results = {}
    for scheme in ("BPSK", "QPSK", "QAM16"):
        print(f"Simulating {scheme} ...", flush=True)
        results[scheme] = sweep(scheme, eb_db, **SETTINGS[scheme])

    eb_fine = np.linspace(0, 10, 400)
    plot_ber(results, eb_fine,
             os.path.join(RESULTS_DIR, "ber_vs_snr.png"))
    plot_constellations(os.path.join(RESULTS_DIR, "constellations.png"))

    report = verify(results)
    print("\n" + report)
    with open(os.path.join(RESULTS_DIR, "verification.txt"), "w") as f:
        f.write(report + "\n")

    # compact table: simulated vs theory at selected points
    print("\nBER table (simulated vs theoretical):")
    print(f"{'Eb/N0':>6} | {'BPSK sim':>10} {'BPSK th':>10} | "
          f"{'QPSK sim':>10} {'QPSK th':>10} | "
          f"{'QAM16 sim':>10} {'QAM16 th':>10}")
    for i, snr in enumerate(eb_db):
        row = f"{snr:6.0f} |"
        for scheme in ("BPSK", "QPSK", "QAM16"):
            s = results[scheme]["ber"][i]
            t = float(THEORY[scheme](snr))
            row += f" {s:10.3e} {t:10.3e} |"
        print(row)
    print(f"\nPlots and report saved in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
