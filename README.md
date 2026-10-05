# Digital Communication System Simulator

A Monte Carlo simulator of a baseband digital communication link: random bit
generation, Gray-coded **BPSK / QPSK / 16-QAM** modulation, an **AWGN**
channel, hard-decision demodulation, and **BER vs Eb/N0** analysis with
theoretical reference curves.

Built as portfolio project #1 for an Electrical Engineering (Communication)
major. No hardware or licence required — pure Python + NumPy/SciPy/Matplotlib.

## Theory in brief

**Transmitter.** A block of pseudo-random bits is grouped into symbols of
*k* bits (k = 1, 2, 4) and mapped to complex constellation points. All
constellations are Gray-coded (adjacent points differ in exactly one bit,
so the most likely symbol error corrupts only one bit) and normalized to
unit average symbol energy, E[|s|²] = 1:

- BPSK:  ±1
- QPSK:  (±1 ± j)/√2
- 16-QAM: (±1, ±3) × (±1, ±3) scaled by 1/√10

**Channel.** The AWGN channel adds circularly symmetric complex Gaussian
noise. For a target Eb/N0 (linear γ) and unit-energy symbols, Eb = 1/k,
N0 = Eb/γ, and each of the real/imaginary noise components has variance
N0/2.

**Receiver.** Hard-decision minimum-Euclidean-distance detection: each
received symbol is decoded to the nearest constellation point, then mapped
back to bits. The bit error rate (BER) is (# bit errors) / (# bits sent).

**Theoretical reference curves** (standard textbook results for Gray-coded
constellations over AWGN):

- BPSK:  Pb = Q(√(2·Eb/N0))
- QPSK:  Pb = Q(√(2·Eb/N0))  — Gray-coded QPSK matches BPSK bit-for-bit,
  because each bit sees an equivalent BPSK channel
- 16-QAM: Pb ≈ (3/8)·erfc(√(0.4·Eb/N0))  (Proakis-style approximation)

## How to run

```bash
pip install -r requirements.txt
python3 run_simulation.py
```

This runs the Monte Carlo sweep (Eb/N0 = 0–10 dB; each point simulates until
100 bit errors are observed, capped at 10M bits for BPSK/QPSK and 2M bits
for 16-QAM) and writes to `results/`:

- `ber_vs_snr.png` — simulated vs theoretical BER curves (log scale)
- `constellations.png` — clean and noisy constellation diagrams per scheme
- `verification.txt` — numerical simulation-vs-theory comparison

Takes on the order of a minute on a modern laptop.

## Results

Monte Carlo sweep, Eb/N0 = 0–10 dB (100 target errors per point; up to 10M
bits for BPSK/QPSK, 2M for 16-QAM; ~7 s runtime). Simulated curves sit on
the theoretical curves across the whole range:

| Eb/N0 (dB) | BPSK sim | BPSK theory | QPSK sim | QPSK theory | 16-QAM sim | 16-QAM theory |
|---|---|---|---|---|---|---|
| 0 | 7.78e-02 | 7.87e-02 | 7.79e-02 | 7.87e-02 | 1.41e-01 | 1.39e-01 |
| 5 | 5.94e-03 | 5.95e-03 | 5.69e-03 | 5.95e-03 | 4.13e-02 | 4.19e-02 |
| 10 | 3.60e-06 | 3.87e-06 | 3.40e-06 | 3.87e-06 | 1.86e-03 | 1.75e-03 |

Agreement (max |log10(sim) − log10(theory)| over points with ≥10 errors):
BPSK 0.050, QPSK 0.056, 16-QAM 0.024 — all within Monte Carlo tolerance.
As expected, Gray-coded QPSK matches BPSK exactly, and 16-QAM needs
~4 dB more Eb/N0 than BPSK for the same BER (the price of 4× spectral
efficiency). Full numbers: `results/verification.txt`; plots:
`results/ber_vs_snr.png`, `results/constellations.png`.

## Project structure

```
digital-comm-simulator/
├── digital_comm_sim/
│   ├── bits.py         # seedable pseudo-random bit generation
│   ├── modulation.py   # Gray-coded BPSK/QPSK/16-QAM mod + demod
│   ├── channel.py      # AWGN channel calibrated to Eb/N0
│   ├── theory.py       # closed-form BER expressions
│   ├── simulate.py     # Monte Carlo BER sweep driver
│   └── plots.py        # BER curves + constellation diagrams
├── run_simulation.py   # end-to-end run: sweep, plots, verification
├── requirements.txt
└── README.md
```

## Possible extensions

- Soft-decision decoding and LLRs
- Fading channels (Rayleigh/Rician)
- Forward error correction (convolutional / LDPC codes)
- Higher-order QAM (64-QAM, 256-QAM) and pulse shaping
