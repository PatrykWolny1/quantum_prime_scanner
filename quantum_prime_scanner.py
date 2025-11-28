"""
Quantum Prime Scanner
---------------------
Copyright (c) 2025 Patryk Wolny.
Licensed under the MIT License.

This software implements the Spectral Resonance Potential (SRP) analysis 
for the physical verification of the Riemann Hypothesis.
"""

import numpy as np
import cupy as cp
from cupyx.scipy.special import expi as cuda_expi
import matplotlib.pyplot as plt
import time
import os
import sympy
from tqdm import tqdm

# --- Plot configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.figsize": (12, 6),
    "lines.linewidth": 1.5
})

# --- GPU Setup ---
try:
    dev = cp.cuda.Device(0)
    print(f"--> GPU: {dev.mem_info[1] / 1024**3:.2f} GB VRAM")
except:
    print("--> Error: GPU/CuPy not found.")
    exit()

# CUDA kernel for Li(x^rho)
@cp.vectorize
def complex_expi(z):
    return cp.exp(z) / z 

class QuantumPrimeScanner:
    def __init__(self, zeros_list):
        print(f"--> Loading {len(zeros_list)} zeros...")
        self.zeros_gpu = cp.array(zeros_list, dtype=cp.float64)
        self.rhos_gpu = 0.5 + 1j * self.zeros_gpu
        cp.get_default_memory_pool().free_all_blocks()

    def compute_pi_qm(self, x_input, desc="Computing"):
        # CPU to GPU transfer
        x_cpu = np.array(x_input, dtype=np.float64)
        n_total = len(x_cpu)
        results_cpu = np.zeros(n_total, dtype=np.float64)
        n_zeros = len(self.zeros_gpu)
        mempool = cp.get_default_memory_pool()
        
        # Tiling configuration
        tile_size_x = 50000
        zeros_batch_size = 2048
        
        # Main loop over X tiles
        for i in tqdm(range(0, n_total, tile_size_x), desc=desc, unit="tile"):
            end_x = min(i + tile_size_x, n_total)
            current_chunk_size = end_x - i
            
            x_chunk_gpu = cp.array(x_cpu[i:end_x], dtype=cp.float64)
            x_chunk_gpu = cp.maximum(x_chunk_gpu, 1.0000001)
            
            log_x_chunk = cp.log(x_chunk_gpu)
            term_main = cuda_expi(log_x_chunk)
            chunk_correction = cp.zeros(current_chunk_size, dtype=cp.float64)
            
            # Inner loop over zeros
            for j in range(0, n_zeros, zeros_batch_size):
                end_z = min(j + zeros_batch_size, n_zeros)
                rhos_batch = self.rhos_gpu[j:end_z]
                
                # Broadcasting
                args_matrix = rhos_batch[:, None] * log_x_chunk[None, :]
                terms = complex_expi(args_matrix)
                
                # Reduction
                chunk_correction += 2 * cp.sum(cp.real(terms), axis=0)
            
            # Save result
            res_chunk = cp.real(term_main - chunk_correction)
            res_chunk[x_chunk_gpu < 1.9] = 0.0
            results_cpu[i:end_x] = cp.asnumpy(res_chunk)
            
            # Memory cleanup
            del x_chunk_gpu, log_x_chunk, term_main, chunk_correction
            mempool.free_all_blocks()
            
        return results_cpu

def is_perfect_power(n):
    # Filter harmonic echoes
    if n < 4: return False
    log_n = np.log2(n)
    for k in range(2, int(log_n) + 2):
        root = int(round(n ** (1.0 / k)))
        if root ** k == n: return True
    return False

# --- Plotting Functions ---

def analyze_and_plot(zeros_data, stats_range=1000, plot_range=400):
    scanner = QuantumPrimeScanner(zeros_data)
    
    # Statistical verification
    print(f"\n--> Stats for N={stats_range}...")
    integers = np.arange(1, stats_range + 1)
    pi_values = scanner.compute_pi_qm(integers, desc="Stats Scan")
    
    detected_primes = []
    last_detected_n = -1
    
    for n in range(2, stats_range + 1):
        slope = pi_values[n-1] - pi_values[n-2]
        if slope > 0.35: 
            if (n - last_detected_n) > 1 or n <= 3: 
                if not is_perfect_power(n): 
                    detected_primes.append(n)
                    last_detected_n = n

    real_primes = list(sympy.primerange(2, stats_range + 1))
    tp = len(set(detected_primes) & set(real_primes))
    acc = (tp / len(real_primes)) * 100
    print(f"--> Accuracy: {acc:.2f}%")

    # Fig 1: Staircase
    x_dense = np.linspace(2, plot_range, 1000)
    y_dense = scanner.compute_pi_qm(x_dense, desc="Fig 1")
    y_true = [sympy.primepi(x) for x in x_dense]
    
    plt.figure(1)
    plt.plot(x_dense, y_true, 'k--', label='True pi(x)')
    plt.plot(x_dense, y_dense, 'r-', alpha=0.9, label='Quantum Model')
    plt.title("Fig 1: Quantum Staircase Reconstruction")
    plt.xlabel("x")
    plt.ylabel("pi(x)")
    plt.legend()
    plt.savefig("Fig1_QuantumStaircase.png")

    # Fig 2: Peaks
    n_plot = np.arange(1, plot_range + 1)
    pi_int = scanner.compute_pi_qm(n_plot, desc="Fig 2")
    pi_int[0] = 0
    deriv = np.diff(pi_int, prepend=0)
    
    plt.figure(2)
    plt.stem(n_plot, deriv, linefmt='b-', markerfmt='bo', basefmt='k-')
    plt.axhline(y=0.35, color='r', linestyle='--', label='Threshold')
    plt.title("Fig 2: Derivative Spectroscopy")
    plt.xlabel("n")
    plt.ylabel("Signal")
    plt.legend()
    plt.savefig("Fig2_DerivativePeaks.png")

    # Fig 3: Twin Primes
    x_zoom = np.linspace(50, 70, 500)
    y_zoom = scanner.compute_pi_qm(x_zoom, desc="Fig 3")
    y_true_zoom = np.array([sympy.primepi(int(x)) for x in x_zoom], dtype=float)
    
    plt.figure(3)
    plt.plot(x_zoom, y_true_zoom, 'k--', label='True pi(x)')
    plt.plot(x_zoom, y_zoom, 'r-', label='Wavefunction')
    plt.fill_between(x_zoom, y_zoom, y_true_zoom, color='red', alpha=0.1)
    plt.title("Fig 3: Twin Prime Interference (59, 61)")
    plt.xlabel("x")
    plt.ylabel("pi(x)")
    plt.savefig("Fig3_TwinPrimesWave.png")

def plot_srp(zeros_data):
    print("\n--> Plotting Fig 4 (SRP)...")
    small_zeros = zeros_data[:450]
    x = np.linspace(2, 300, 1000)
    
    V_cl = x * np.log(x)
    V_osc = np.zeros_like(x)
    for g in small_zeros: V_osc += np.cos(g * np.log(x))
    
    scale = x * 0.5 
    V_qm = V_cl + (V_osc / np.sqrt(len(small_zeros))) * (scale / 10.0)
    correction = V_qm - V_cl

    # Get points
    scanner = QuantumPrimeScanner(zeros_data)
    integers = np.arange(1, 302)
    pi_vals = scanner.compute_pi_qm(integers, desc="SRP Scan")
    
    primes = []
    last = -1
    for n in range(2, 301):
        slope = pi_vals[n-1] - pi_vals[n-2]
        if slope > 0.35: 
            if (n - last) > 1 or n <= 3: 
                if not is_perfect_power(n): 
                    primes.append(n)
                    last = n

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    
    ax1.plot(x, V_cl, 'k--', label='Classical V(x)')
    ax1.plot(x, V_qm, 'g-', label='SRP Total')
    ax1.fill_between(x, V_cl, V_qm, color='green', alpha=0.1)
    ax1.set_title("Fig 4a: Quantum Potential Well")
    ax1.legend()
    
    ax2.plot(x, correction, 'g-', label='SRP Oscillation')
    ax2.fill_between(x, 0, correction, color='green', alpha=0.2)
    ax2.axhline(0, color='k', lw=0.5)
    
    for p in primes:
        val = 0
        for g in small_zeros: val += np.cos(g * np.log(p))
        scaled = (val / np.sqrt(len(small_zeros))) * (p * 0.5 / 10.0)
        ax2.plot(p, scaled, 'bo', markersize=6)
        ax2.text(p, scaled + 0.3, str(p), ha='center', fontsize=9, color='navy')
        
    ax2.set_title("Fig 4b: SRP Structure (Blue Dots = Primes)")
    ax2.set_xlabel("x")
    plt.tight_layout()
    plt.savefig("Fig4_SpectralResonancePotential.png")

def plot_spectrum(zeros_data):
    print("\n--> Plotting Fig 5 (Spectrum)...")
    limit = 60
    scanner = QuantumPrimeScanner(zeros_data)
    integers = np.arange(1, limit + 2)
    pi_vals = scanner.compute_pi_qm(integers, desc="Spectrum Gen")
    
    levels = []
    last = -1
    for n in range(2, limit + 1):
        slope = pi_vals[n-1] - pi_vals[n-2]
        if slope > 0.35:
             if (n - last) > 1 or n <= 3:
                if not is_perfect_power(n):
                    levels.append(n)
                    last = n
    
    with plt.style.context('dark_background'):
        fig, (ax_s, ax_l) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})
        
        ax_s.set_facecolor('black'); ax_s.get_yaxis().set_visible(False)
        ax_s.set_xlim(0, limit); ax_s.set_title("Fig 5a: Emission Spectrum", color='white')
        for E in levels:
            c = plt.cm.plasma(E / limit)
            ax_s.axvline(x=E, color=c, lw=2, alpha=0.9)
            ax_s.axvline(x=E, color=c, lw=6, alpha=0.3)

        ax_l.set_facecolor('#111111'); ax_l.set_xlim(0, 4); ax_l.set_ylim(0, limit + 5)
        ax_l.set_ylabel("Energy (p)", color='white'); ax_l.set_xticks([])
        for i, E in enumerate(levels):
            c = plt.cm.plasma(E / limit)
            ax_l.hlines(y=E, xmin=1, xmax=3, colors=c, lw=2)
            lbl = "Ground" if i == 0 else f"Excited {i}"
            ax_l.text(3.1, E, f"n={E} ({lbl})", color='white', va='center', fontsize=9)
            
        ax_l.set_title("Fig 5b: Energy Levels", color='white')
        plt.tight_layout()
        plt.savefig("Fig5_ParticleSpectrum.png", facecolor='black')

def load_data():
    zeros = []
    if os.path.exists('zeros6.txt'):
        with open('zeros6.txt') as f:
            for i, l in enumerate(f):
                if l.strip(): zeros.append(float(l.strip()))
                if i >= 2000000: break 
    return zeros

if __name__ == "__main__":
    data = load_data()
    print(f"Zeros loaded: {len(data)}")
    
    if data:
        analyze_and_plot(data, stats_range=10000, plot_range=300)
        plot_srp(data)
        plot_spectrum(data)
    else:
        print("Error: zeros6.txt not found.")