QUANTUM PRIME SCANNER
=====================
Implementation of the Spectral Resonance Potential (SRP) analysis.

REQUIREMENTS
------------
To run this simulation, you need:
1. NVIDIA GPU (RTX series recommended, min. 4GB VRAM)
2. Python 3.8+
3. CUDA Toolkit installed

LIBRARIES
---------
Install dependencies via pip:
$ pip install numpy cupy-cuda12x matplotlib sympy tqdm

(Note: Replace 'cupy-cuda12x' with the version matching your CUDA installation, e.g., 'cupy-cuda11x' or just 'cupy' if configuring manually).

DATASET
-------
The script requires a file named 'zeros6.txt' in the same directory.
This file should contain the imaginary parts of non-trivial Riemann zeros (one floating-point number per line).
Sample data can be obtained from Andrew Odlyzko's tables.

USAGE
-----
1. Place 'zeros6.txt' inside the script folder.
2. Run the script:
   $ python quantum_prime_scanner.py

OUTPUT
------
The script generates 5 visualization files (PNG) in the working directory:
- Fig1_QuantumStaircase.png
- Fig2_DerivativePeaks.png
- Fig3_TwinPrimesWave.png
- Fig4_SpectralResonancePotential.png
- Fig5_ParticleSpectrum.png

It also prints real-time statistical accuracy (ACC) to the console.

TROUBLESHOOTING (OOM Errors)
----------------------------
If you encounter a "cupy.cuda.memory.OutOfMemoryError", it means the batch size is too large for your GPU's VRAM.

To fix this, open 'quantum_prime_scanner.py' and edit the configuration in the 'compute_pi_qm' method:

1. Reduce 'tile_size_x':
   Change: tile_size_x = 50000  ->  tile_size_x = 10000

2. Reduce 'zeros_batch_size':
   Change: zeros_batch_size = 4096  ->  zeros_batch_size = 1024

Smaller values reduce VRAM usage but may slightly increase computation time.

COPYRIGHT & LICENSE
-------------------
Copyright (C) 2025 Patryk Wolny.

- Source Code: Licensed under the MIT License.
- Article & Figures: Licensed under Creative Commons Attribution 4.0 (CC BY 4.0).

You are free to use, modify, and distribute this work, provided that original authorship 
is credited.