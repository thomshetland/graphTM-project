# dump_npz.py
import os
import numpy as np
import ctypes as ct
from numpy.ctypeslib import ndpointer

D = 12         # must match BOARD_DIM used to compile the C lib
N = D * D
NG = 100000               # number of games to generate
SEED = 12345

# Load library (choose the right name per platform)
libname_candidates = ["./libhex.so", "./libhex.dylib", "./hexlib.dll", "./libhex.dll"]
for cand in libname_candidates:
    if os.path.exists(cand):
        LIBPATH = cand
        break
else:
    raise FileNotFoundError("Shared library not found (libhex.so/.dylib/.dll)")

lib = ct.CDLL(os.path.abspath(LIBPATH))

# Signatures
lib.generate_games.argtypes = [
    ct.c_int,
    ndpointer(np.int32, flags="C_CONTIGUOUS"),  # moves_out
    ndpointer(np.int32, flags="C_CONTIGUOUS"),  # winners_out
    ndpointer(np.int32, flags="C_CONTIGUOUS"),  # lengths_out
    ct.c_uint,                                  # seed
]
lib.generate_games.restype = ct.c_int

# Allocate outputs
moves   = np.empty((NG, N), dtype=np.int32)
winners = np.empty(NG, dtype=np.int32)
lengths = np.empty(NG, dtype=np.int32)

# Call C batch generator
rc = lib.generate_games(NG, moves, winners, lengths, ct.c_uint(SEED))
if rc != 0:
    raise RuntimeError(f"generate_games returned error code {rc}")

# Cast to compact dtypes to save space
moves16   = moves.astype(np.int16, copy=False)     # N<=225 fits in int16 for D<=15
lengths16 = lengths.astype(np.int16, copy=False)
winners8  = winners.astype(np.uint8, copy=False)

# Save to NPZ
out_path = f"hex_{D}x{D}_{NG}.npz"
np.savez_compressed(out_path, moves=moves16, lengths=lengths16, winners=winners8)
print(f"Saved {NG} games to {out_path}")
