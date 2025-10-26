#!/usr/bin/env python3
import numpy as np
import math

# === CONFIG ===
NPZ_FILE = "dataset/hex_5x5_5000.npz"   # change if your file has a different name
GAME_INDEX = 0                    # which game to print (0 = first one)
# ===============

data = np.load(NPZ_FILE)
print("Loaded file:", NPZ_FILE)
print("Arrays in file:", data.files)
print()

# Print how each array looks
if "moves" in data and "lengths" in data and "winners" in data:
    moves   = data["moves"]
    lengths = data["lengths"]
    winners = data["winners"]

    print("moves shape:", moves.shape)
    print("lengths shape:", lengths.shape)
    print("winners shape:", winners.shape)
    print()

    g = GAME_INDEX
    N = moves.shape[1]
    D = int(math.isqrt(N))
    L = int(lengths[g])
    W = int(winners[g])

    print(f"=== Game {g} ===")
    print(f"Board size: {D}x{D}")
    print(f"Length (moves played): {L}")
    print(f"Winner: Player {W}\n")

    print("Raw moves array for this game:")
    print(moves[g])
    print()

    # Make the board as a grid of integers: 0=empty, 1=player0, 2=player1
    board = np.zeros((D, D), dtype=int)
    for t in range(L):
        cell = moves[g, t]
        if cell < 0:
            break
        r, c = divmod(cell, D)
        board[r, c] = 1 if (t % 2) == 0 else 2

    print("Board as numeric grid (1 = Player0 X, 2 = Player1 O):")
    print(board)
    print()

elif "X" in data and "winners" in data:
    X = data["X"]
    winners = data["winners"]
    print("X shape:", X.shape)
    print("winners shape:", winners.shape)
    print()

    g = GAME_INDEX
    twoN = X.shape[1]
    D = int(math.isqrt(twoN // 2))
    N = D * D
    W = int(winners[g])

    print(f"=== Game {g} ===")
    print(f"Board size: {D}x{D}")
    print(f"Winner: Player {W}\n")

    p0 = X[g, :N].reshape(D, D)
    p1 = X[g, N:].reshape(D, D)
    print("Player 0 board (1 = X):")
    print(p0)
    print("Player 1 board (1 = O):")
    print(p1)

else:
    print("Unrecognized NPZ structure — expected moves/lengths/winners or X/winners.")
