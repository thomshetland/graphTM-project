import numpy as np

def build_boards_from_moves(moves: np.ndarray,
                            lengths: np.ndarray,
                            offset: int = 0,
                            signed: bool = False) -> np.ndarray:
    """
    Convert move lists into a SINGLE 2D board per game.

    Parameters
    ----------
    moves : np.ndarray, shape (n_games, D*D), int
        Each row is the move sequence (cell indices, -1 padded).
    lengths : np.ndarray, shape (n_games,), int
        Number of valid moves in each game.
    offset : int
        How many moves before the end to cut off.
        0 -> final board, 1 -> one move before final, etc.
    signed : bool
        If False (default): 0=empty, 1=Player0(X), 2=Player1(O)
        If True:            0=empty, +1=Player0(X), -1=Player1(O)

    Returns
    -------
    X : np.ndarray, shape (n_games, D, D)
        One 2D board per game as described above.
    """
    if moves.ndim != 2:
        raise ValueError(f"`moves` must be 2D, got shape {moves.shape}")
    if lengths.ndim != 1 or lengths.shape[0] != moves.shape[0]:
        raise ValueError("`lengths` must be 1D with same number of rows as `moves`.")

    G, N = moves.shape
    D = int(np.sqrt(N))
    if D * D != N:
        raise ValueError(f"Cannot infer board dimension: N={N} is not a perfect square.")

    # dtype: small integer to save RAM
    X = np.zeros((G, D, D), dtype=np.int8)

    for g in range(G):
        L = int(lengths[g]) - offset
        if L < 0:
            L = 0

        # fill board by applying first L moves
        for t in range(L):
            c = int(moves[g, t])
            if c < 0:
                break

            # map flat index -> 2D coordinates (row-major)
            i, j = divmod(c, D)

            if signed:
                # +1 for Player 0, -1 for Player 1
                X[g, i, j] = 1 if (t % 2) == 0 else -1
            else:
                # 1 for Player 0, 2 for Player 1 (0 = empty)
                X[g, i, j] = 1 if (t % 2) == 0 else 2

    return X

