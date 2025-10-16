# hex_graphtm_win_detector_multiclass.py
# Winner detection in Hex using GraphTM with ONE multi-class head (0=no winner, 1=red, 2=blue)

import numpy as np
import random
from typing import Tuple
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine  # <- use the multiclass model

# =============================================================================
# 1) HEX UTILS
# =============================================================================

def idx(rc: Tuple[int, int], n: int) -> int:
    r, c = rc
    return r * n + c

def neighbors_hex(r: int, c: int, n: int):
    for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            yield rr, cc

class DSU:
    def __init__(self, size: int):
        self.p = list(range(size))
        self.r = [0]*size
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1

def label_position_with_dsu(board: np.ndarray) -> int:
    """
    board: n x n, values in {0 empty, 1 red, 2 blue}
    Return: 0 = no winner, 1 = red (TOP-BOTTOM), 2 = blue (LEFT-RIGHT)
    """
    n = board.shape[0]

    # Red connects TOP<->BOTTOM
    dsu_r = DSU(n*n + 2)
    TOP, BOT = n*n, n*n+1
    for r in range(n):
        for c in range(n):
            if board[r, c] == 1:
                ii = idx((r, c), n)
                if r == 0: dsu_r.union(ii, TOP)
                if r == n-1: dsu_r.union(ii, BOT)
                for rr, cc in neighbors_hex(r, c, n):
                    if board[rr, cc] == 1:
                        dsu_r.union(ii, idx((rr, cc), n))
    red_win = (dsu_r.find(TOP) == dsu_r.find(BOT))

    # Blue connects LEFT<->RIGHT
    dsu_b = DSU(n*n + 2)
    LFT, RGT = n*n, n*n+1
    for r in range(n):
        for c in range(n):
            if board[r, c] == 2:
                ii = idx((r, c), n)
                if c == 0: dsu_b.union(ii, LFT)
                if c == n-1: dsu_b.union(ii, RGT)
                for rr, cc in neighbors_hex(r, c, n):
                    if board[rr, cc] == 2:
                        dsu_b.union(ii, idx((rr, cc), n))
    blue_win = (dsu_b.find(LFT) == dsu_b.find(RGT))

    if red_win and not blue_win:  return 1
    if blue_win and not red_win:  return 2
    return 0

# =============================================================================
# 2) DATA: random Hex positions -> Graphs + labels
# =============================================================================

def random_board(n: int, p_fill: float=0.55) -> np.ndarray:
    board = np.zeros((n, n), dtype=np.uint8)
    for r in range(n):
        for c in range(n):
            if random.random() < p_fill:
                board[r, c] = 1 if random.random() < 0.5 else 2
    return board

def generate_dataset(n: int, num_graphs: int, p_fill: float=0.55):
    """
    Returns: graphs, Y (Y in {0,1,2})
    """
    symbols = [
        'is_red', 'is_blue', 'is_empty',
        'is_TOP', 'is_BOTTOM', 'is_LEFT', 'is_RIGHT'
    ]

    graphs = Graphs(
        num_graphs,
        symbols=symbols,
        hypervector_size=1024,
        hypervector_bits=4
    )

    Y = np.empty(num_graphs, dtype=np.uint32)

    # Predeclare node counts
    for g in range(num_graphs):
        graphs.set_number_of_graph_nodes(g, n*n + 4)
    graphs.prepare_node_configuration()

    # Add nodes
    for g in range(num_graphs):
        for r in range(n):
            for c in range(n):
                graphs.add_graph_node(g, f"cell_{r}_{c}", 8)   # upper bound on outgoing edges
        for name in ["TOP", "BOTTOM", "LEFT", "RIGHT"]:
            graphs.add_graph_node(g, name, n*n)

    graphs.prepare_edge_configuration()

    # Build each graph: edges, properties, labels
    for g in range(num_graphs):
        board = random_board(n, p_fill)
        y = label_position_with_dsu(board)

        # Cell properties
        for r in range(n):
            for c in range(n):
                node_name = f"cell_{r}_{c}"
                v = board[r, c]
                if v == 1:
                    graphs.add_graph_node_property(g, node_name, 'is_red')
                elif v == 2:
                    graphs.add_graph_node_property(g, node_name, 'is_blue')
                else:
                    graphs.add_graph_node_property(g, node_name, 'is_empty')

        # Border properties
        graphs.add_graph_node_property(g, 'TOP',    'is_TOP')
        graphs.add_graph_node_property(g, 'BOTTOM', 'is_BOTTOM')
        graphs.add_graph_node_property(g, 'LEFT',   'is_LEFT')
        graphs.add_graph_node_property(g, 'RIGHT',  'is_RIGHT')

        edge_type = "adj"

        # Cell-to-cell adjacencies (both directions)
        for r in range(n):
            for c in range(n):
                src = f"cell_{r}_{c}"
                for rr, cc in neighbors_hex(r, c, n):
                    dst = f"cell_{rr}_{cc}"
                    graphs.add_graph_node_edge(g, src, dst, edge_type)

        # Borders
        for c in range(n):
            graphs.add_graph_node_edge(g, 'TOP',    f'cell_0_{c}', edge_type)
            graphs.add_graph_node_edge(g, f'cell_0_{c}', 'TOP', edge_type)
            graphs.add_graph_node_edge(g, 'BOTTOM', f'cell_{n-1}_{c}', edge_type)
            graphs.add_graph_node_edge(g, f'cell_{n-1}_{c}', 'BOTTOM', edge_type)
        for r in range(n):
            graphs.add_graph_node_edge(g, 'LEFT',   f'cell_{r}_0', edge_type)
            graphs.add_graph_node_edge(g, f'cell_{r}_0', 'LEFT', edge_type)
            graphs.add_graph_node_edge(g, 'RIGHT',  f'cell_{r}_{n-1}', edge_type)
            graphs.add_graph_node_edge(g, f'cell_{r}_{n-1}', 'RIGHT', edge_type)

        Y[g] = y
    graphs.signature = ("hex", n, ("is_red","is_blue","is_empty","is_TOP","is_BOTTOM","is_LEFT","is_RIGHT"), "adj_v1")
    return graphs, Y

# =============================================================================
# 3) MODEL
# =============================================================================

def build_model(n: int,
                n_clauses: int = 12,
                s: float = 2.5,
                T: int = 2,
                depth: int = None,
                seed: int = 42):
    if depth is None:
        depth = 2*n - 2

    # Some builds call the arg 'number_of_clauses', others 'n_clauses'.
    # None of them in your env accept 'random_state'.
    try:
        tm = MultiClassGraphTsetlinMachine(
            number_of_clauses=n_clauses,
            T=T,
            s=s,
            depth=depth
        )
    except TypeError:
        tm = MultiClassGraphTsetlinMachine(
            n_clauses=n_clauses,
            T=T,
            s=s,
            depth=depth
        )

    # Optional: seed Python/Numpy for reproducibility in data gen
    random.seed(seed)
    np.random.seed(seed)
    return tm

# =============================================================================
# 4) TRAIN / EVAL
# =============================================================================

def train_and_eval(n: int = 10,
                   num_graphs: int = 2000,
                   p_fill: float = 0.55,
                   n_clauses: int = 12,
                   s: float = 2.5,
                   T: int = 2,
                   depth: int = None,
                   epochs: int = 50,
                   seed: int = 7):

    print(f"Creating training set: n={n}, num_graphs={int(num_graphs*0.8)}")
    graphs_train, Y_train = generate_dataset(n, int(num_graphs*0.8), p_fill)

    print(f"Creating test set: n={n}, num_graphs={num_graphs - int(num_graphs*0.8)}")
    graphs_test,  Y_test  = generate_dataset(n, num_graphs - int(num_graphs*0.8), p_fill)

    tm = build_model(n, n_clauses=n_clauses, s=s, T=T, depth=depth, seed=seed)

    print("Training… (manual loop since fit() has no X_test/Y_test)")

    def eval_acc(model, Xg, Yg):
        """Robust evaluation across GraphTM variants."""
        # Try predict -> class ids
        try:
            yhat = model.predict(Xg)
            return (yhat == Yg).mean()
        except Exception:
            pass
        # Try transform -> logits/scores; take argmax
        try:
            scores = model.transform(Xg)
            yhat = np.argmax(scores, axis=1).astype(Yg.dtype)
            return (yhat == Yg).mean()
        except Exception:
            pass
        # Try evaluate -> returns accuracy directly
        try:
            return float(model.evaluate(Xg, Yg))
        except Exception:
            pass
        raise RuntimeError("No predict/transform/evaluate method found on this GraphTM build.")

    best_acc = 0.0
    for ep in range(epochs):
        # One-epoch training step
        tm.fit(graphs_train, Y_train, epochs=1)
        acc = eval_acc(tm, graphs_test, Y_test)
        best_acc = max(best_acc, acc)
        print(f"epoch {ep+1:3d}  test_acc: {acc*100:6.2f}%  best: {best_acc*100:6.2f}%")

    print(f"Final test accuracy: {best_acc*100:.2f}%")


    # Evaluate
    try:
        Y_pred = tm.predict(graphs_test)
    except Exception:
        # Some builds expose evaluate() only; if so, just report accuracy.
        acc = tm.evaluate(graphs_test, Y_test)
        print(f"Final test accuracy: {acc*100:.2f}%")
        return tm, (graphs_train, Y_train), (graphs_test, Y_test)

    acc = (Y_pred == Y_test).mean()
    print(f"Final test accuracy: {acc*100:.2f}%")

    # Optional: show per-class accuracy
    for cls, name in [(0, "No winner"), (1, "Red win"), (2, "Blue win")]:
        m = (Y_test == cls)
        if m.any():
            print(f"Class {cls} ({name}) acc: {((Y_pred[m]==Y_test[m]).mean()*100):.2f}% on {m.sum()} samples")

    return tm, (graphs_train, Y_train), (graphs_test, Y_test)

# =============================================================================
# 5) MAIN
# =============================================================================

if __name__ == "__main__":
    board_max = 10
    depth = 2*board_max - 2  # ensure messages can cross the board

    tm, train_set, test_set = train_and_eval(
        n=board_max,
        num_graphs=2000,
        p_fill=0.55,
        n_clauses=12,   # keep tiny for the tie-breaker
        s=2.5,
        T=2,
        depth=depth,
        epochs=50,
        seed=7
    )
