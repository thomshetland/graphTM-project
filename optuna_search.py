import csv
from time import time

import numpy as np
import optuna
from sklearn.model_selection import train_test_split

from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from src.utils.x_builder import build_boards_from_moves
from src.utils.utils import (
    transform_dataset,
    build_symbol_list,
    boards_to_games_dict,
    build_hex_adjacency,
    build_graphs,
)


def main():
    # -----------------------------
    # Configuration
    # -----------------------------
    # Epochs used during Optuna search (can be lower to speed up tuning)
    epochs_optuna = 50

    # Epochs for final training of best model
    epochs_final = 300

    # Optuna search settings
    n_trials = 20  # change this to run more/less trials

    depth = 2
    hv_bits = 1
    hv_size = 1
    msg_bits = 32
    msg_size = 512
    board_size = 8
    n_board = board_size ** 2

    # -----------------------------
    # Data loading and preprocessing
    # -----------------------------
    print("Loading dataset...")
    dataset = np.load("dataset/hex_8x8_100000.npz")
    moves = dataset["moves"]
    lengths = dataset["lengths"]
    y_ds = dataset["winners"]
    x_ds = build_boards_from_moves(moves, lengths, offset=0)

    print("Final X shape:", x_ds.shape)

    print("Pre-processing training and test dataset")
    # Split 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        x_ds, y_ds, test_size=0.2, random_state=42
    )

    print(X_train[0])

    print("Transforming dataset")
    X_train = transform_dataset(X_train)
    X_test = transform_dataset(X_test)
    print(f"Transformed dataset to {X_train[0]}")

    y_train = y_train.astype(np.uint32)
    y_test = y_test.astype(np.uint32)

    train_graph_length = X_train.shape[0]
    test_graph_length = X_test.shape[0]

    print("Converting board to game dictionaries")
    train_games = boards_to_games_dict(X_train, board_size)
    test_games = boards_to_games_dict(X_test, board_size)

    print("Creating nodes and edges")
    edges = build_hex_adjacency(board_size)

    print("Creating symbols")
    symbols = build_symbol_list(board_size)
    edge_symbols = ["Plain", "Player 1", "Player 2", "Boarder"]
    symbols.extend(edge_symbols)
    hv_size = len(symbols)

    print("Building train graphs")
    graphs_train = build_graphs(
        "Train",
        X=X_train,
        games=train_games,
        symbols=symbols,
        edges=edges,
        board_size=board_size,
        hv_size=hv_size,
        hv_bits=hv_bits,
    )

    print("Building test graphs")
    graphs_test = build_graphs(
        "Test",
        graphs_train,
        X_test,
        test_games,
        symbols,
        edges,
        board_size,
        hv_size,
        hv_bits,
    )

    # -----------------------------
    # CSV for logging ALL trials + epochs (Optuna search)
    # -----------------------------
    csv_filename_optuna = "8x8_optuna_trials_epochs.csv"
    with open(csv_filename_optuna, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trial",
                "epoch",
                "clauses",
                "T",
                "s",
                "train_accuracy",
                "test_accuracy",
                "training_time",
                "inference_time",
                "pruned",
            ]
        )

    # -----------------------------
    # Optuna objective function
    # -----------------------------
    def objective(trial: optuna.Trial) -> float:
        # Hyperparameters to search
        clauses = trial.suggest_int("clauses", 5000, 10000, step=1000)
        T = trial.suggest_int("T", 3000, 10000, step=1000)  # 3000, 13000, ...
        s = trial.suggest_float("s", 1.0, 10.0, step=1.0)

        print(
            f"\n[Trial {trial.number}] Testing params: "
            f"clauses={clauses}, T={T}, s={s}"
        )

        tm = MultiClassGraphTsetlinMachine(
            clauses,
            T,
            s,
            depth=depth,
            message_size=msg_size,
            message_bits=msg_bits,
            grid=(16 * 13, 1, 1),
            block=(128, 1, 1),
        )

        best_test_acc = 0.0

        for epoch in range(1, epochs_optuna + 1):
            # ---- Training ----
            start_training = time()
            tm.fit(graphs_train, y_train, epochs=1, incremental=True)
            stop_training = time()
            training_time = stop_training - start_training

            # ---- Inference ----
            start_testing = time()
            y_pred_test = tm.predict(graphs_test)
            stop_testing = time()
            inference_time = stop_testing - start_testing

            y_pred_train = tm.predict(graphs_train)

            result_test = 100.0 * (y_pred_test == y_test).mean()
            result_train = 100.0 * (y_pred_train == y_train).mean()

            best_test_acc = max(best_test_acc, result_test)

            # Print every epoch accuracy for this trial
            print(
                f"[Trial {trial.number}] Epoch {epoch} | "
                f"Train acc: {result_train:.2f}%, "
                f"Test acc: {result_test:.2f}%, "
                f"Train time: {training_time:.2f}s, "
                f"Test time: {inference_time:.2f}s"
            )

            # Report intermediate value for pruning
            trial.report(best_test_acc, step=epoch)

            # Check if trial should be pruned
            pruned = False
            if trial.should_prune():
                pruned = True

            # Log this epoch (for this trial) to CSV
            with open(csv_filename_optuna, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        trial.number,
                        epoch,
                        clauses,
                        T,
                        s,
                        result_train,
                        result_test,
                        training_time,
                        inference_time,
                        int(pruned),
                    ]
                )

            if pruned:
                print(
                    f"[Trial {trial.number}] Pruned at epoch {epoch} "
                    f"with best_test_acc={best_test_acc:.2f}"
                )
                raise optuna.exceptions.TrialPruned()

        print(
            f"[Trial {trial.number}] Finished with best_test_acc={best_test_acc:.2f}"
        )
        return best_test_acc

    # -----------------------------
    # Run Optuna study
    # -----------------------------
    print("\nStarting Optuna hyperparameter search...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)

    print("\nHyperparameter search finished.")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best test accuracy: {study.best_value:.2f}%")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # -----------------------------
    # Final training with best hyperparameters
    # -----------------------------
    best_clauses = study.best_params["clauses"]
    best_T = study.best_params["T"]
    best_s = study.best_params["s"]

    print(
        f"\nTraining final model with best hyperparameters: "
        f"clauses={best_clauses}, T={best_T}, s={best_s}"
    )

    # Prepare CSV logging for final model
    csv_filename = "8x8_performance_best_params.csv"
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_accuracy",
                "test_accuracy",
                "training_time",
                "inference_time",
                "clauses",
                "T",
                "s",
            ]
        )

    tm = MultiClassGraphTsetlinMachine(
        best_clauses,
        best_T,
        best_s,
        depth=depth,
        message_size=msg_size,
        message_bits=msg_bits,
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    )

    train_acc = []
    test_acc = []
    epoch_list = []
    epoch = 0

    print("Starting final training...")

    for i in range(epochs_final):
        epoch += 1

        # ---- Training ----
        start_training = time()
        tm.fit(graphs_train, y_train, epochs=1, incremental=True)
        stop_training = time()
        training_time = stop_training - start_training

        # ---- Inference ----
        start_testing = time()
        result_test = 100.0 * (tm.predict(graphs_test) == y_test).mean()
        stop_testing = time()
        inference_time = stop_testing - start_testing

        result_train = 100.0 * (tm.predict(graphs_train) == y_train).mean()

        # Store values in lists (optional)
        train_acc.append(result_train)
        test_acc.append(result_test)
        epoch_list.append(epoch)

        # Print to console
        print(
            f"Epoch: {epoch}, "
            f"Train acc: {result_train:.2f}%, "
            f"Test acc: {result_test:.2f}%, "
            f"Training time: {training_time:.2f}s, "
            f"Inference time: {inference_time:.2f}s"
        )

        # ---- Append to CSV ----
        with open(csv_filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    result_train,
                    result_test,
                    training_time,
                    inference_time,
                    best_clauses,
                    best_T,
                    best_s,
                ]
            )


if __name__ == "__main__":
    main()
