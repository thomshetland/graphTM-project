import os
import pandas as pd
import matplotlib.pyplot as plt

# Increase global font sizes
plt.rcParams.update({
    "font.size": 18,          # default text
    "axes.titlesize": 20,     # plot titles
    "axes.labelsize": 18,     # x/y labels
    "legend.fontsize": 16,    # legend text
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

def main():
    base_dir = "results/completed"
    
    # Range of i values
    i_values = range(5, 13)  # 5–12 inclusive

    # Prepare figures
    plt.figure(figsize=(10, 6))
    train_ax = plt.gca()

    plt.figure(figsize=(10, 6))
    test_ax = plt.gca()

    for i in i_values:
        filename = f"{i}x{i}_performance.csv"
        filepath = os.path.join(base_dir, filename)

        if not os.path.isfile(filepath):
            print(f"Warning: file not found, skipping: {filepath}")
            continue

        df = pd.read_csv(filepath)

        # Epochs
        epochs = df['epoch'] if 'epoch' in df.columns else range(1, len(df) + 1)

        if 'train_accuracy' not in df.columns or 'test_accuracy' not in df.columns:
            print(f"Warning: Missing accuracy columns in {filepath}, skipping.")
            continue

        train_acc = df['train_accuracy']
        test_acc = df['test_accuracy']

        # Training plot
        plt.figure(train_ax.figure.number)
        train_ax.plot(epochs, train_acc, label=f"i={i}", linewidth=2)

        # Test plot
        plt.figure(test_ax.figure.number)
        test_ax.plot(epochs, test_acc, label=f"i={i}", linewidth=2)

    # Finalize training plot
    plt.figure(train_ax.figure.number)
    train_ax.set_title("Training Accuracy vs Epoch")
    train_ax.set_xlabel("Epoch")
    train_ax.set_ylabel("Training Accuracy")
    train_ax.set_xlim(0, 300)
    train_ax.legend()
    train_ax.grid(True)
    plt.tight_layout()
    plt.savefig("training_accuracy.png", dpi=300)

    # Finalize test plot
    plt.figure(test_ax.figure.number)
    test_ax.set_title("Test Accuracy vs Epoch")
    test_ax.set_xlabel("Epoch")
    test_ax.set_ylabel("Test Accuracy")
    test_ax.set_xlim(0, 300)
    test_ax.legend()
    test_ax.grid(True)
    plt.tight_layout()
    plt.savefig("test_accuracy.png", dpi=300)

    plt.show()

if __name__ == "__main__":
    main()
