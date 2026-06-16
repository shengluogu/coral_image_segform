#处理日志，得到训练和验证的损失、准确率、mIoU、背景、健康、死亡、霉变类的IoU曲线
import re
import os
import matplotlib.pyplot as plt

# =====================================================
# 修改这里
# =====================================================
LOG_FILE = "logs/tra_sgfmb5_0.6560_416.log"
# =====================================================


def parse_log(log_path):
    epochs = []
    train_losses = []
    val_losses = []
    val_accs = []
    val_mious = []

    bg_iou = []
    healthy_iou = []
    dead_iou = []
    molded_iou = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:

        # Epoch Summary
        m = re.search(r"Epoch\s+(\d+)/\d+\s+Summary", line)
        if m:
            epochs.append(int(m.group(1)))

        # Train Loss
        m = re.search(r"Train Loss:\s*([0-9.]+)", line)
        if m:
            train_losses.append(float(m.group(1)))

        # Val Loss
        m = re.search(r"Val Loss:\s*([0-9.]+)", line)
        if m:
            val_losses.append(float(m.group(1)))

        # Val Acc
        m = re.search(r"Val Acc:\s*([0-9.]+)", line)
        if m:
            val_accs.append(float(m.group(1)))

        # Val mIoU
        m = re.search(r"Val mIoU:\s*([0-9.]+)", line)
        if m:
            val_mious.append(float(m.group(1)))

        # Per-class IoU
        m = re.search(r"Background\s*: IoU = ([0-9.]+)", line)
        if m:
            bg_iou.append(float(m.group(1)))

        m = re.search(r"Healthy\s*: IoU = ([0-9.]+)", line)
        if m:
            healthy_iou.append(float(m.group(1)))

        m = re.search(r"Dead\s*: IoU = ([0-9.]+)", line)
        if m:
            dead_iou.append(float(m.group(1)))

        m = re.search(r"Molded\s*: IoU = ([0-9.]+)", line)
        if m:
            molded_iou.append(float(m.group(1)))

    return {
        "epochs": epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_mious": val_mious,
        "bg_iou": bg_iou,
        "healthy_iou": healthy_iou,
        "dead_iou": dead_iou,
        "molded_iou": molded_iou,
    }


def plot_loss_curve(data):
    plt.figure(figsize=(8, 5))

    plt.plot(
        data["epochs"],
        data["train_losses"],
        linewidth=2,
        label="Train Loss"
    )

    plt.plot(
        data["epochs"],
        data["val_losses"],
        linewidth=2,
        label="Val Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()

    print("Saved: loss_curve.png")


def plot_miou_curve(data):
    plt.figure(figsize=(8, 5))

    plt.plot(
        data["epochs"],
        data["val_mious"],
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Validation mIoU")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("miou_curve.png", dpi=300)
    plt.close()

    print("Saved: miou_curve.png")


def plot_accuracy_curve(data):
    plt.figure(figsize=(8, 5))

    plt.plot(
        data["epochs"],
        data["val_accs"],
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=300)
    plt.close()

    print("Saved: accuracy_curve.png")


def plot_class_iou_curve(data):

    n = min(
        len(data["epochs"]),
        len(data["bg_iou"]),
        len(data["healthy_iou"]),
        len(data["dead_iou"]),
        len(data["molded_iou"])
    )

    if n == 0:
        print("No class IoU information found.")
        return

    epochs = data["epochs"][:n]

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        data["bg_iou"][:n],
        linewidth=2,
        label="Background"
    )

    plt.plot(
        epochs,
        data["healthy_iou"][:n],
        linewidth=2,
        label="Healthy"
    )

    plt.plot(
        epochs,
        data["dead_iou"][:n],
        linewidth=2,
        label="Dead"
    )

    plt.plot(
        epochs,
        data["molded_iou"][:n],
        linewidth=2,
        label="Molded"
    )

    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Per-Class IoU")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("class_iou_curve.png", dpi=300)
    plt.close()

    print("Saved: class_iou_curve.png")


def check_data(data):

    print("\n========== Parsed Results ==========")

    print(f"Epochs       : {len(data['epochs'])}")
    print(f"Train Loss   : {len(data['train_losses'])}")
    print(f"Val Loss     : {len(data['val_losses'])}")
    print(f"Val Acc      : {len(data['val_accs'])}")
    print(f"Val mIoU     : {len(data['val_mious'])}")

    print(f"Background   : {len(data['bg_iou'])}")
    print(f"Healthy      : {len(data['healthy_iou'])}")
    print(f"Dead         : {len(data['dead_iou'])}")
    print(f"Molded       : {len(data['molded_iou'])}")

    print("====================================\n")


def main():

    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError(LOG_FILE)

    data = parse_log(LOG_FILE)

    check_data(data)

    plot_loss_curve(data)
    plot_miou_curve(data)
    plot_accuracy_curve(data)
    plot_class_iou_curve(data)

    print("\nDone.")


if __name__ == "__main__":
    main()