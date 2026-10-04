"""Lightweight demonstration: image, prediction and probability bar chart."""
import argparse
from pathlib import Path

from PIL import Image

from steel_defect.config import DEFAULT_RUN_DIR
from steel_defect.predict import predict_image
from steel_defect.utils import cli_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, help="Image path; omit to enter one at the prompt.")
    parser.add_argument("--model", type=Path, default=DEFAULT_RUN_DIR / "models" / "best.keras")
    parser.add_argument("--save", type=Path, help="Save the demonstration figure, useful for slides.")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a window; requires --save.")
    args = parser.parse_args()
    if args.no_show and args.save is None:
        parser.error("--no-show requires --save")
    image_path = args.image or Path(input("Enter steel image path: ").strip().strip('"'))
    result = predict_image(args.model, image_path)
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    with Image.open(image_path) as image:
        axes[0].imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
    axes[0].axis("off")
    axes[0].set_title(f"Predicted: {result['predicted_class']}\nConfidence: {result['confidence']:.1%}")
    names = list(result["probabilities"])
    values = list(result["probabilities"].values())
    bars = axes[1].barh([name.replace("_", " ") for name in names], values, color="#3267A8")
    axes[1].bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=4)
    axes[1].set(xlim=(0, 1.15), xlabel="Softmax probability", title="Six defect classes")
    fig.suptitle("UCS321 AI Mini Project — Steel Surface Defect Classification")
    print(f"Predicted defect: {result['predicted_class']} | Confidence: {result['confidence']:.2%}")
    for name, value in result["probabilities"].items():
        print(f"  {name:22} {value:.2%}")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=220, bbox_inches="tight")
        print(f"Saved demonstration: {args.save}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    cli_guard(main)
