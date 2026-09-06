import sys
from pathlib import Path

CLASS_NAMES = [
    "healthy",
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "powdery_mildew",
]

DATA_DIR = Path("dataset")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_invalid_images(folder: Path):
    invalid = []
    for file_path in folder.rglob("*"):
        if file_path.is_file():
            if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                invalid.append(str(file_path))
    return invalid


def image_count(folder: Path) -> int:
    return sum(
        1
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def main():
    if not DATA_DIR.exists():
        print("Dataset folder not found: dataset", file=sys.stderr)
        sys.exit(1)

    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print("Dataset must contain train/ and val/ folders.", file=sys.stderr)
        sys.exit(1)

    missing = []
    for class_name in CLASS_NAMES:
        train_class_dir = TRAIN_DIR / class_name
        val_class_dir = VAL_DIR / class_name
        if not train_class_dir.exists():
            missing.append(f"Missing training folder: {train_class_dir}")
        if not val_class_dir.exists():
            missing.append(f"Missing validation folder: {val_class_dir}")

    if missing:
        print("Dataset structure is invalid:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        sys.exit(1)

    train_invalid = find_invalid_images(TRAIN_DIR)
    val_invalid = find_invalid_images(VAL_DIR)

    all_invalid = train_invalid + val_invalid
    if all_invalid:
        print("These files have unsupported extensions:")
        for item in all_invalid:
            print(f"  - {item}")
        sys.exit(1)

    train_count = image_count(TRAIN_DIR)
    val_count = image_count(VAL_DIR)

    print("Dataset validation successful.")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Classes checked: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
