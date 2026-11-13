"""Create deterministic synthetic token files for the CPU quick start."""
from pathlib import Path
import numpy as np


def main():
    root = Path("data/smoke")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, count in (("train", 8192), ("val", 2048)):
        path = root / f"{split}.bin"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        rng.integers(0, 97, size=count, dtype=np.uint16).tofile(path)
    print(f"Wrote synthetic data to {root}")


if __name__ == "__main__":
    main()
