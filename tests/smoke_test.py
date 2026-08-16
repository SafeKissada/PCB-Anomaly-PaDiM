"""Smoke test — เช็คว่า PaDiM pipeline รันจบไม่มี error ก่อนรันจริง"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config
from scripts.run_padim import run


def make_dummy_dataset(root: Path, n_good=20, n_defect=8, size=(64, 64)):
    good_dir = root / "good"
    defect_dir = root / "defect"
    good_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n_good):
        arr = rng.randint(100, 140, (*size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(good_dir / f"good_{i:03d}.png")
    for i in range(n_defect):
        arr = rng.randint(0, 255, (*size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(defect_dir / f"defect_{i:03d}.png")


def main():
    tmp_root = Path("/tmp/padim_smoke_test")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    data_root = tmp_root / "data"
    make_dummy_dataset(data_root)

    cfg = Config(
        DATA_ROOT=str(data_root),
        SPLIT_CACHE_PATH=str(tmp_root / "splits" / "split_assignment.csv"),
        SAVE_PATH=str(tmp_root / "save/logs"),
        OUTPUT_PATH=str(tmp_root / "save/results"),
        IMAGE_SIZE=(64, 64),
        BATCH_SIZE=4,
        NUM_WORKERS=0,
        BACKBONE="resnet18",
        PRETRAINED=False,  # sandbox นี้โหลด pretrained weight ไม่ได้ — รันจริงต้อง True
        FEATURE_LAYERS=("layer1", "layer2"),
        NUM_SELECTED_CHANNELS=32,
        COV_REG_EPSILON=0.01,
        EXPERIMENT="smoke_test",
    )

    val_result, test_result = run(cfg)

    assert val_result.image_scores.shape[0] == len(val_result.labels)
    assert test_result.image_scores.shape[0] == len(test_result.labels)
    assert val_result.pixel_maps.shape[1:] == cfg.IMAGE_SIZE
    assert not np.isnan(val_result.image_scores).any()
    assert not np.isnan(test_result.image_scores).any()

    print("\n✅ SMOKE TEST PASSED (PaDiM) — pipeline รันจบไม่มี error")
    shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
