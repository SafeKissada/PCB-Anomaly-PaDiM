"""
Data pipeline: good/defect directory scanning, the good(3-way)/defect(2-way)
train/val/test split, the AnomalyDataset, image transforms, and DataLoader
factory.
"""
import logging
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

logger = logging.getLogger("ConvNeXtAutoencoder")


def to_grayscale(img: Image.Image) -> Image.Image:
    """Convert an RGB PIL image to grayscale (no equalization), replicated
    back into 3 channels (RGB) so it stays compatible with a pretrained
    3-channel backbone."""
    gray = img.convert("L")             # [H, W] single channel
    return gray.convert("RGB")          # replicate L -> R=G=B


def grayscale_equalize(img: Image.Image) -> Image.Image:
    """Convert an RGB PIL image to grayscale then apply histogram equalization.

    The single equalized channel is replicated back into 3 channels (RGB) so
    the output stays compatible with a pretrained 3-channel backbone
    (e.g. ConvNeXt expects 3-channel input + ImageNet normalization).
    """
    gray = np.array(img.convert("L"))          # [H, W] uint8
    equalized = cv2.equalizeHist(gray)          # [H, W] uint8, contrast-stretched
    equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(equalized_rgb)


def _apply_clahe(gray: np.ndarray, clip_limit: float, tile_grid_size: tuple) -> np.ndarray:
    """Shared CLAHE core used by both grayscale_clahe() and
    grayscale_equalize_clahe(). Kept as one function so the two callers
    can never diverge in how CLAHE itself is parameterized/applied.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)  # [H, W] uint8


def grayscale_clahe(
    img: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> Image.Image:
    """Convert an RGB PIL image to grayscale, then apply CLAHE
    (Contrast Limited Adaptive Histogram Equalization) directly — i.e.
    WITHOUT a prior global equalizeHist() pass.

    Unlike cv2.equalizeHist (global equalization, can over-amplify noise
    in near-uniform regions), CLAHE operates on local tiles
    (tile_grid_size) and clips the histogram at clip_limit before
    redistributing the clipped count, which bounds noise amplification.

    Output is replicated to 3 channels (RGB) for backbone compatibility,
    same convention as grayscale_equalize().
    """
    gray = np.array(img.convert("L"))                       # [H, W] uint8
    clahe_out = _apply_clahe(gray, clip_limit, tile_grid_size)
    clahe_rgb = cv2.cvtColor(clahe_out, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(clahe_rgb)


def grayscale_equalize_clahe(
    img: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> Image.Image:
    """Convert an RGB PIL image to grayscale, apply global equalizeHist(),
    THEN apply CLAHE on top of the equalized result.

    This composes grayscale_equalize()'s global contrast stretch with a
    subsequent local/adaptive pass. Note this is a genuinely different
    signal than grayscale_clahe() (CLAHE alone) — equalizeHist() first
    changes the global histogram shape CLAHE then operates on, so the two
    "+CLAHE" modes are not interchangeable and should be treated as
    distinct ablation arms, not variants of the same thing.
    """
    gray = np.array(img.convert("L"))          # [H, W] uint8
    equalized = cv2.equalizeHist(gray)          # [H, W] uint8, global equalize first
    clahe_out = _apply_clahe(equalized, clip_limit, tile_grid_size)
    clahe_rgb = cv2.cvtColor(clahe_out, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(clahe_rgb)


class Grayscale:
    """torchvision-style transform wrapper around `to_grayscale` (no equalization).

    Insert into a `v2.Compose([...])` pipeline (operates on PIL images,
    so place it BEFORE `v2.ToImage()` / `v2.ToDtype()` / `v2.Normalize()`).
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        return to_grayscale(img)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class GrayscaleEqualize:
    """torchvision-style transform wrapper around `grayscale_equalize`.

    Insert into a `v2.Compose([...])` pipeline (operates on PIL images,
    so place it BEFORE `v2.ToImage()` / `v2.ToDtype()` / `v2.Normalize()`).
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        return grayscale_equalize(img)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class GrayscaleCLAHE:
    """torchvision-style transform wrapper around `grayscale_clahe`.

    clip_limit / tile_grid_size are bound at construction time (from
    cfg.CLAHE_CLIP_LIMIT / cfg.CLAHE_TILE_GRID_SIZE in build_transforms())
    so they can't silently drift from the config the run was recorded with.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        return grayscale_clahe(img, self.clip_limit, self.tile_grid_size)

    def __repr__(self):
        return (f"{self.__class__.__name__}(clip_limit={self.clip_limit}, "
                f"tile_grid_size={self.tile_grid_size})")


class GrayscaleEqualizeCLAHE:
    """torchvision-style transform wrapper around `grayscale_equalize_clahe`
    (global equalizeHist() followed by CLAHE)."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        return grayscale_equalize_clahe(img, self.clip_limit, self.tile_grid_size)

    def __repr__(self):
        return (f"{self.__class__.__name__}(clip_limit={self.clip_limit}, "
                f"tile_grid_size={self.tile_grid_size})")


def _list_labeled_files(cfg) -> pd.DataFrame:
    """Scan cfg.DATA_ROOT/{cfg.GOOD_DIRNAME, cfg.DEFECT_DIRNAME}. Label comes
    directly from which of the two folders a file is under — no filename
    keyword guessing, so there is no "ambiguous keyword" case to drop
    (removes the class of bugs described in the fix log under 2.8/2.9).
    """
    root = Path(cfg.DATA_ROOT)
    rows = []
    for label, dirname in [("normal", cfg.GOOD_DIRNAME), ("anomaly", cfg.DEFECT_DIRNAME)]:
        d = root / dirname
        if not d.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {d}. cfg.DATA_ROOT must contain "
                f"exactly two subfolders: cfg.GOOD_DIRNAME ({cfg.GOOD_DIRNAME!r}) "
                f"and cfg.DEFECT_DIRNAME ({cfg.DEFECT_DIRNAME!r}).")
        files = sorted(
            f for f in d.rglob("*")
            if f.is_file() and f.suffix.lower() in cfg.VALID_EXT
        )
        if not files:
            logger.warning(f"No valid image files found under {d}")
        for f in files:
            rows.append({"path": str(f), "filename": f.name, "label": label})

    if not rows:
        raise FileNotFoundError(
            f"No valid image files found under {root}/{{{cfg.GOOD_DIRNAME},"
            f"{cfg.DEFECT_DIRNAME}}}")

    df = pd.DataFrame(rows)
    df = df.sort_values("path", kind="stable").reset_index(drop=True)

    if cfg.GROUP_ID_REGEX:
        pattern = re.compile(cfg.GROUP_ID_REGEX)
        def _extract_group_id(fn):
            m = pattern.match(fn)
            if not m:
                raise ValueError(
                    f"GROUP_ID_REGEX {cfg.GROUP_ID_REGEX!r} did not match "
                    f"filename {fn!r}. Every filename under DATA_ROOT must "
                    f"match this pattern, or set GROUP_ID_REGEX=None to "
                    f"split per-file instead of per-group.")
            return m.group(1)
        df["group_id"] = df["filename"].apply(_extract_group_id)
    else:
        df["group_id"] = df["path"]

    return df


def _split_good_three_way(sub: pd.DataFrame, ratios, rng: np.random.RandomState) -> dict:
    """Split the "normal" (good) group_ids into train/val/test using the
    full SPLIT_RATIOS tuple (e.g. 70/15/15). This is the ONLY class that ever
    gets a train share. Returns {group_id: split_name}.
    """
    train_r, val_r, _test_r = ratios
    group_ids = sorted(sub["group_id"].unique())  # deterministic order first
    rng.shuffle(group_ids)                         # then seeded shuffle
    n = len(group_ids)

    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))

    assigned = {}
    for gid in group_ids[:n_train]:
        assigned[gid] = "train"
    for gid in group_ids[n_train:n_train + n_val]:
        assigned[gid] = "val"
    for gid in group_ids[n_train + n_val:]:
        assigned[gid] = "test"
    return assigned


def _split_defect_two_way(sub: pd.DataFrame, ratios, rng: np.random.RandomState) -> dict:
    """Split the "anomaly" (defect) group_ids into val/test ONLY.

    There is no train branch here at all — it is not possible for this
    function to assign a defect group_id to "train", by construction, not by
    a config flag that could be turned off. The val:test split uses the
    val:test portion of `ratios`, renormalized to sum to 1 since there is no
    train share for this label (e.g. (0.70, 0.15, 0.15) -> defect val/test
    50/50). Returns {group_id: split_name}, values in {"val", "test"} only.
    """
    _train_r, val_r, test_r = ratios
    val_share = val_r / (val_r + test_r)
    group_ids = sorted(sub["group_id"].unique())  # deterministic order first
    rng.shuffle(group_ids)                         # then seeded shuffle
    n = len(group_ids)

    n_val = int(round(n * val_share))

    assigned = {}
    for gid in group_ids[:n_val]:
        assigned[gid] = "val"
    for gid in group_ids[n_val:]:
        assigned[gid] = "test"
    return assigned


def _stratified_group_split(df: pd.DataFrame, ratios, seed: int) -> pd.DataFrame:
    """Assign each row of `df` (must have 'label' and 'group_id' columns) to
    'train' / 'val' / 'test', keeping every group_id entirely inside a
    single split (see GROUP_ID_REGEX above).

    "good" (normal) and "defect" (anomaly) are split by two separate,
    dedicated functions rather than one shared loop with a branch inside it:
      - normal -> _split_good_three_way()  (train + val + test)
      - anomaly -> _split_defect_two_way() (val + test only, never train)
    This keeps "defect can never enter train" a structural property of which
    function defect rows go through, not a runtime condition that a stray
    flag could disable.
    """
    rng = np.random.RandomState(seed)
    assigned = {}

    for label, sub in df.groupby("label"):
        if label == "normal":
            assigned.update(_split_good_three_way(sub, ratios, rng))
        elif label == "anomaly":
            assigned.update(_split_defect_two_way(sub, ratios, rng))
        else:
            raise ValueError(
                f"Unexpected label {label!r} in _stratified_group_split(); "
                f"expected only 'normal' or 'anomaly' (from _list_labeled_files()).")

    out = df.copy()
    out["split"] = out["group_id"].map(assigned)
    return out


def _assert_no_defect_in_train(full_df: pd.DataFrame, cache_path: Path) -> None:
    """Hard safety net: raise (never just warn) if any defect/anomaly file
    ended up in the train split, whether the split was just computed or
    loaded from a (possibly stale/hand-edited) cache file. Defect images
    must never be usable for training or scoring-as-train — there is no
    config flag to bypass this check.
    """
    n_anomaly_in_train = int(((full_df["label"] == "anomaly")
                              & (full_df["split"] == "train")).sum())
    if n_anomaly_in_train > 0:
        raise RuntimeError(
            f"Data integrity violation: {n_anomaly_in_train} defect/anomaly "
            f"file(s) are assigned to the train split in {cache_path}. "
            f"Defect images must NEVER appear in train — not for training, "
            f"not for scoring. Delete {cache_path} and re-run to recompute "
            f"a correct split (or, if the file was hand-edited, fix/remove "
            f"the offending row(s)).")


def scan_and_split(cfg) -> dict:
    """Build the train/val/test DataFrames from cfg.DATA_ROOT/{good,defect},
    using a seed-based, class-specific split that is cached at
    cfg.SPLIT_CACHE_PATH.

    Replaces the older workflow of manually pre-sorting images into separate
    train/, val/, test/ folders on disk. Advantages:
      - label comes directly from the good/ vs defect/ folder name, not from
        guessing keywords in the filename -> removes the "ambiguous label,
        silently dropped" class of bugs entirely (no keyword parsing at all).
      - good (normal) images are split into train/val/test (3-way); defect
        (anomaly) images are split into val/test only (2-way) — via two
        separate, dedicated functions (see _split_good_three_way() /
        _split_defect_two_way()) rather than one split call handling both
        classes generically. A defect file can therefore structurally never
        end up in train, and this is verified again below regardless of
        whether the split was freshly computed or loaded from cache.
      - the split is reproducible from one seed + one ratio tuple in config
        -> documentable in a single sentence in the Methodology chapter,
        instead of an undocumented manual folder arrangement.
      - the split is CACHED to cfg.SPLIT_CACHE_PATH the first time it is
        computed. Every subsequent call — including from a different one of
        the E0-E8 ablation experiments — reuses that exact cached split, so
        all experiments are compared on identical train/val/test membership
        (otherwise the split itself would be an uncontrolled confound
        between experiments).

    Returns {'train': df, 'val': df, 'test': df}, each with columns
    path/filename/label. 'train' contains only "normal" (good) rows;
    'val'/'test' each contain a mix of "normal" and "anomaly" rows.
    """
    cache_path = Path(cfg.SPLIT_CACHE_PATH)

    if cache_path.exists():
        logger.info(f"Loading cached split assignment from {cache_path}")
        full_df = pd.read_csv(cache_path)
        missing = [p for p in full_df["path"] if not Path(p).exists()]
        if missing:
            logger.warning(
                f"{len(missing)} path(s) listed in the cached split no "
                f"longer exist on disk (e.g. {missing[0]!r}). The cached "
                f"split may be stale relative to the current contents of "
                f"{cfg.DATA_ROOT}. Delete {cache_path} to force recomputing "
                f"the split if you have added/removed images.")
    else:
        logger.info(f"No cached split at {cache_path} — computing a new one "
                    f"(seed={cfg.SEED}, ratios={cfg.SPLIT_RATIOS}): good -> "
                    f"train/val/test, defect -> val/test only.")
        raw_df = _list_labeled_files(cfg)
        full_df = _stratified_group_split(raw_df, cfg.SPLIT_RATIOS, cfg.SEED)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(cache_path, index=False)
        logger.info(f"Saved split assignment ({len(full_df)} files) to {cache_path}")

    _assert_no_defect_in_train(full_df, cache_path)

    result = {}
    for split_name in ["train", "val", "test"]:
        split_df = (full_df[full_df["split"] == split_name]
                    [["path", "filename", "label"]].reset_index(drop=True))
        split_df.attrs["n_scanned"] = len(split_df)
        split_df.attrs["n_dropped_ambiguous_or_unlabelled"] = 0
        split_df.attrs["source_dir"] = str(cfg.DATA_ROOT)
        logger.info(f"[{split_name:5}] total={len(split_df):,}  "
                    f"{split_df['label'].value_counts().to_dict()}")
        result[split_name] = split_df

    return result


class AnomalyDataset(Dataset):
    """Returns (normalized_tensor, display_tensor, preproc_display_tensor,
    path, label, (orig_w, orig_h)).

    - normalized_tensor      : model input (ImageNet-normalized, color-mode applied)
    - display_tensor         : always the plain RGB image (0..1, no normalize) —
                               used to show the "real photo" regardless of color mode
    - preproc_display_tensor : the actual preprocessed image fed to the model
                               (grayscale / grayscale+equalized), 0..1, no normalize,
                               used for visual comparison. If no color-mode transform
                               is configured (RGB mode), this equals display_tensor.
    """
    MAX_LOAD_RETRIES = 5

    def __init__(self, df: pd.DataFrame, norm_tf, orig_tf, image_size=(224, 224),
                 preproc_tf=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()

        self.norm_tf = norm_tf
        self.orig_tf = orig_tf
        self.preproc_tf = preproc_tf
        self.image_size = image_size
        self.n_fallbacks = 0  # running count of successful fallback substitutions

    def __len__(self):
        return len(self.paths)

    def _load_one(self, idx):
        """Load+transform a single sample. Raises on failure (no fallback here)."""
        path = self.paths[idx]
        label = self.labels[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            ow, oh = img.size
            norm_t = self.norm_tf(img)
            orig_t = self.orig_tf(img)
            preproc_t = self.preproc_tf(img) if self.preproc_tf is not None else orig_t
        return norm_t, orig_t, preproc_t, path, label, (ow, oh)

    def __getitem__(self, idx):
        try:
            return self._load_one(idx)
        except Exception as e:
            logger.error(f"Load failed {self.paths[idx]}: {e}")

        # Bounded retry loop (replaces the old unbounded recursive fallback).
        for attempt in range(self.MAX_LOAD_RETRIES):
            random_idx = random.randint(0, len(self) - 1)
            try:
                sample = self._load_one(random_idx)
                self.n_fallbacks += 1
                logger.warning(
                    f"Substituted index {idx} -> {random_idx} "
                    f"(fallback attempt {attempt + 1}/{self.MAX_LOAD_RETRIES}, "
                    f"total fallbacks so far this run: {self.n_fallbacks})")
                return sample
            except Exception as e:
                logger.error(f"Fallback load also failed {self.paths[random_idx]}: {e}")

        raise RuntimeError(
            f"AnomalyDataset: failed to load a usable sample for index {idx} "
            f"after {self.MAX_LOAD_RETRIES} random fallback attempts. "
            f"Too many corrupt/unreachable files in this split — check the "
            f"data source (e.g. a dropped network drive) rather than retrying "
            f"indefinitely.")


def build_transforms(cfg):
    """Build the ImageNet-normalized transform (with/without augmentation),
    and the plain 'display' transform used for visualisation.

    Color mode is controlled by cfg.USE_GRAYSCALE / cfg.USE_GRAYSCALE_EQUALIZATION
    (see config.py for the 3 supported combinations: RGB / Grayscale /
    Grayscale+Equalization). The color-mode step is applied only to the
    model-input pipelines (imagenet_tf / train_aug_tf); `display_tf` used
    for the "original image" gallery is left untouched so users can still
    see the true original photo regardless of the selected mode.
    """
    color_mode = getattr(cfg, 'COLOR_MODE', 'RGB')
    if color_mode != 'RGB':
        logger.warning(
            f"COLOR_MODE={color_mode!r}: input images are converted before "
            f"the ImageNet-pretrained backbone sees them. This is a known "
            f"domain-shift risk (see Findings 2.10) — interpret color-mode "
            f"ablation results (e.g. E1/E2/E4/E5/E7/E8) with this in mind.")

    clahe_clip = getattr(cfg, 'CLAHE_CLIP_LIMIT', 2.0)
    clahe_tile = getattr(cfg, 'CLAHE_TILE_GRID_SIZE', (8, 8))

    if color_mode == 'GRAYSCALE_EQUALIZATION_CLAHE':
        color_step = [GrayscaleEqualizeCLAHE(clahe_clip, clahe_tile)]
    elif color_mode == 'GRAYSCALE_CLAHE':
        color_step = [GrayscaleCLAHE(clahe_clip, clahe_tile)]
    elif color_mode == 'GRAYSCALE_EQUALIZATION':
        color_step = [GrayscaleEqualize()]
    elif color_mode == 'GRAYSCALE':
        color_step = [Grayscale()]
    elif color_mode == 'RGB':
        color_step = []
    else:
        raise ValueError(f"Unknown cfg.COLOR_MODE: {color_mode!r}")

    imagenet_tf = v2.Compose([
        v2.Resize(cfg.IMAGE_SIZE),
        *color_step,
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    train_aug_tf = v2.Compose([
        v2.Resize(cfg.IMAGE_SIZE),
        v2.ColorJitter(brightness=cfg.AUG_COLOR_JITTER, contrast=cfg.AUG_COLOR_JITTER),
        *color_step,
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    display_tf = v2.Compose([
        v2.Resize(cfg.IMAGE_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    if color_step:
        preproc_display_tf = v2.Compose([
            v2.Resize(cfg.IMAGE_SIZE),
            *color_step,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    else:
        preproc_display_tf = None  # RGB mode: nothing extra to preview

    return imagenet_tf, train_aug_tf, display_tf, preproc_display_tf


def make_loader(ds, cfg, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False,
    )


def build_datasets_and_loaders(cfg):
    """Build val/test datasets/dataloaders plus the normal-only train loader
    (the only loader ever used for training the autoencoder) from
    cfg.DATA_ROOT/{good,defect}, via a cached, seed-based split (see
    scan_and_split()).

    There is deliberately no "full train" Dataset/DataLoader here: the train
    split contains only "normal" (good) rows to begin with (scan_and_split()
    guarantees defect can never land in train), and no code path in this
    project ever needs a train-split Dataset/DataLoader other than
    normal_loader below — training reads only normal_loader, and
    scoring/reporting reads only val_loader/test_loader.
    """
    split = scan_and_split(cfg)
    df_train, df_val, df_test = split["train"], split["val"], split["test"]

    imagenet_tf, train_aug_tf, display_tf, preproc_display_tf = build_transforms(cfg)

    val_ds = AnomalyDataset(df_val, imagenet_tf, display_tf, cfg.IMAGE_SIZE, preproc_display_tf)
    test_ds = AnomalyDataset(df_test, imagenet_tf, display_tf, cfg.IMAGE_SIZE, preproc_display_tf)

    val_loader = make_loader(val_ds, cfg)
    test_loader = make_loader(test_ds, cfg)
    df_train_normal = df_train[df_train["label"] == "normal"].reset_index(drop=True)
    normal_norm_tf = train_aug_tf if cfg.USE_AUGMENTATION else imagenet_tf
    normal_ds = AnomalyDataset(df_train_normal, normal_norm_tf, display_tf, cfg.IMAGE_SIZE, preproc_display_tf)
    normal_loader = make_loader(normal_ds, cfg, shuffle=True)

    return {
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "val_ds": val_ds, "test_ds": test_ds, "normal_ds": normal_ds,
        "val_loader": val_loader, "test_loader": test_loader,
        "normal_loader": normal_loader,
    }