#!/usr/bin/env python3
"""
prepare_massachusetts_dataset.py

Usage:
  /home/samama/venvs/swin/bin/python prepare_massachusetts_dataset.py \
    --extracted /home/samama/Desktop/ML/massachusetts_dataset/extracted \
    --out /home/samama/Desktop/ML/massachusetts_dataset/prepared
"""

from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import shutil

def find_best_base(extracted: Path):
    """Prefer PNG tree, else TIFF tree. Return base Path containing train/val/test subdirs."""
    png_base = extracted / "png"
    tiff_base = extracted / "tiff"
    if png_base.exists():
        return png_base
    if tiff_base.exists():
        return tiff_base
    # fallback: search for directories named 'train' with sibling '*_labels'
    for candidate in extracted.iterdir():
        if candidate.is_dir() and (candidate / "train").exists():
            return candidate
    raise RuntimeError(f"Could not find png/ or tiff/ tree under {extracted}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def convert_mask_to_binary(mask_path: Path, out_path: Path, threshold=128):
    img = Image.open(mask_path).convert("L")
    arr = np.array(img)
    # threshold to binary
    bin_arr = (arr > threshold).astype(np.uint8) * 255
    Image.fromarray(bin_arr).save(out_path)

def prepare(extracted_root: Path, out_root: Path):
    base = find_best_base(extracted_root)
    print(f"Using base folder: {base}")

    splits = ["train", "val", "test"]
    out_images = out_root / "images"
    out_masks = out_root / "masks"
    split_dir = out_root / "splits"
    ensure_dir(out_images); ensure_dir(out_masks); ensure_dir(split_dir)

    total = 0
    for split in splits:
        img_src = base / split
        mask_src = base / f"{split}_labels"
        if not img_src.exists():
            print(f"  - Warning: image folder not found for split '{split}': {img_src} (skipping)")
            continue
        if not mask_src.exists():
            print(f"  - Warning: mask folder not found for split '{split}': {mask_src} (skipping masks)")
        out_img_split = out_images / split
        out_mask_split = out_masks / split
        ensure_dir(out_img_split); ensure_dir(out_mask_split)

        pairs = []
        # iterate images (common image extensions)
        for img_path in sorted(img_src.glob("*")):
            if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                continue
            stem = img_path.stem
            # look for mask with same stem (any extension) in mask_src
            mask_candidates = list(mask_src.glob(stem + ".*")) if mask_src.exists() else []
            if not mask_candidates:
                # sometimes mask filenames are prefixed/suffixed; try simple heuristics
                # search for file that contains stem substring
                if mask_src.exists():
                    for m in mask_src.iterdir():
                        if stem in m.stem:
                            mask_candidates.append(m)
            if not mask_candidates:
                # skip sample if no mask found
                # (we still copy image to keep consistent indexing if desired; here we skip)
                print(f"⚠️  No mask found for image: {img_path.name} (skipping)")
                continue

            # choose first mask candidate
            mask_path = mask_candidates[0]

            # Copy/convert image -> PNG
            out_img_path = out_img_split / (stem + ".png")
            try:
                img = Image.open(img_path).convert("RGB")
                img.save(out_img_path)
            except Exception as e:
                print(f"Failed to process image {img_path}: {e}")
                continue

            # Convert mask -> single-channel binary PNG
            out_mask_path = out_mask_split / (stem + ".png")
            try:
                convert_mask_to_binary(mask_path, out_mask_path, threshold=128)
            except Exception as e:
                print(f"Failed to convert mask {mask_path}: {e}")
                # remove copied image if mask conversion fails
                if out_img_path.exists():
                    out_img_path.unlink()
                continue

            pairs.append((out_img_path.name, out_mask_path.name))
            total += 1

        # save CSV for split
        csv_path = split_dir / f"{split}.csv"
        with open(csv_path, "w") as fh:
            for imname, mname in pairs:
                fh.write(f"{imname},{mname}\n")

        print(f"Prepared split '{split}': {len(pairs)} samples -> images: {out_img_split}, masks: {out_mask_split}")

    print(f"\nTotal prepared samples: {total}")
    print(f"Prepared dataset root: {out_root}")
    return out_root

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted", required=True, type=str, help="Path to extracted archive root (contains png/ or tiff/)")
    parser.add_argument("--out", required=False, type=str, default="./massachusetts_dataset/prepared", help="Output prepared dataset root")
    args = parser.parse_args()

    extracted_root = Path(args.extracted)
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    prepare(extracted_root, out_root)
