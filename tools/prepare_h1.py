"""Copy (dereference) verified H1 assets/motions and freeze a content-hashed clip manifest."""

import argparse
import json
import shutil
from pathlib import Path

from blinkmimic.checkpoint import asset_sha256, file_sha256
from blinkmimic.h1.h1_constants import load_h1_npz


def prepare(source: Path, destination: Path, overwrite: bool = False) -> dict:
    asset = source / "h1_data/h1_training.xml"
    motions = sorted((source / "data/npz/h1").glob("*.npz"))
    if not asset.is_file() or not motions:
        raise ValueError(
            "Source must contain h1_data/h1_training.xml and data/npz/h1/*.npz"
        )
    expected = asset_sha256(str(asset))
    records = []
    for path in motions:
        p, _, _, _ = load_h1_npz(str(path), expected_asset_hash=expected)
        records.append(
            {"file": path.name, "frames": len(p), "sha256": file_sha256(path)}
        )
    copies = [(asset, destination / "h1_description/h1_training.xml")]
    copies += [
        (p, destination / "h1_description" / p.relative_to(asset.parent))
        for p in sorted((asset.parent / "meshes").rglob("*"))
        if p.is_file()
    ]
    copies += [(p, destination / "npz/h1" / p.name) for p in motions]
    manifest = {"robot": "h1", "asset_sha256": expected, "clips": records}
    manifest_path = destination / "h1_manifest.json"
    # Check every destination before copying; reruns do not overwrite user changes silently.
    for src, dst in copies:
        if dst.exists() and file_sha256(src) != file_sha256(dst) and not overwrite:
            raise FileExistsError(f"{dst} differs; use --overwrite to replace it")
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text()) != manifest
        and not overwrite
    ):
        raise FileExistsError(f"{manifest_path} differs; use --overwrite to replace it")
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or file_sha256(src) != file_sha256(dst):
            shutil.copy2(src, dst, follow_symlinks=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source", type=Path, required=True, help="successful SuperMimic H1 repository"
    )
    ap.add_argument(
        "--destination", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    manifest = prepare(a.source.resolve(), a.destination.resolve(), a.overwrite)
    print(
        f"Prepared {len(manifest['clips'])} H1 clips; manifest: {a.destination / 'h1_manifest.json'}"
    )
