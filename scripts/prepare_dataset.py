"""Prepara um recorte reproduzivel do Open Images para a Fase 6.

O script seleciona 40 imagens de cachorro e 40 de carro da divisao de
validacao do Open Images. As caixas oficiais sao convertidas para o formato
YOLO e as imagens sao divididas, por classe, em 32/4/4 para
treino/validacao/teste.
"""

from __future__ import annotations

import csv
import random
import shutil
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


SEED = 572084
CLASS_NAMES = ("Dog", "Car")
SPLIT_COUNTS = {"train": 32, "val": 4, "test": 4}
CLASS_DESCRIPTIONS_URL = (
    "https://storage.googleapis.com/openimages/v7/"
    "oidv7-class-descriptions-boxable.csv"
)
BBOX_ANNOTATIONS_URL = (
    "https://storage.googleapis.com/openimages/v5/"
    "validation-annotations-bbox.csv"
)
IMAGE_URL_TEMPLATE = (
    "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
)


def download(url: str, destination: Path) -> None:
    """Baixa um arquivo somente quando ele ainda nao existe."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)


def load_class_ids(class_csv: Path) -> dict[str, str]:
    with class_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = {display_name: label_name for label_name, display_name in csv.reader(handle)}
    missing = [name for name in CLASS_NAMES if name not in rows]
    if missing:
        raise RuntimeError(f"Classes ausentes no catalogo: {missing}")
    return {name: rows[name] for name in CLASS_NAMES}


def load_candidates(
    bbox_csv: Path, class_ids: dict[str, str]
) -> tuple[dict[str, list[str]], dict[tuple[str, str], list[tuple[float, ...]]]]:
    id_to_name = {label_id: name for name, label_id in class_ids.items()}
    boxes: dict[tuple[str, str], list[tuple[float, ...]]] = defaultdict(list)

    with bbox_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            class_name = id_to_name.get(row["LabelName"])
            if class_name is None:
                continue
            if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1" or row["IsInside"] == "1":
                continue
            xmin = float(row["XMin"])
            xmax = float(row["XMax"])
            ymin = float(row["YMin"])
            ymax = float(row["YMax"])
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes[(row["ImageID"], class_name)].append((xmin, xmax, ymin, ymax))

    image_classes: dict[str, set[str]] = defaultdict(set)
    for image_id, class_name in boxes:
        image_classes[image_id].add(class_name)

    candidates: dict[str, list[str]] = {name: [] for name in CLASS_NAMES}
    for (image_id, class_name), image_boxes in boxes.items():
        if len(image_classes[image_id]) != 1:
            continue
        largest_area = max((xmax - xmin) * (ymax - ymin) for xmin, xmax, ymin, ymax in image_boxes)
        if largest_area >= 0.12:
            candidates[class_name].append(image_id)

    rng = random.Random(SEED)
    for class_name in CLASS_NAMES:
        candidates[class_name] = sorted(set(candidates[class_name]))
        rng.shuffle(candidates[class_name])
    return candidates, boxes


def download_image(image_id: str, destination: Path) -> tuple[str, bool, str]:
    try:
        download(IMAGE_URL_TEMPLATE.format(image_id=image_id), destination)
        with Image.open(destination) as image:
            image.verify()
        return image_id, True, ""
    except Exception as exc:  # noqa: BLE001 - registrar e tentar outro candidato
        destination.unlink(missing_ok=True)
        return image_id, False, str(exc)


def choose_images(
    candidates: dict[str, list[str]], cache_dir: Path
) -> dict[str, list[str]]:
    total_per_class = sum(SPLIT_COUNTS.values())
    selected: dict[str, list[str]] = {name: [] for name in CLASS_NAMES}

    for class_name in CLASS_NAMES:
        pool = candidates[class_name][: max(80, total_per_class * 2)]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    download_image, image_id, cache_dir / "images" / f"{image_id}.jpg"
                ): image_id
                for image_id in pool
            }
            for future in as_completed(futures):
                image_id, ok, _ = future.result()
                if ok:
                    selected[class_name].append(image_id)
        selected[class_name].sort(key=pool.index)
        selected[class_name] = selected[class_name][:total_per_class]
        if len(selected[class_name]) < total_per_class:
            raise RuntimeError(
                f"Somente {len(selected[class_name])} imagens validas para {class_name}."
            )
    return selected


def write_dataset(
    project_root: Path,
    cache_dir: Path,
    selected: dict[str, list[str]],
    boxes: dict[tuple[str, str], list[tuple[float, ...]]],
) -> None:
    dataset_root = project_root / "data" / "openimages_dog_car"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    for split in SPLIT_COUNTS:
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int]] = []
    class_to_id = {"Dog": 0, "Car": 1}

    for class_name in CLASS_NAMES:
        offset = 0
        for split, count in SPLIT_COUNTS.items():
            for image_id in selected[class_name][offset : offset + count]:
                class_id = class_to_id[class_name]
                filename = f"{class_name.lower()}_{image_id}.jpg"
                shutil.copy2(
                    cache_dir / "images" / f"{image_id}.jpg",
                    dataset_root / "images" / split / filename,
                )

                label_path = dataset_root / "labels" / split / filename.replace(".jpg", ".txt")
                yolo_lines = []
                for xmin, xmax, ymin, ymax in boxes[(image_id, class_name)]:
                    x_center = (xmin + xmax) / 2
                    y_center = (ymin + ymax) / 2
                    width = xmax - xmin
                    height = ymax - ymin
                    yolo_lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )
                label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

                manifest_rows.append(
                    {
                        "image_id": image_id,
                        "class_name": class_name.lower(),
                        "class_id": class_id,
                        "split": split,
                        "file_name": filename,
                        "source_url": IMAGE_URL_TEMPLATE.format(image_id=image_id),
                        "annotation_source": BBOX_ANNOTATIONS_URL,
                    }
                )
            offset += count

    with (dataset_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    (dataset_root / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: dog\n"
        "  1: car\n",
        encoding="utf-8",
    )

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in manifest_rows:
        counts[(str(row["class_name"]), str(row["split"]))] += 1
    print("Dataset criado em", dataset_root)
    for class_name in ("dog", "car"):
        print(class_name, {split: counts[(class_name, split)] for split in SPLIT_COUNTS})


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root.parents[1] / "work" / "fase6_openimages_cache"
    class_csv = cache_dir / "oidv7-class-descriptions-boxable.csv"
    bbox_csv = cache_dir / "validation-annotations-bbox.csv"

    print("Baixando metadados oficiais do Open Images...")
    download(CLASS_DESCRIPTIONS_URL, class_csv)
    download(BBOX_ANNOTATIONS_URL, bbox_csv)
    class_ids = load_class_ids(class_csv)
    print("Classes selecionadas:", class_ids)
    candidates, boxes = load_candidates(bbox_csv, class_ids)
    print("Candidatos:", {name: len(values) for name, values in candidates.items()})
    selected = choose_images(candidates, cache_dir)
    write_dataset(project_root, cache_dir, selected, boxes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
