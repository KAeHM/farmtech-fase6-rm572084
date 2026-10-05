"""Valida o dataset YOLO e gera uma amostra visual com caixas."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {0: "#22c55e", 1: "#3b82f6"}
NAMES = {0: "dog", 1: "car"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "data" / "openimages_dog_car"
    evidence = root / "evidencias"
    evidence.mkdir(exist_ok=True)

    counts = {}
    samples: list[tuple[Path, Path]] = []
    for split in ("train", "val", "test"):
        images = sorted((dataset / "images" / split).glob("*.jpg"))
        labels = sorted((dataset / "labels" / split).glob("*.txt"))
        if len(images) != len(labels):
            raise RuntimeError(f"Quantidade divergente em {split}: {len(images)} imagens/{len(labels)} labels")
        counts[split] = len(images)
        for image_path in images:
            label_path = dataset / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            with Image.open(image_path) as image:
                image.verify()
            for line in label_path.read_text(encoding="utf-8").splitlines():
                class_id, xc, yc, width, height = line.split()
                values = [float(xc), float(yc), float(width), float(height)]
                if int(class_id) not in NAMES or not all(0.0 <= value <= 1.0 for value in values):
                    raise ValueError(f"Rotulo invalido em {label_path}: {line}")
            if split == "test":
                samples.append((image_path, label_path))
    if counts != {"train": 64, "val": 8, "test": 8}:
        raise RuntimeError(f"Divisao inesperada: {counts}")

    thumb_w, thumb_h = 360, 260
    canvas = Image.new("RGB", (thumb_w * 4, thumb_h * 2), "#111827")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (image_path, label_path) in enumerate(samples):
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            source.thumbnail((thumb_w, thumb_h - 28))
            tile = Image.new("RGB", (thumb_w, thumb_h), "#111827")
            x_offset = (thumb_w - source.width) // 2
            y_offset = 24 + (thumb_h - 24 - source.height) // 2
            tile.paste(source, (x_offset, y_offset))
            tile_draw = ImageDraw.Draw(tile)
            for line in label_path.read_text(encoding="utf-8").splitlines():
                class_id_s, xc_s, yc_s, width_s, height_s = line.split()
                class_id = int(class_id_s)
                xc, yc, width, height = map(float, (xc_s, yc_s, width_s, height_s))
                x1 = x_offset + (xc - width / 2) * source.width
                y1 = y_offset + (yc - height / 2) * source.height
                x2 = x_offset + (xc + width / 2) * source.width
                y2 = y_offset + (yc + height / 2) * source.height
                tile_draw.rectangle((x1, y1, x2, y2), outline=COLORS[class_id], width=3)
                tile_draw.text((x1 + 3, max(25, y1 + 3)), NAMES[class_id], fill=COLORS[class_id], font=font)
            tile_draw.text((8, 7), image_path.name, fill="white", font=font)
            x = (index % 4) * thumb_w
            y = (index // 4) * thumb_h
            canvas.paste(tile, (x, y))

    output = evidence / "dataset_teste_rotulado.jpg"
    canvas.save(output, quality=92)
    print("Validacao concluida:", counts)
    print("Amostra:", output)


if __name__ == "__main__":
    main()
