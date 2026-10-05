"""Executa os experimentos obrigatorios da Fase 6.

Experimentos:
1. YOLO customizada por 30 epocas.
2. YOLO customizada por 60 epocas, partindo dos mesmos pesos base.
3. YOLO padrao (COCO), sem ajuste, sobre o conjunto de teste.
4. CNN simples treinada do zero para classificar cachorro x carro.

Os resultados sao gravados em ``resultados/`` e os modelos finais em
``modelos/``. O script reutiliza artefatos existentes para evitar treinos
duplicados; use ``--force`` para recomecar.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO


SEED = 572084
CLASS_NAMES = ["dog", "car"]
COCO_TO_LOCAL = {16: 0, 2: 1}  # dog e car no COCO
IMAGE_SIZE_YOLO = 320
IMAGE_SIZE_CNN = 128


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_float(value: object) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def runtime_yaml(dataset_root: Path, output_dir: Path) -> Path:
    path = output_dir / "dataset_runtime.yaml"
    normalized = dataset_root.resolve().as_posix()
    path.write_text(
        f"path: {normalized}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: dog\n"
        "  1: car\n",
        encoding="utf-8",
    )
    return path


def train_yolo(
    epochs: int,
    data_yaml: Path,
    results_root: Path,
    models_root: Path,
    force: bool,
) -> dict:
    run_name = f"epocas_{epochs}"
    run_dir = results_root / "treinos_yolo" / run_name
    summary_path = results_root / f"metricas_yolo_{epochs}.json"
    copied_best = models_root / f"yolo_custom_{epochs}_best.pt"

    if not force and summary_path.exists() and copied_best.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    model = YOLO("yolov8n.pt")
    started = time.perf_counter()
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=IMAGE_SIZE_YOLO,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0,
        project=str(results_root / "treinos_yolo"),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        seed=SEED,
        deterministic=True,
        patience=epochs,
        plots=True,
        verbose=True,
        cache=True,
    )
    training_seconds = time.perf_counter() - started
    run_dir = Path(model.trainer.save_dir)
    best_path = run_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Pesos nao encontrados: {best_path}")
    models_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, copied_best)

    best_model = YOLO(str(best_path))
    validation = best_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMAGE_SIZE_YOLO,
        batch=8,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0,
        project=str(results_root / "avaliacoes_yolo"),
        name=f"teste_{epochs}",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    rd = validation.results_dict
    summary = {
        "abordagem": f"YOLO customizada - {epochs} epocas",
        "epocas": epochs,
        "tempo_treinamento_s": round(training_seconds, 3),
        "precision": round(as_float(rd.get("metrics/precision(B)", 0.0)), 6),
        "recall": round(as_float(rd.get("metrics/recall(B)", 0.0)), 6),
        "map50": round(as_float(rd.get("metrics/mAP50(B)", 0.0)), 6),
        "map50_95": round(as_float(rd.get("metrics/mAP50-95(B)", 0.0)), 6),
        "fitness": round(as_float(rd.get("fitness", 0.0)), 6),
        "modelo": copied_best.relative_to(models_root.parent).as_posix(),
        "diretorio_treino": run_dir.relative_to(results_root.parent).as_posix(),
    }
    write_json(summary_path, summary)
    return summary


def read_ground_truth(label_path: Path, image_width: int, image_height: int) -> list[dict]:
    ground_truth = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id_s, xc_s, yc_s, width_s, height_s = line.split()
        class_id = int(class_id_s)
        xc, yc, width, height = map(float, (xc_s, yc_s, width_s, height_s))
        ground_truth.append(
            {
                "class_id": class_id,
                "box": [
                    (xc - width / 2) * image_width,
                    (yc - height / 2) * image_height,
                    (xc + width / 2) * image_width,
                    (yc + height / 2) * image_height,
                ],
            }
        )
    return ground_truth


def iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def detection_scores(all_ground_truth: list[list[dict]], all_predictions: list[list[dict]]) -> dict:
    true_positives = false_positives = false_negatives = 0
    for ground_truth, predictions in zip(all_ground_truth, all_predictions, strict=True):
        matched = set()
        for prediction in sorted(predictions, key=lambda row: row["confidence"], reverse=True):
            best_index = None
            best_iou = 0.0
            for index, target in enumerate(ground_truth):
                if index in matched or target["class_id"] != prediction["class_id"]:
                    continue
                candidate_iou = iou(target["box"], prediction["box"])
                if candidate_iou > best_iou:
                    best_iou = candidate_iou
                    best_index = index
            if best_index is not None and best_iou >= 0.5:
                matched.add(best_index)
                true_positives += 1
            else:
                false_positives += 1
        false_negatives += len(ground_truth) - len(matched)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision_iou50": round(precision, 6),
        "recall_iou50": round(recall, 6),
        "f1_iou50": round(f1, 6),
    }


def evaluate_yolo_predictions(
    model_path: str | Path,
    dataset_root: Path,
    output_dir: Path,
    standard_coco: bool,
) -> dict:
    model = YOLO(str(model_path))
    image_paths = sorted((dataset_root / "images" / "test").glob("*.jpg"))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=IMAGE_SIZE_YOLO,
        # Limiar pratico padrao: reduz falsos positivos sem mascarar deteccoes
        # relevantes no pequeno conjunto de teste.
        conf=0.25,
        iou=0.5,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_ground_truth: list[list[dict]] = []
    all_predictions: list[list[dict]] = []
    classification_correct = 0
    inference_times = []

    for image_path, result in zip(image_paths, results, strict=True):
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = dataset_root / "labels" / "test" / f"{image_path.stem}.txt"
        ground_truth = read_ground_truth(label_path, width, height)
        all_ground_truth.append(ground_truth)
        predictions = []
        for box in result.boxes:
            source_class = int(box.cls.item())
            if standard_coco:
                if source_class not in COCO_TO_LOCAL:
                    continue
                class_id = COCO_TO_LOCAL[source_class]
            else:
                class_id = source_class
                if class_id not in (0, 1):
                    continue
            predictions.append(
                {
                    "class_id": class_id,
                    "confidence": float(box.conf.item()),
                    "box": [float(value) for value in box.xyxy[0].tolist()],
                }
            )
        all_predictions.append(predictions)
        expected_class = 0 if image_path.name.startswith("dog_") else 1
        if predictions:
            predicted_class = max(predictions, key=lambda row: row["confidence"])["class_id"]
            classification_correct += int(predicted_class == expected_class)
        inference_times.append(float(result.speed.get("inference", 0.0)))

        plotted = result.plot()
        Image.fromarray(plotted[..., ::-1]).save(output_dir / image_path.name)

    scores = detection_scores(all_ground_truth, all_predictions)
    scores.update(
        {
            "classification_accuracy": round(classification_correct / len(image_paths), 6),
            "mean_inference_ms": round(float(np.mean(inference_times)), 3),
            "test_images": len(image_paths),
        }
    )
    return scores


class ImageClassificationDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str, transform: transforms.Compose):
        self.paths = sorted((dataset_root / "images" / split).glob("*.jpg"))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path = self.paths[index]
        label = 0 if path.name.startswith("dog_") else 1
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, path.name


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.30), nn.Linear(128, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


def classification_metrics(expected: list[int], predicted: list[int]) -> dict:
    matrix = [[0, 0], [0, 0]]
    for target, guess in zip(expected, predicted, strict=True):
        matrix[target][guess] += 1
    accuracy = sum(matrix[i][i] for i in range(2)) / len(expected)
    per_class = {}
    for class_id, name in enumerate(CLASS_NAMES):
        tp = matrix[class_id][class_id]
        fp = sum(matrix[row][class_id] for row in range(2) if row != class_id)
        fn = sum(matrix[class_id][column] for column in range(2) if column != class_id)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "accuracy": round(accuracy, 6),
        "macro_precision": round(float(np.mean([row["precision"] for row in per_class.values()])), 6),
        "macro_recall": round(float(np.mean([row["recall"] for row in per_class.values()])), 6),
        "macro_f1": round(float(np.mean([row["f1"] for row in per_class.values()])), 6),
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def run_cnn(
    dataset_root: Path,
    results_root: Path,
    models_root: Path,
    force: bool,
    epochs: int = 30,
) -> dict:
    summary_path = results_root / "metricas_cnn.json"
    model_path = models_root / "cnn_from_scratch_best.pt"
    if not force and summary_path.exists() and model_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE_CNN, IMAGE_SIZE_CNN)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE_CNN, IMAGE_SIZE_CNN)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = ImageClassificationDataset(dataset_root, "train", train_transform)
    val_ds = ImageClassificationDataset(dataset_root, "val", eval_transform)
    test_ds = ImageClassificationDataset(dataset_root, "test", eval_transform)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history: list[EpochMetrics] = []
    best_val_accuracy = -1.0
    started = time.perf_counter()
    models_root.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_correct = train_total = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += inputs.size(0)

        model.eval()
        val_loss = val_correct = val_total = 0.0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += inputs.size(0)

        row = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss / train_total,
            train_accuracy=train_correct / train_total,
            val_loss=val_loss / val_total,
            val_accuracy=val_correct / val_total,
        )
        history.append(row)
        if row.val_accuracy > best_val_accuracy:
            best_val_accuracy = row.val_accuracy
            torch.save(model.state_dict(), model_path)
        print(
            f"CNN epoca {epoch:02d}/{epochs}: "
            f"loss={row.train_loss:.4f}, acc={row.train_accuracy:.3f}, "
            f"val_loss={row.val_loss:.4f}, val_acc={row.val_accuracy:.3f}"
        )

    training_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    probabilities: list[list[float]] = []
    names: list[str] = []
    inference_times = []
    with torch.no_grad():
        for inputs, labels, filenames in test_loader:
            inputs = inputs.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_start = time.perf_counter()
            outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append((time.perf_counter() - infer_start) * 1000)
            probs = torch.softmax(outputs, dim=1)[0].cpu().tolist()
            expected.append(int(labels.item()))
            predicted.append(int(outputs.argmax(1).item()))
            probabilities.append(probs)
            names.append(filenames[0])

    metrics = classification_metrics(expected, predicted)
    metrics.update(
        {
            "abordagem": "CNN treinada do zero",
            "epocas": epochs,
            "tempo_treinamento_s": round(training_seconds, 3),
            "mean_inference_ms": round(float(np.mean(inference_times)), 3),
            "best_validation_accuracy": round(best_val_accuracy, 6),
            "modelo": model_path.relative_to(models_root.parent).as_posix(),
            "predicoes_teste": [
                {
                    "arquivo": name,
                    "esperado": CLASS_NAMES[target],
                    "predito": CLASS_NAMES[guess],
                    "probabilidades": {
                        CLASS_NAMES[index]: round(probability, 6)
                        for index, probability in enumerate(probs)
                    },
                }
                for name, target, guess, probs in zip(
                    names, expected, predicted, probabilities, strict=True
                )
            ],
        }
    )
    write_json(summary_path, metrics)

    epochs_x = [row.epoch for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_x, [row.train_loss for row in history], label="treino")
    axes[0].plot(epochs_x, [row.val_loss for row in history], label="validacao")
    axes[0].set_title("CNN - perda")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()
    axes[1].plot(epochs_x, [row.train_accuracy for row in history], label="treino")
    axes[1].plot(epochs_x, [row.val_accuracy for row in history], label="validacao")
    axes[1].set_title("CNN - acuracia")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(results_root / "cnn_curvas_treinamento.png", dpi=160)
    plt.close(fig)

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, int(matrix[row, column]), ha="center", va="center")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("CNN - matriz de confusao")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(results_root / "cnn_matriz_confusao.png", dpi=160)
    plt.close(fig)

    # Evidencia visual das classificacoes do conjunto de teste.
    canvas = Image.new("RGB", (4 * 320, 2 * 240), "#111827")
    for index, prediction in enumerate(metrics["predicoes_teste"]):
        path = dataset_root / "images" / "test" / prediction["arquivo"]
        with Image.open(path) as source:
            source = source.convert("RGB")
            source.thumbnail((310, 195))
            tile = Image.new("RGB", (320, 240), "#111827")
            tile.paste(source, ((320 - source.width) // 2, 35))
            draw = ImageDraw.Draw(tile)
            color = "#22c55e" if prediction["esperado"] == prediction["predito"] else "#ef4444"
            draw.text(
                (8, 8),
                f"real: {prediction['esperado']} | predito: {prediction['predito']}",
                fill=color,
            )
            canvas.paste(tile, ((index % 4) * 320, (index // 4) * 240))
    canvas.save(results_root / "cnn_predicoes_teste.jpg", quality=92)
    return metrics


def create_comparison(results_root: Path, summaries: list[dict]) -> None:
    rows = []
    for summary in summaries:
        rows.append(summary)
    write_json(results_root / "comparacao_final.json", {"resultados": rows})

    headers = [
        "abordagem",
        "acuracia_classificacao",
        "precision",
        "recall",
        "map50",
        "tempo_treinamento_s",
        "inferencia_ms",
    ]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.get("abordagem", "")),
                    str(row.get("classification_accuracy", row.get("accuracy", ""))),
                    str(row.get("precision_iou50", row.get("precision", row.get("macro_precision", "")))),
                    str(row.get("recall_iou50", row.get("recall", row.get("macro_recall", "")))),
                    str(row.get("map50", "")),
                    str(row.get("tempo_treinamento_s", 0)),
                    str(row.get("mean_inference_ms", "")),
                ]
            )
        )
    (results_root / "comparacao_final.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Refaz todos os treinamentos")
    args = parser.parse_args()
    seed_everything()

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "openimages_dog_car"
    results_root = project_root / "resultados"
    models_root = project_root / "modelos"
    results_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    data_yaml = runtime_yaml(dataset_root, results_root)

    print("Dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    yolo_30 = train_yolo(30, data_yaml, results_root, models_root, args.force)
    yolo_60 = train_yolo(60, data_yaml, results_root, models_root, args.force)

    custom_30_predictions = evaluate_yolo_predictions(
        models_root / "yolo_custom_30_best.pt",
        dataset_root,
        results_root / "predicoes_yolo_custom_30",
        standard_coco=False,
    )
    custom_30_predictions.update(yolo_30)
    custom_30_predictions["abordagem"] = "YOLO customizada - 30 epocas"
    write_json(results_root / "avaliacao_yolo_custom_30.json", custom_30_predictions)

    custom_60_predictions = evaluate_yolo_predictions(
        models_root / "yolo_custom_60_best.pt",
        dataset_root,
        results_root / "predicoes_yolo_custom_60",
        standard_coco=False,
    )
    custom_60_predictions.update(yolo_60)
    custom_60_predictions["abordagem"] = "YOLO customizada - 60 epocas"
    write_json(results_root / "avaliacao_yolo_custom_60.json", custom_60_predictions)

    standard_predictions = evaluate_yolo_predictions(
        "yolov8n.pt",
        dataset_root,
        results_root / "predicoes_yolo_padrao",
        standard_coco=True,
    )
    standard_predictions["abordagem"] = "YOLO padrao (COCO, sem novo treino)"
    standard_predictions["tempo_treinamento_s"] = 0
    write_json(results_root / "avaliacao_yolo_padrao.json", standard_predictions)

    cnn = run_cnn(dataset_root, results_root, models_root, args.force)
    create_comparison(
        results_root,
        [custom_30_predictions, custom_60_predictions, standard_predictions, cnn],
    )
    print("Experimentos concluidos. Resultados em", results_root)


if __name__ == "__main__":
    main()
