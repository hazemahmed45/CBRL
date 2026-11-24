"""Download and prepare the classification datasets exactly as in the thesis.

    python prepare_data.py deepweeds                 # downloads, splits, writes label splits, verifies
    python prepare_data.py plantvillage              # needs the Kaggle CLI (or --source-dir)
    python prepare_data.py plantdoc
    python prepare_data.py new_plant_diseases        # needs the Kaggle CLI (or --source-dir)
    python prepare_data.py lucas_cover_photos        # ~196k photos from the JRC FTP (resumable)

Each dataset ends up in data/<Name>/{train,val,test}/<class>/<image> (symlinks into
data/_downloads by default). Datasets that ship without a test split are split 70/15/15 per class
with seed 0, as in the thesis. The script then writes the stratified 10% label splits for seeds
0, 1 and 2 to data/splits/<dataset>/ and checks the file lists and split indices against the
checksums of the thesis data. A mismatch means the downloaded source differs from the one used
in the thesis (for example a changed Kaggle mirror), and the results will not be comparable.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import http.cookiejar
import io
import json
import random
import re
import shutil
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEED, TRAIN_RATIO, VAL_RATIO = 0, 0.7, 0.15
FOLDERS = {"deepweeds": "DeepWeeds", "plantvillage": "PlantVillage", "plantdoc": "PlantDoc",
           "new_plant_diseases": "NewPlantDiseases", "lucas_cover_photos": "LUCASCoverPhotos"}
DEEPWEEDS_LABELS_URL = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/labels.csv"
DEEPWEEDS_IMAGES_GOOGLE_DRIVE_ID = "1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj"
PLANTDOC_URL = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
LUCAS_IMAGES_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/LUCAS/LUCAS_COVER/LUCAS2018/"
LUCAS_METADATA_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/LUCAS/LUCAS_COVER/tables/lucas_cover_attr.csv.zip"
NEW_PLANT_DISEASES_TEST_CLASSES = {
    "AppleCedarRust": "Apple___Cedar_apple_rust", "AppleScab": "Apple___Apple_scab",
    "CornCommonRust": "Corn_(maize)___Common_rust_", "PotatoEarlyBlight": "Potato___Early_blight",
    "PotatoHealthy": "Potato___healthy", "TomatoEarlyBlight": "Tomato___Early_blight",
    "TomatoHealthy": "Tomato___healthy", "TomatoYellowCurlVirus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}

# sha256 of the sorted "split/class/file" list and the image count of every split of the thesis data
EXPECTED_LAYOUT = {
    "deepweeds": {"train": ("93a0dbf4efe1744cb871df1004b9c641bd7a30f70e49397975b0e5a808b83eda", 12256),
                  "val": ("42672af6b13c65716b8487ad8b625cffbaf5a0b4a4529f18951a35d680281261", 2626),
                  "test": ("48248a4aaeb45d3c73d6eda64eec4c4c426325ed6661d561cce47f5056f895be", 2627)},
    "plantvillage": {"train": ("7c8cc4a54975c69b857876fd43aedd7f2467ee0d10fdf336ff70e634a7dc904b", 38012),
                     "val": ("ba0994be881f1b35efe03b5cf401d14f9b79956177e12cc96181e2de7f99e498", 8146),
                     "test": ("36198f011fe00d99ae41cc996ebbecededa5aad2ac43f53fda551d6223dce10d", 8147)},
    "plantdoc": {"train": ("458c1fb23c4183983f0146d4ec8441e365f02116ad4ebe86f5afb1675031af1e", 1990),
                 "val": ("797da7967e8609cf8e390a152e32de2b52e780dc9bc1dc5ad27c9562a7d4c6cb", 352),
                 "test": ("9e7357117c146c227e5dc20acd4f232662d67f354f575cb9268553af5995b49c", 236)},
    "new_plant_diseases": {"train": ("261c5f12d6516c630a4663e32db92c0748a246ebbcd21eaae22e6b55851da100", 70295),
                           "val": ("0cfe7ababe3e550b7d2a9fefeac2db14072c6c0060681f3505a96d8ef94cfa85", 17572),
                           "test": ("5d383cf78ac8cafbb8231db3288bdb76aa7aa997248356edc52278fa7965d1a2", 33)},
    "lucas_cover_photos": {"train": ("79f851d3f6c6e9cc8247ae90a3be0f2faeb51081b260710ab86959857237b97f", 136647),
                           "val": ("cce78e6677858bf1099039012a33f73b8d91434ec3364b6a94a1844ba8c1adac", 29281),
                           "test": ("8d4dc4e2df5870b5d7c288a7464e57fc5790e2a68bc49c2d4aec6652e035b991", 29285)},
}
# sha256 of json.dumps(indices) of the 10% label splits used in the thesis, per seed
EXPECTED_SPLITS = {
    "deepweeds": ["69bcb1e8d933f4c9d37101468892782aaf79f5aea7d05686d75fdde1031c54b8",
                  "096fde5ba38028b0bc41ca949fa98c8e448754538569e0868217e0584022a5dc",
                  "c9d8052c5cade75b8723808ef88234f6a2aea0056273824dc04c46033b01fcdc"],
    "plantvillage": ["7225d4a5f004142c52b6241e3f69322a0f38915e4086c24ce7cf0a9b713375d3",
                     "e962fe2b9b2bb07cf49dd39c35f5742ed50136b89130f04e3780f1be8c921dd8",
                     "19e2b75d4c636aa58f911d224c8abc5c89a943029d822e2b6cded2c1933d908f"],
    "plantdoc": ["65669bb70e69904f27d3558d15cf842e1c693340ff3b874d37777cf69bf63dcf",
                 "91c4fe8caaf74a23ff32f9106316a53c86bed581bf20682f89c742ce16c4a862",
                 "aacfe4373c01e9816f107dccd19aa759af6ab16e74fd4ca4129b1d80dd789521"],
    "new_plant_diseases": ["cd2f694106de30b3343199d3f5958177941d30954728a52b9b0729bdb63491a4",
                           "8f41e46f73bfa43b239db1b577c557b9cfe5b0abac5fb22fddea804ca93e7da9",
                           "ade9cb4ea1198f2b08ebf4088100662ad021ade7fb8aea2d98a6826a78d6a377"],
    "lucas_cover_photos": ["022a975b6c9cbfd048cd9d5b9d3c7b789c69b764e85a4f5b2423edee8db19533",
                           "06944331c08cc5abffb5811e93baa1c711bd45ab8ed7630b3ef6b325e6b4f413",
                           "fa12538828575f08800196f6b01dd0d176bba82b36e1cd93b1114605057bd0a1"],
}


# ----------------------------------------------------------------------------- per-dataset preparation
def prepare_deepweeds(args, destination: Path, work_dir: Path) -> None:
    if args.source_dir:
        source = Path(args.source_dir)
    else:
        source = work_dir / "deepweeds"
        download_url(DEEPWEEDS_LABELS_URL, source / "labels.csv")
        if not list_images(source / "images"):
            archive = download_google_drive_file(DEEPWEEDS_IMAGES_GOOGLE_DRIVE_ID, source / "images.zip")
            (source / "images").mkdir(parents=True, exist_ok=True)
            print(f"extracting {archive}")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(source / "images")
    labels_csv = next(iter(source.rglob("labels.csv")))
    image_root = max([source] + [p for p in source.rglob("*") if p.is_dir()], key=lambda p: len(list_images(p)))
    split_csv_classification(labels_csv, image_root, destination, args)


def prepare_plantvillage(args, destination: Path, work_dir: Path) -> None:
    source = Path(args.source_dir) if args.source_dir else download_kaggle(
        "abdallahalidev/plantvillage-dataset", work_dir / "abdallahalidev_plantvillage-dataset")
    split_classification_folder(find_classification_root(source, ["color", "PlantVillage", "plantvillage"]),
                                destination, args)


def prepare_plantdoc(args, destination: Path, work_dir: Path) -> None:
    if args.source_dir:
        source = Path(args.source_dir)
    else:
        archive = download_url(PLANTDOC_URL, work_dir / "master.zip")
        source = extract_archive(archive, work_dir / "master")
    split_root = find_classification_split_root(source)
    if split_root is not None:
        normalize_existing_classification_split(split_root, destination, args)
    else:
        split_classification_folder(find_classification_root(source, ["train", "Train", "PlantDoc-Classification"]),
                                    destination, args)


def prepare_new_plant_diseases(args, destination: Path, work_dir: Path) -> None:
    source = Path(args.source_dir) if args.source_dir else download_kaggle(
        "vipoooool/new-plant-diseases-dataset", work_dir / "vipoooool_new-plant-diseases-dataset")
    split_root = find_classification_split_root(source)
    if split_root is None:
        raise FileNotFoundError(f"No train/valid class folders found under {source}")
    normalize_existing_classification_split(split_root, destination, args,
                                            additional_test_images=find_new_plant_diseases_test_images(source, split_root))


def prepare_lucas(args, destination: Path, work_dir: Path) -> None:
    """LUCAS 2018 land-cover photos labelled with the level-1 land-cover code (lc1)."""
    with gzip.open(Path(__file__).with_name("manifests") / "lucas2018_images.txt.gz", "rt") as handle:
        wanted = [line.strip() for line in handle if line.strip()]  # the 195,814 photos used in the thesis
    source = Path(args.source_dir) if args.source_dir else work_dir / "lucas" / "LUCAS2018"
    metadata = Path(args.metadata) if args.metadata else work_dir / "lucas" / "tables" / "lucas_cover_attr.csv.zip"
    if not args.source_dir:
        download_lucas_images(wanted, source, args.workers)
    if not args.metadata:
        download_url(LUCAS_METADATA_URL, metadata)
    missing = [path for path in wanted if not (source / path).is_file()]
    if missing:
        print(f"WARNING: {len(missing)} of {len(wanted)} thesis photos are missing; the split will differ.")
    images = sorted(source / path for path in wanted if (source / path).is_file())
    split_lucas2018(images, metadata, destination, args)


def download_lucas_images(relative_paths: list[str], destination: Path, workers: int) -> None:
    todo = [path for path in relative_paths if not (destination / path).is_file()]
    print(f"LUCAS: {len(relative_paths) - len(todo)} photos present, {len(todo)} to download")

    def fetch(path: str) -> None:
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(LUCAS_IMAGES_URL + path, timeout=60) as response:
                    data = response.read()
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(target)
                return
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, path) for path in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except OSError as error:
                print(f"  failed: {error}")
            if done % 5000 == 0:
                print(f"  downloaded {done}/{len(todo)}")


PREPARE = {"deepweeds": prepare_deepweeds, "plantvillage": prepare_plantvillage, "plantdoc": prepare_plantdoc,
           "new_plant_diseases": prepare_new_plant_diseases, "lucas_cover_photos": prepare_lucas}


# ----------------------------------------------------------------------------- splitting (seed 0, 70/15/15)
def split_sequence(items: list, train_ratio: float, val_ratio: float) -> dict[str, list]:
    train_end = round(len(items) * train_ratio)
    val_end = train_end + round(len(items) * val_ratio)
    return {"train": items[:train_end], "val": items[train_end:val_end], "test": items[val_end:]}


def split_sequence_nonempty(items: list, train_ratio: float, val_ratio: float) -> dict[str, list]:
    train_count = max(1, round(len(items) * train_ratio))
    val_count = max(1, round(len(items) * val_ratio))
    if train_count + val_count >= len(items):
        train_count = max(1, len(items) - val_count - 1)
    if train_count + val_count >= len(items):
        val_count = max(1, len(items) - train_count - 1)
    return {"train": items[:train_count], "val": items[train_count:train_count + val_count],
            "test": items[train_count + val_count:]}


def place(groups: dict[str, list[Path]], destination: Path, args, nonempty: bool = False) -> None:
    """Shuffle each class with one shared seeded generator and link its images into train/val/test."""
    rng = random.Random(SEED)
    for label, images in sorted(groups.items()):
        rng.shuffle(images)
        splitter = split_sequence_nonempty if nonempty else split_sequence
        for split, paths in splitter(images, TRAIN_RATIO, VAL_RATIO).items():
            (destination / split / label).mkdir(parents=True, exist_ok=True)
            for image in paths:
                link_or_copy(image, destination / split / label / image.name, args.copy_mode)


def split_classification_folder(source: Path, destination: Path, args) -> None:
    classes = sorted(path for path in source.iterdir() if path.is_dir() and list_images(path))
    reset_destination(destination, args.force)
    place({class_dir.name: list_images(class_dir) for class_dir in classes}, destination, args)


def split_csv_classification(labels_csv: Path, image_root: Path, destination: Path, args) -> None:
    images_by_name = {image.name: image for image in list_images(image_root)}
    groups: dict[str, list[Path]] = defaultdict(list)
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = row.get("Filename") or row.get("filename") or row.get("image") or row.get("image_path")
            species = row.get("Species") or row.get("species") or row.get("label") or row.get("Label")
            image = images_by_name.get(Path(filename).name) if filename and species else None
            if image is not None:
                groups[sanitize_class_name(species)].append(image)
    reset_destination(destination, args.force)
    place(groups, destination, args)


def split_lucas2018(images: list[Path], metadata: Path, destination: Path, args) -> None:
    images_by_name = {image.name: image for image in images}
    groups: dict[str, list[Path]] = defaultdict(list)
    with open_lucas_metadata(metadata) as handle:
        for row in csv.DictReader(handle):
            if row["year"] != "2018":
                continue
            image = images_by_name.get(Path(row["file_path_ftp_cover"]).name)
            code, label = row["lc1"].strip(), row["lc1_label"].strip()
            if image is None or not code or code in {"8", "NA"} or not label or label.lower() in {"na", "not relevant"}:
                continue
            groups[sanitize_class_name(f"{code}_{label}")].append(image)
    reset_destination(destination, args.force)
    place({label: paths for label, paths in groups.items() if len(paths) >= 3}, destination, args, nonempty=True)


def open_lucas_metadata(path: Path):
    if path.suffix.lower() != ".zip":
        return path.open("r", encoding="utf-8-sig", newline="")
    archive = zipfile.ZipFile(path)
    name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
    text = io.TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline="")

    class Reader:
        def __enter__(self):
            return text

        def __exit__(self, *_):
            text.close()
            archive.close()

    return Reader()


def normalize_existing_classification_split(source: Path, destination: Path, args, additional_test_images=None) -> None:
    """Keep a dataset's own train/val/test folders; carve a seeded validation split if it has none."""
    reset_destination(destination, args.force)
    aliases = {"train": ["train", "Train", "training"], "val": ["val", "Val", "valid", "validation"], "test": ["test", "Test"]}
    splits = {split: first_existing(source / alias for alias in names) for split, names in aliases.items()}
    if splits["train"] is not None and splits["val"] is None:
        rng = random.Random(SEED)
        for class_dir in sorted(path for path in splits["train"].iterdir() if path.is_dir()):
            images = list_images(class_dir)
            if not images:
                continue
            rng.shuffle(images)
            val_count = max(1, round(len(images) * VAL_RATIO)) if len(images) > 1 else 0
            for split, paths in {"train": images[:-val_count] if val_count else images,
                                 "val": images[-val_count:] if val_count else []}.items():
                (destination / split / class_dir.name).mkdir(parents=True, exist_ok=True)
                for image in paths:
                    link_or_copy(image, destination / split / class_dir.name / image.name, args.copy_mode)
        splits["train"] = None
    for split, split_dir in splits.items():
        if split_dir is None:
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            (destination / split / class_dir.name).mkdir(parents=True, exist_ok=True)
            for image in list_images(class_dir):
                link_or_copy(image, destination / split / class_dir.name / image.name, args.copy_mode)
    for image, class_name in additional_test_images or []:
        link_or_copy(image, destination / "test" / class_name / image.name, args.copy_mode)


def find_new_plant_diseases_test_images(source: Path, split_root: Path) -> list[tuple[Path, str]]:
    """Label the Kaggle dataset's flat test/ folder from the filename prefixes."""
    candidates = []
    for candidate in [source] + [path for path in source.rglob("*") if path.is_dir()]:
        if candidate.name.lower() != "test":
            continue
        images = sorted(p for p in candidate.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if images and not candidate.is_relative_to(split_root):
            candidates.append((len(images), candidate, images))
    if not candidates:
        return []
    _, _, images = max(candidates, key=lambda item: (item[0], -len(str(item[1]))))
    labelled = []
    for image in images:
        class_name = NEW_PLANT_DISEASES_TEST_CLASSES.get(re.sub(r"\d+$", "", image.stem))
        if class_name is None:
            raise ValueError(f"Cannot infer the class of test image {image.name}")
        labelled.append((image, class_name))
    return labelled


# ----------------------------------------------------------------------------- label-fraction splits
def write_label_splits(dataset: str, data_root: Path, seeds=(0, 1, 2), fraction: float = 0.1) -> list[Path]:
    """Stratified label subsets of the training split; indices refer to torchvision ImageFolder order."""
    from torchvision.datasets import ImageFolder

    train = ImageFolder(str(data_root / FOLDERS[dataset] / "train"))
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(train.targets):
        by_class[int(target)].append(index)
    written = []
    for seed in seeds:
        rng = random.Random(seed)
        selected = []
        for indices in by_class.values():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            selected.extend(shuffled[:max(1, round(len(shuffled) * fraction))])
        selected.sort()
        out = data_root / "splits" / dataset / f"seed_{seed}_{round(fraction * 100)}percent.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"dataset": dataset, "task": "classification", "seed": seed, "fraction": fraction,
                                   "indices": selected, "class_to_idx": train.class_to_idx},
                                  indent=2, sort_keys=True) + "\n")
        written.append(out)
    return written


def verify(dataset: str, data_root: Path) -> bool:
    ok = True
    root = data_root / FOLDERS[dataset]
    for split, (expected_hash, expected_count) in EXPECTED_LAYOUT[dataset].items():
        files = sorted(str(p.relative_to(root)) for p in (root / split).rglob("*") if p.is_file())
        digest = hashlib.sha256(("\n".join(files) + "\n").encode()).hexdigest()
        match = digest == expected_hash
        ok &= match
        print(f"  {split:5s} {len(files):7,d} images (thesis {expected_count:7,d})  {'OK' if match else 'DIFFERS'}")
    for seed, expected in enumerate(EXPECTED_SPLITS[dataset]):
        path = data_root / "splits" / dataset / f"seed_{seed}_10percent.json"
        digest = hashlib.sha256(json.dumps(json.loads(path.read_text())["indices"]).encode()).hexdigest()
        ok &= digest == expected
        print(f"  10% label split seed {seed}: {'OK' if digest == expected else 'DIFFERS'}")
    return ok


# ----------------------------------------------------------------------------- download helpers
def download_url(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return destination


def download_google_drive_file(file_id: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = "https://docs.google.com/uc?export=download"
    cookies = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
    print(f"downloading Google Drive file {file_id}")
    response = opener.open(f"{url}&id={file_id}")
    token = next((c.value for c in cookies if c.name.startswith("download_warning")), None)
    if token:
        response = opener.open(f"{url}&confirm={token}&id={file_id}")
    elif "text/html" in response.headers.get("Content-Type", ""):
        html = response.read().decode("utf-8", errors="replace")
        action = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', html)
        inputs = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', html))
        if not action or not inputs.get("confirm"):
            raise ValueError("Could not parse the Google Drive download form; download images.zip manually.")
        response = opener.open(f"{urllib.parse.urljoin(url, action.group(1))}?{urllib.parse.urlencode(inputs)}")
    with response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if not zipfile.is_zipfile(destination):
        raise ValueError(f"Google Drive did not return a ZIP archive: {destination}")
    return destination


def download_kaggle(slug: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    if not any(destination.iterdir()):
        if shutil.which("kaggle") is None:
            raise SystemExit("Install and configure the Kaggle CLI, or pass --source-dir.")
        subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(destination), "--unzip"], check=True)
    return destination


def extract_archive(archive: Path, destination: Path) -> Path:
    if not (destination.exists() and any(destination.iterdir())):
        destination.mkdir(parents=True, exist_ok=True)
        print(f"extracting {archive}")
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(destination)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(destination)
    children = [child for child in destination.iterdir() if child.is_dir()]
    if len(children) == 1 and not any(child.is_file() for child in destination.iterdir()):
        return children[0]
    return destination


# ----------------------------------------------------------------------------- small helpers
def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def find_classification_root(source: Path, preferred_names: list[str]) -> Path:
    candidates = [source]
    candidates += [path for name in preferred_names for path in source.rglob(name) if path.is_dir()]
    candidates += [path for path in source.rglob("*") if path.is_dir()]
    scored = []
    for candidate in candidates:
        count = sum(1 for child in candidate.iterdir() if child.is_dir() and list_images(child))
        if count:
            scored.append((count, len(str(candidate)), candidate))
    if not scored:
        raise FileNotFoundError(f"No class folders found under {source}")
    return sorted(scored, key=lambda item: (-item[0], item[1]))[0][2]


def find_classification_split_root(source: Path) -> Path | None:
    scored = []
    for candidate in [source] + [path for path in source.rglob("*") if path.is_dir()]:
        train = first_existing(candidate / name for name in ("train", "Train", "training"))
        held_out = first_existing(candidate / name for name in ("test", "Test", "val", "Val", "valid", "validation"))
        if train is None or held_out is None:
            continue
        train_classes = sum(1 for child in train.iterdir() if child.is_dir() and list_images(child))
        held_out_classes = sum(1 for child in held_out.iterdir() if child.is_dir() and list_images(child))
        if train_classes and held_out_classes:
            scored.append((train_classes + held_out_classes, len(str(candidate)), candidate))
    return sorted(scored, key=lambda item: (-item[0], item[1]))[0][2] if scored else None


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        try:
            destination.hardlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
    else:
        shutil.copy2(source, destination)


def reset_destination(destination: Path, force: bool) -> None:
    if destination.exists() and any(destination.iterdir()):
        if not force:
            raise SystemExit(f"{destination} already exists; pass --force to rebuild it.")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)


def first_existing(paths) -> Path | None:
    return next((Path(path) for path in paths if Path(path).exists()), None)


def sanitize_class_name(name: str) -> str:
    return name.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(FOLDERS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--source-dir", help="an already downloaded copy of the raw dataset")
    parser.add_argument("--metadata", help="LUCAS only: lucas_cover_attr.csv(.zip) if already downloaded")
    parser.add_argument("--copy-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--workers", type=int, default=16, help="LUCAS only: parallel downloads")
    parser.add_argument("--force", action="store_true", help="rebuild an existing prepared dataset")
    parser.add_argument("--verify-only", action="store_true", help="only rewrite the label splits and check")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    if not args.verify_only:
        PREPARE[args.dataset](args, data_root / FOLDERS[args.dataset], data_root / "_downloads")
    for path in write_label_splits(args.dataset, data_root):
        print(f"wrote {path}")
    print(f"Checking {args.dataset} against the thesis data:")
    if verify(args.dataset, data_root):
        print("All splits match the thesis data.")
    else:
        print("WARNING: the prepared data differ from the thesis data; results will not be directly comparable.")


if __name__ == "__main__":
    main()
