import argparse
import re
from pathlib import Path

def normalize(s: str) -> str:
    s = s.replace("_", " ")
    s = re.sub(r"[()]+", " ", s)
    s = re.sub(r"[-/]", " ", s)
    s = re.sub(r"[^a-z0-9, ]+", "", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_synonyms(raw: str):
    # split on commas/semicolons and " or "
    parts = re.split(r",|;| or ", raw)
    return [normalize(p) for p in parts if normalize(p)]

def load_wnids(path: Path):
    wnids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(wnids) == 200, f"Expected 200 wnids in {path}, got {len(wnids)}"
    return wnids

def load_words(path: Path):
    # words.txt lines look like: n02124075  Egyptian cat
    m = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        wid, name = ln.split("\t") if "\t" in ln else ln.split(" ", 1)
        m[wid] = split_synonyms(name)
    return m

def build_teacher_names(model):
    names = model.names  # list[str] or dict[int->str]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    # For each teacher index, prepare multiple normalized variants
    idx_to_norm_syns = {}
    for i, raw in enumerate(names):
        # Many imagenet labels look like "Egyptian cat, Ctesias"
        syns = split_synonyms(str(raw))
        if not syns:
            syns = [normalize(str(raw))]
        idx_to_norm_syns[i] = syns
    return idx_to_norm_syns

def best_match(target_syns, idx_to_syns):
    # Try exact, then substring, then token-overlap (Jaccard)
    target_set_list = [set(s.split()) for s in target_syns]

    # exact
    for idx, syns in idx_to_syns.items():
        if any(t in syns for t in target_syns):
            return idx

    # substring (either direction)
    for idx, syns in idx_to_syns.items():
        for t in target_syns:
            if any(t in s or s in t for s in syns):
                return idx

    # token Jaccard
    best_idx, best_score = None, 0.0
    for idx, syns in idx_to_syns.items():
        for s in syns:
            s_set = set(s.split())
            for t_set in target_set_list:
                if not t_set or not s_set:
                    continue
                inter = len(s_set & t_set)
                union = len(s_set | t_set)
                j = inter / union if union else 0.0
                if j > best_score:
                    best_score, best_idx = j, idx
    return best_idx if best_score >= 0.5 else None

def main():
    ap = argparse.ArgumentParser(description="Build Tiny-ImageNet → ImageNet-1k mapping using words.txt + teacher names")
    ap.add_argument("--wnids", type=Path,  default="./data/tiny-imagenet-200/wnids.txt", help="tiny-imagenet-200/wnids.txt")
    ap.add_argument("--words", type=Path,  default="./data/tiny-imagenet-200/words.txt", help="tiny-imagenet-200/words.txt")
    ap.add_argument("--teacher", type=str, default="yolov8n-cls.pt", help="Ultralytics YOLOv8 cls weights or local .pt")
    ap.add_argument("--out", type=Path, default=Path("timnet_to_imagenet1k_indices.txt"))
    args = ap.parse_args()

    wnids = load_wnids(args.wnids)
    words = load_words(args.words)

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit("Please `pip install ultralytics` first.") from e

    model = YOLO(args.teacher)
    idx_to_syns = build_teacher_names(model)

    # Fast path: if teacher names are already WNIDs (rare), map directly
    if all(re.fullmatch(r"n\d{8}", s) for syns in idx_to_syns.values() for s in syns):
        wnid_to_idx = {syns[0]: i for i, syns in idx_to_syns.items()}
        indices = []
        missing = []
        for w in wnids:
            if w in wnid_to_idx:
                indices.append(wnid_to_idx[w])
            else:
                missing.append(w)
        if missing:
            raise SystemExit(f"Teacher names look like WNIDs but some were missing: {missing}")
        args.out.write_text("\n".join(map(str, indices)), encoding="utf-8")
        print(f"[OK] wrote {args.out} (fast WNID path)")
        return

    # General path: match via words.txt synonyms ↔ teacher names
    indices = []
    missing = []
    for w in wnids:
        syns = words.get(w)
        if not syns:
            missing.append(w)
            continue
        idx = best_match(syns, idx_to_syns)
        if idx is None:
            missing.append(w)
        else:
            indices.append(idx)

    if missing:
        print("[WARN] Could not auto-match the following WNIDs:")
        for x in missing:
            print("  ", x, "->", ", ".join(words.get(x, ["<no words>"])))
        raise SystemExit(
            f"Auto-matching failed for {len(missing)} class(es). "
            "You can:\n"
            "  (1) Use a larger teacher (e.g., yolov8m-cls.pt) if its names contain richer synonyms, or\n"
            "  (2) Manually add a small override map in this script for the listed WNIDs."
        )

    if len(indices) != 200:
        raise SystemExit(f"Mapped {len(indices)}/200 classes; something went wrong.")

    args.out.write_text("\n".join(str(i) for i in indices), encoding="utf-8")
    print(f"[OK] wrote {args.out} ({len(indices)} indices)")

if __name__ == "__main__":
    main()
