# File: check_mmap.py
import argparse
import os
import sys

from megatron.core.datasets.indexed_dataset import IndexedDataset


def collect_prefixes(paths):
    prefixes = []

    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fname in files:
                    if fname.endswith(".bin"):
                        prefixes.append(os.path.join(root, fname[:-4]))
        else:
            prefix = path
            if prefix.endswith(".bin") or prefix.endswith(".idx"):
                prefix = prefix[:-4]
            prefixes.append(prefix)

    prefixes = sorted(set(prefixes))
    return prefixes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, nargs="+")
    parser.add_argument("--mmap", action="store_true")
    args = parser.parse_args()

    prefixes = collect_prefixes(args.data_path)
    if not prefixes:
        print("No prefixes found", file=sys.stderr)
        return 1

    failed = False
    for prefix in prefixes:
        bin_path = prefix + ".bin"
        idx_path = prefix + ".idx"
        if not (os.path.isfile(bin_path) and os.path.isfile(idx_path)):
            print(f"[skip] missing: {bin_path} / {idx_path}")
            continue
        try:
            ds = IndexedDataset(prefix, multimodal=False, mmap=args.mmap)
            _ = int(ds.sequence_lengths.sum())
            print(f"[ok] {prefix} (mmap={args.mmap})")
            del ds
        except Exception as exc:
            print(f"[fail] {prefix} (mmap={args.mmap}): {exc}", file=sys.stderr)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
