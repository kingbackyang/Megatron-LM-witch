import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch preprocess jsonl files with Megatron-LM utilities.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing jsonl files to preprocess.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="./qwen0_6B_customv2/",
        help="Path to the tokenizer model passed to preprocess_data.py.",
    )
    parser.add_argument(
        "--tokenizer-type",
        default="HuggingFaceTokenizer",
        help="Tokenizer type used by preprocess_data.py.",
    )
    parser.add_argument(
        "--preprocess-script",
        default="tools/preprocess_data.py",
        help="Path to Megatron-LM's preprocess_data.py script.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of files to preprocess concurrently.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of workers passed to preprocess_data.py.",
    )
    parser.add_argument(
        "--no-append-eod",
        action="store_false",
        dest="append_eod",
        help="Disable --append-eod when calling preprocess_data.py.",
    )
    parser.set_defaults(append_eod=True)
    return parser.parse_args()


def collect_jsonl_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for current_root, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".jsonl"):
                files.append(Path(current_root) / filename)
    return sorted(files)


def output_prefix_for_file(input_root: Path, file_path: Path) -> Path:
    parts = list(input_root.parts)
    if "pretrain_data" in parts:
        parts[parts.index("pretrain_data")] = "pretrain_data_megatron_bin"
        output_root = Path(*parts)
    else:
        output_root = input_root.parent / f"{input_root.name}_megatron_bin"
    relative = file_path.relative_to(input_root)
    return (output_root / relative).with_suffix("")


def run_preprocess(
    script: str,
    input_file: Path,
    output_prefix: Path,
    tokenizer_type: str,
    tokenizer_model: str,
    workers: int,
    append_eod: bool,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        script,
        "--input",
        str(input_file),
        "--output-prefix",
        str(output_prefix),
        "--tokenizer-type",
        tokenizer_type,
        "--tokenizer-model",
        tokenizer_model,
        "--workers",
        str(workers),
    ]
    if append_eod:
        cmd.append("--append-eod")
    subprocess.run(cmd, check=True)


def process_file(file_path: Path, args: argparse.Namespace, input_root: Path) -> None:
    output_prefix = output_prefix_for_file(input_root, file_path)
    run_preprocess(
        args.preprocess_script,
        file_path,
        output_prefix,
        args.tokenizer_type,
        args.tokenizer_model,
        args.workers,
        args.append_eod,
    )


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_dir).expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_root}")

    jsonl_files = collect_jsonl_files(input_root)
    if not jsonl_files:
        print(f"No jsonl files found under {input_root}")
        return

    max_threads = max(1, args.threads)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(process_file, file_path, args, input_root): file_path
            for file_path in jsonl_files
        }
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                future.result()
                print(f"Finished preprocessing: {file_path}")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Failed preprocessing {file_path}: {exc}")


if __name__ == "__main__":
    main()
