"""Command-line entry point for SD-MKD preprocessing."""
from __future__ import annotations

import argparse
from pathlib import Path

from .preprocessor import DataPreprocessor, PreprocessConfig
from .utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess encrypted traffic for SD-MKD")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., ISCXVPN2016)")
    parser.add_argument("--data_path", required=True, help="Path to raw dataset root")
    parser.add_argument("--packets_per_flow", type=int, default=20)
    parser.add_argument("--bytes_per_packet", type=int, default=150)
    parser.add_argument("--max_length", type=int, default=1500)
    parser.add_argument("--image_height", type=int, default=30)
    parser.add_argument("--image_width", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger()
    config = PreprocessConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        packets_per_flow=args.packets_per_flow,
        bytes_per_packet=args.bytes_per_packet,
        max_length=args.max_length,
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )

    preprocessor = DataPreprocessor(config, logger=logger)
    dataset, encoder = preprocessor.process_dataset()
    logger.info(f"Processed {len(dataset)} flows with {encoder.get_num_classes()} classes")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
