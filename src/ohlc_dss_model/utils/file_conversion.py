import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import polars as pl


def csv_to_parquet(
    input: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
    separator: str = "\t",
) -> Path:
    input_path = Path(input)

    if not input_path.exists():
        raise FileNotFoundError(f"File {input} doesn't exist!")

    if output is None:
        output_path = input_path.with_suffix(".parquet")
    else:
        output_path = Path(output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        (
            pl.scan_csv(input_path, separator=separator)
            .with_columns(pl.col("DateTime").str.to_datetime("%Y.%m.%d %H:%M:%S"))
            .sink_parquet(output_path)
        )
    except Exception as e:
        raise RuntimeError(f"Conversion failed: {e}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV to Parquet convertor")
    parser.add_argument("input", type=str, help="Path to CSV file")
    parser.add_argument(
        "-o", "--output", type=str, help="Path to output parquet file", default=None
    )
    parser.add_argument(
        "-s", "--separator", type=str, default="\t", help="Separator character"
    )

    args = parser.parse_args()
    sep = "\t" if args.separator == "\\t" else args.separator

    try:
        csv_to_parquet(input=args.input, output=args.output, separator=sep)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
