# ohlc-excursion-model

## 1. Install Nix

Follow the official installation guide: [https://nix.dev/install-nix.html](https://nix.dev/install-nix.html)

## 2. Enter Dev Shell
```bash
nix develop
```

## 3. Run Tests
In the Nix dev shell:
```bash
python -m pytest
```

Tests expect local parquet data at `data/raw/nq_30m.parquet`. If the file is missing,
the tests that require this data will be skipped.

Outside Nix:
```bash
python -m pip install -e .[test]
python -m pytest
```
