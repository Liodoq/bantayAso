"""Entry point: python -m bantayaso  (pipeline arrives in Batch 1)."""
from . import config


def main() -> None:
    cfg = config.load()
    config.ensure_dirs()
    print("BantayAso skeleton OK")
    print(f"  device : {config.resolve_device(cfg)}")
    print(f"  camera : USB index {cfg['camera']['index']} ({cfg['camera']['backend']})")
    print("  Live pipeline is added in Batch 1.")


if __name__ == "__main__":
    main()
