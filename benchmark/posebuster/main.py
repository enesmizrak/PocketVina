"""Historical posebuster command spelling; the shared protocol runner owns all execution."""


def main():
    # Direct script invocation must select this checkout, even with a shared editable installation.
    import sys
    from pathlib import Path

    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmark.run import legacy_main

    return legacy_main("posebuster")


if __name__ == "__main__":
    raise SystemExit(main())
