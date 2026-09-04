#!/usr/bin/env python3
import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    base_script = script_dir / "train_unet.py"
    argv = [str(base_script), "--dataset", "asbuilt"] + sys.argv[1:]
    with base_script.open("rb") as fh:
        code = compile(fh.read(), str(base_script), "exec")
    sys.argv = argv
    globals_dict = {"__name__": "__main__", "__file__": str(base_script)}
    exec(code, globals_dict)


if __name__ == "__main__":
    main()
