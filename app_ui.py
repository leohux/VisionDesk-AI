"""Launch VisionDesk Gradio console.

Wrapped so a crash prints a full traceback instead of the window vanishing.
"""
from __future__ import annotations

import sys
import traceback


def _run() -> int:
    from ui.gradio_app import main

    main()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_run())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
    except BaseException:
        traceback.print_exc()
        print("\nVisionDesk UI crashed — see traceback above.", flush=True)
        if sys.stdin is not None and sys.stdin.isatty():
            input("Press Enter to close...")
        raise SystemExit(1)
