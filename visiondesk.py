"""VisionDesk AI CLI entrypoint.

Examples:
  python visiondesk.py ui
  python visiondesk.py replay video.mp4 --profile traffic
  python visiondesk.py replay shot.jpg --profile general --no-deep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="visiondesk", description="VisionDesk AI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ui", help="Launch Gradio cockpit")

    p_rep = sub.add_parser("replay", help="Replay video/image + benchmark")
    p_rep.add_argument("media")
    p_rep.add_argument("--profile", default="traffic")
    p_rep.add_argument("--stride", type=int, default=1)
    p_rep.add_argument("--max-frames", type=int, default=None)
    p_rep.add_argument("--no-deep", action="store_true")
    p_rep.add_argument("--save-dir", type=str, default=None)
    p_rep.add_argument("--json", action="store_true")
    p_rep.add_argument("--no-events", action="store_true")

    args, rest = parser.parse_known_args(argv)

    if args.cmd == "ui":
        from ui.gradio_app import main as ui_main

        ui_main()
        return 0

    if args.cmd == "replay":
        from tools.replay import main as replay_main

        # rebuild argv for replay parser
        rargv = [args.media, "--profile", args.profile, "--stride", str(args.stride)]
        if args.max_frames is not None:
            rargv += ["--max-frames", str(args.max_frames)]
        if args.no_deep:
            rargv.append("--no-deep")
        if args.save_dir:
            rargv += ["--save-dir", args.save_dir]
        if args.json:
            rargv.append("--json")
        if args.no_events:
            rargv.append("--no-events")
        rargv += rest
        return replay_main(rargv)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
