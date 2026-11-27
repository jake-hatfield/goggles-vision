import argparse
import os
from pathlib import Path
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path, default=WEIGHTS_DIR / "rf-detr-basketball.pth")
    parser.add_argument("--pose-checkpoint", type=Path, default=WEIGHTS_DIR / "rtmpose-s_simcc-aic-coco.pth")
    parser.add_argument("--reid-model", type=Path, default=WEIGHTS_DIR / "osnet_x1_0_msmt17.pt")
    parser.add_argument("--jersey-ocr-model", type=Path, default=WEIGHTS_DIR / "basketball_jersey_number.ckpt")
    parser.add_argument("--court-model", type=Path, default=WEIGHTS_DIR / "court_mobilenetv4_896x512.pt")
    parser.add_argument(
        "--court-boundary", type=Path,
        help="Use a JSON polygon of normalized image points to identify on-court players.",
    )
    parser.add_argument(
        "--identity-review", type=Path,
        help="Apply reviewed local track intervals to the final IDs and labels for this exact clip.",
    )
    parser.add_argument("--output", type=Path, default=WORKSPACE_ROOT / 'output.mp4')
    parser.add_argument("--tracks", type=Path, default=WORKSPACE_ROOT / 'tracks.jsonl')
    parser.add_argument(
        "--events", type=Path, default=WORKSPACE_ROOT / "events.json",
        help="Write all completed events to this JSON file (default: %(default)s).",
    )
    parser.add_argument("--player-threshold", type=float, default=0.4)
    parser.add_argument("--ball-threshold", type=float, default=0.5)
    parser.add_argument("--rim-threshold", type=float, default=0.3)
    parser.add_argument("--holder-threshold", type=float, default=0.4)
    parser.add_argument("--scorer-threshold", type=float, default=0.4)
    parser.add_argument("--referee-threshold", type=float, default=0.3)
    parser.add_argument("--ball-in-basket-threshold", type=float, default=0.55)
    parser.add_argument("--short-side", type=int, default=768)
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument("--cmc-interval", type=int, default=2)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--trail-length", type=int, default=16)
    parser.add_argument("--pose-batch-size", type=int, default=32)
    parser.add_argument("--pose-interval", type=int, default=1)
    parser.add_argument("--team-cache-ttl-seconds", type=float, default=10.0)
    parser.add_argument("--court-interval", type=int, default=5)
    parser.add_argument("--holder-hold-frames", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["LOG_LEVEL"] = "ERROR"
    from pipeline.video_analysis import VideoAnalysisRunner

    args = _parse_args() if args is None else args
    args.output = None if str(args.output).lower() == "none" else args.output
    VideoAnalysisRunner(args).run()


if __name__ == "__main__":
    main()
