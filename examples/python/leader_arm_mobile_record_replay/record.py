#!/usr/bin/env python3
"""Run leader-arm/mobile teleoperation and record a synchronized trajectory."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

import teleop
from camera_io import camera_sidecar_path


def main() -> int:
    parser = teleop.create_parser(description=__doc__)
    default_name = f"recordings/teleop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(default_name),
        help=f"Output NPZ path (default: {default_name})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file",
    )
    args = parser.parse_args()
    outputs = [args.output]
    if not args.no_camera_recording:
        outputs.append(camera_sidecar_path(args.output))
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        parser.error(f"Output already exists: {joined} (use --overwrite to replace it)")
    if args.overwrite:
        for path in existing:
            path.unlink()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return teleop.run(args, record_path=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
