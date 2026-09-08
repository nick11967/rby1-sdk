#!/usr/bin/env python3
"""Web-based 3D URDF Data Analyzer & Episode Visualizer Server for RBY1.

Provides a standalone interactive web interface (Light Theme, High-Speed Buffer) to:
- Browse and select recorded episodes from recordings/ (.npz + .cameras.h5)
- Relative-odometry 3D URDF robot kinematic pose (always centered at origin)
- High-speed in-memory prefetching for buttery smooth 30+ FPS 3-camera playback
- Rich metadata overlays: resolution (640x480), Hz (29.9 FPS), current frame counter (# / total)
- Telemetry matrix (joint angles in deg/rad, base velocity, odometry)
- Playback / pause / speed control with keyboard shortcuts
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
import os
from pathlib import Path
import sys
import urllib.parse

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
RECORDINGS_DIR = ROOT_DIR / "recordings"
MODELS_DIR = ROOT_DIR / "models"


def get_episode_list(recordings_dir: Path) -> list[dict]:
    """Scan recordings directory and return sorted list of episode metadata."""
    candidate_dirs = [recordings_dir, Path("/home/nvidia/recordings"), Path("/mnt/ssd/rby1-sdk/recordings")]
    seen_ids = set()
    episodes = []

    for r_dir in candidate_dirs:
        if not r_dir.exists():
            continue
        npz_files = sorted(r_dir.glob("*.npz"), reverse=True)
        for npz_path in npz_files:
            stem = npz_path.stem
            if stem in seen_ids:
                continue
            seen_ids.add(stem)

            cam_h5 = npz_path.with_suffix(".cameras.h5")
            has_cameras = cam_h5.exists()

            try:
                stat = npz_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                mod_time = stat.st_mtime

                sample_count = 0
                duration_s = 0.0
                with np.load(npz_path, allow_pickle=True) as data:
                    if "time_s" in data:
                        sample_count = len(data["time_s"])
                        duration_s = float(data["time_s"][-1]) if sample_count > 0 else 0.0

                cam_frames = 0
                if has_cameras and h5py is not None:
                    try:
                        with h5py.File(cam_h5, "r") as h5:
                            if "cameras/head/jpeg" in h5:
                                cam_frames = len(h5["cameras/head/jpeg"])
                            elif "timestamps/unix_ns" in h5:
                                cam_frames = len(h5["timestamps/unix_ns"])
                    except Exception:
                        pass

                cam_fps = round(cam_frames / duration_s, 1) if (has_cameras and duration_s > 0 and cam_frames > 0) else 0.0

                episodes.append({
                    "id": stem,
                    "name": npz_path.name,
                    "filename": npz_path.name,
                    "has_cameras": has_cameras,
                    "sample_count": sample_count,
                    "duration_s": round(duration_s, 2),
                    "cam_frames": cam_frames,
                    "cam_fps": cam_fps,
                    "size_mb": round(size_mb, 2),
                    "mod_time": mod_time,
                })
            except Exception as exc:
                logging.warning("Error reading %s: %s", npz_path, exc)

    episodes.sort(key=lambda x: x["mod_time"], reverse=True)
    return episodes


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RBY1 Episode Data Analyzer & 3D Visualizer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Light Theme Palette */
            --bg-base: #f1f5f9;
            --bg-surface: #ffffff;
            --bg-card: #ffffff;
            --bg-card-alt: #f8fafc;
            --border-color: #cbd5e1;
            --border-focus: #6366f1;
            --accent-primary: #4f46e5;
            --accent-primary-hover: #4338ca;
            --accent-cyan: #0284c7;
            --accent-green: #15803d;
            --accent-amber: #d97706;
            --accent-red: #dc2626;
            --text-main: #0f172a;
            --text-muted: #475569;
            --text-dim: #64748b;
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }

        body {
            font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
            background: var(--bg-base);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            font-size: 14px;
        }

        /* Top Header */
        header {
            height: 60px;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            z-index: 20;
            box-shadow: var(--shadow-sm);
        }

        .brand-section {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-logo {
            width: 34px;
            height: 34px;
            background: linear-gradient(135deg, var(--accent-primary), #06b6d4);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 16px;
            color: #ffffff;
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);
        }

        .brand-title {
            font-size: 17px;
            font-weight: 700;
            letter-spacing: -0.3px;
            color: var(--text-main);
        }

        .brand-badge {
            font-size: 11px;
            font-weight: 700;
            background: #eef2ff;
            color: var(--accent-primary);
            border: 1px solid #c7d2fe;
            padding: 3px 8px;
            border-radius: 6px;
            text-transform: uppercase;
        }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .selector-label {
            font-size: 13px;
            color: var(--text-muted);
            font-weight: 600;
        }

        select.episode-select {
            background: var(--bg-card);
            color: var(--text-main);
            border: 1.5px solid var(--border-color);
            padding: 7px 14px;
            border-radius: 8px;
            font-size: 14px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            outline: none;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: var(--shadow-sm);
        }

        select.episode-select:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }

        .meta-badges {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .meta-badge {
            font-size: 12px;
            background: #f8fafc;
            border: 1px solid var(--border-color);
            padding: 5px 10px;
            border-radius: 6px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            box-shadow: var(--shadow-sm);
        }

        .meta-badge b {
            color: var(--accent-primary);
            font-weight: 700;
        }

        .btn-icon {
            background: var(--bg-card);
            border: 1.5px solid var(--border-color);
            color: var(--text-muted);
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: var(--shadow-sm);
        }

        .btn-icon:hover {
            color: var(--accent-primary);
            border-color: var(--accent-primary);
            background: #f8fafc;
        }

        /* Main Grid Area */
        main {
            flex: 1;
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 12px;
            padding: 12px;
            height: calc(100vh - 60px - 82px);
            overflow: hidden;
        }

        /* Left Column: 3D URDF & Telemetry */
        .left-col {
            display: flex;
            flex-direction: column;
            gap: 12px;
            height: 100%;
            overflow: hidden;
        }

        .viewport-card {
            flex: 1.5;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: var(--shadow-sm);
        }

        .card-header {
            position: absolute;
            top: 12px;
            left: 14px;
            right: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
            pointer-events: none;
        }

        .card-title {
            font-size: 13px;
            font-weight: 700;
            letter-spacing: -0.2px;
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(8px);
            padding: 5px 12px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
        }

        .card-tools {
            pointer-events: auto;
            display: flex;
            gap: 8px;
        }

        #urdf-canvas-container {
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at center, #ffffff 0%, #e2e8f0 100%);
        }

        .telemetry-card {
            flex: 1;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }

        .telemetry-header {
            padding: 9px 16px;
            background: #f8fafc;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            font-weight: 700;
        }

        .telemetry-body {
            flex: 1;
            padding: 10px 16px;
            overflow-y: auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12.5px;
        }

        .telemetry-group-title {
            font-size: 12px;
            color: var(--accent-primary);
            font-weight: 700;
            margin-bottom: 6px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 3px;
        }

        .telemetry-row {
            display: flex;
            justify-content: space-between;
            padding: 2.5px 0;
            color: var(--text-muted);
        }

        .telemetry-row .val {
            color: var(--text-main);
            font-weight: 600;
        }

        /* Right Column: 3 Cameras */
        .right-col {
            display: flex;
            flex-direction: column;
            gap: 12px;
            height: 100%;
            overflow: hidden;
        }

        .camera-card {
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: var(--shadow-sm);
        }

        .camera-card.head-cam {
            flex: 1.4;
        }

        .wrist-cam-row {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .camera-view-container {
            width: 100%;
            height: 100%;
            background: #090d16;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }

        .camera-img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            display: block;
        }

        .camera-overlay-header {
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            pointer-events: none;
            z-index: 5;
        }

        .camera-label {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(6px);
            color: #ffffff;
            font-size: 12px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .camera-meta-badge {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(6px);
            color: #38bdf8;
            font-size: 11.5px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .cam-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 8px #22c55e;
        }

        .camera-placeholder {
            color: #64748b;
            font-size: 13px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }

        /* Bottom Player Bar */
        footer {
            height: 82px;
            background: var(--bg-surface);
            border-top: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 8px 20px;
            gap: 6px;
            z-index: 20;
            box-shadow: 0 -2px 6px rgba(0, 0, 0, 0.03);
        }

        .timeline-container {
            width: 100%;
            display: flex;
            align-items: center;
            gap: 14px;
        }

        input[type="range"].timeline-slider {
            flex: 1;
            -webkit-appearance: none;
            height: 7px;
            background: #e2e8f0;
            border-radius: 4px;
            outline: none;
            cursor: pointer;
            transition: all 0.2s;
        }

        input[type="range"].timeline-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent-primary);
            box-shadow: 0 2px 6px rgba(79, 70, 229, 0.4);
            cursor: pointer;
            transition: transform 0.1s;
        }

        input[type="range"].timeline-slider::-webkit-slider-thumb:hover {
            transform: scale(1.3);
        }

        .controls-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .player-actions {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .btn-play {
            background: var(--accent-primary);
            color: #ffffff;
            border: none;
            padding: 6px 20px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
            box-shadow: 0 2px 6px rgba(79, 70, 229, 0.3);
        }

        .btn-play:hover {
            background: var(--accent-primary-hover);
        }

        .time-display {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13.5px;
            color: var(--text-main);
            font-weight: 600;
        }

        .speed-btn-group {
            display: flex;
            background: #f8fafc;
            border: 1.5px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }

        .btn-speed {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 5px 12px;
            font-size: 12.5px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-speed.active {
            background: var(--accent-primary);
            color: #ffffff;
            font-weight: 700;
        }

        .buffer-status-badge {
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-green);
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
        }

        .loading-overlay {
            position: absolute;
            inset: 0;
            background: rgba(255, 255, 255, 0.85);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 12px;
            z-index: 50;
            backdrop-filter: blur(4px);
        }

        .spinner {
            width: 36px;
            height: 36px;
            border: 3.5px solid #e2e8f0;
            border-top-color: var(--accent-primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>

    <!-- Importmap for Three.js and URDFLoader -->
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/",
            "three/examples/jsm/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/",
            "three/examples/jsm/loaders/STLLoader.js": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/STLLoader.js",
            "three/examples/jsm/loaders/ColladaLoader.js": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/ColladaLoader.js",
            "urdf-loader": "https://cdn.jsdelivr.net/npm/urdf-loader@0.12.5/src/URDFLoader.js"
        }
    }
    </script>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="brand-section">
            <div class="brand-logo">R</div>
            <div class="brand-title">RBY1 Data Analyzer</div>
            <span class="brand-badge">3D URDF & Multi-Cam</span>
        </div>

        <div class="header-controls">
            <span class="selector-label">Episode:</span>
            <select id="episode-select" class="episode-select">
                <option value="">Loading episodes...</option>
            </select>

            <div class="meta-badges">
                <div class="meta-badge">Samples: <b id="meta-samples">0</b></div>
                <div class="meta-badge">Duration: <b id="meta-duration">0.00s</b></div>
                <div class="meta-badge">Robot Rate: <b id="meta-rate">100 Hz</b></div>
                <div class="meta-badge">Cameras: <b id="meta-cams">-</b></div>
            </div>

            <button class="btn-icon" id="btn-refresh" title="새로고침">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
            </button>
        </div>
    </header>

    <!-- Main Viewport Grid -->
    <main>
        <!-- Left: 3D Robot URDF Viewport & Telemetry -->
        <div class="left-col">
            <div class="viewport-card">
                <div class="card-header">
                    <div class="card-title">3D Kinematic Pose (원점 기준)</div>
                    <div class="card-tools">
                        <button class="btn-icon" id="btn-reset-cam" title="로봇 시점 중앙 초기화">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="m10 15 5-3-5-3v6Z"/></svg>
                        </button>
                    </div>
                </div>
                <div id="urdf-canvas-container"></div>
                <div id="urdf-loading" class="loading-overlay">
                    <div class="spinner"></div>
                    <div style="font-size: 13px; font-weight: 600; color: var(--text-muted)">Loading 3D URDF Robot Model...</div>
                </div>
            </div>

            <div class="telemetry-card">
                <div class="telemetry-header">
                    <span>Telemetry & Actions Matrix</span>
                    <span id="telemetry-time" style="color: var(--accent-primary); font-family: 'JetBrains Mono'">t = 0.00s</span>
                </div>
                <div class="telemetry-body">
                    <div>
                        <div class="telemetry-group-title">Right Arm & Gripper</div>
                        <div class="telemetry-row"><span>R_Shoulder (P/R/Y):</span> <span class="val" id="val-r-shoulder">0.0, 0.0, 0.0</span></div>
                        <div class="telemetry-row"><span>R_Elbow:</span> <span class="val" id="val-r-elbow">0.0°</span></div>
                        <div class="telemetry-row"><span>R_Wrist (Y1/P/Y2):</span> <span class="val" id="val-r-wrist">0.0, 0.0, 0.0</span></div>
                        <div class="telemetry-row"><span>R_Gripper:</span> <span class="val" id="val-r-gripper" style="color: var(--accent-primary); font-weight: 700;">0.0% (Closed)</span></div>

                        <div class="telemetry-group-title" style="margin-top: 10px;">Left Arm & Gripper</div>
                        <div class="telemetry-row"><span>L_Shoulder (P/R/Y):</span> <span class="val" id="val-l-shoulder">0.0, 0.0, 0.0</span></div>
                        <div class="telemetry-row"><span>L_Elbow:</span> <span class="val" id="val-l-elbow">0.0°</span></div>
                        <div class="telemetry-row"><span>L_Wrist (Y1/P/Y2):</span> <span class="val" id="val-l-wrist">0.0, 0.0, 0.0</span></div>
                        <div class="telemetry-row"><span>L_Gripper:</span> <span class="val" id="val-l-gripper" style="color: var(--accent-primary); font-weight: 700;">0.0% (Closed)</span></div>
                    </div>

                    <div>
                        <div class="telemetry-group-title">Torso & Head (deg)</div>
                        <div class="telemetry-row"><span>Torso (0..2):</span> <span class="val" id="val-torso-1">0.0, 0.0, 0.0</span></div>
                        <div class="telemetry-row"><span>Torso (3..5):</span> <span class="val" id="val-torso-2">0.0, 0.0, 0.0</span></div>
                        <div class="telemetry-row"><span>Head (Pan/Tilt):</span> <span class="val" id="val-head">0.0°, 0.0°</span></div>

                        <div class="telemetry-group-title" style="margin-top: 10px;">Mobile Base & Odometry</div>
                        <div class="telemetry-row"><span>Rel Odom (X, Y, θ):</span> <span class="val" id="val-odom">[0.00, 0.00, 0.00]</span></div>
                        <div class="telemetry-row"><span>Wheels (R, L deg):</span> <span class="val" id="val-wheels">0.0, 0.0</span></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right: 3 Synchronized Camera Views -->
        <div class="right-col">
            <div class="camera-card head-cam">
                <div class="camera-overlay-header">
                    <div class="camera-label"><span class="cam-dot"></span> Head ZED Camera</div>
                    <div class="camera-meta-badge" id="badge-head">640x480 @ 30 FPS | Frame 0/0</div>
                </div>
                <div class="camera-view-container">
                    <img id="img-head" class="camera-img" src="" alt="">
                    <div id="placeholder-head" class="camera-placeholder">Head Camera Ready</div>
                </div>
            </div>

            <div class="wrist-cam-row">
                <div class="camera-card">
                    <div class="camera-overlay-header">
                        <div class="camera-label"><span class="cam-dot"></span> Left Wrist Camera</div>
                        <div class="camera-meta-badge" id="badge-left">640x480 | Frame 0/0</div>
                    </div>
                    <div class="camera-view-container">
                        <img id="img-left" class="camera-img" src="" alt="">
                        <div id="placeholder-left" class="camera-placeholder">Left Wrist Ready</div>
                    </div>
                </div>

                <div class="camera-card">
                    <div class="camera-overlay-header">
                        <div class="camera-label"><span class="cam-dot"></span> Right Wrist Camera</div>
                        <div class="camera-meta-badge" id="badge-right">640x480 | Frame 0/0</div>
                    </div>
                    <div class="camera-view-container">
                        <img id="img-right" class="camera-img" src="" alt="">
                        <div id="placeholder-right" class="camera-placeholder">Right Wrist Ready</div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Bottom Player / Scrubber Bar -->
    <footer>
        <div class="timeline-container">
            <span class="time-display" id="time-current">00:00.00</span>
            <input type="range" id="timeline-slider" class="timeline-slider" min="0" max="0" value="0" step="1">
            <span class="time-display" id="time-total" style="color: var(--text-muted)">00:00.00</span>
        </div>

        <div class="controls-container">
            <div class="player-actions">
                <button class="btn-play" id="btn-play">
                    <svg id="icon-play" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                    <span id="text-play">Play</span>
                </button>

                <button class="btn-icon" id="btn-prev-frame" title="이전 프레임 (Left Arrow)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="15 18 9 12 15 6"/></svg>
                </button>

                <button class="btn-icon" id="btn-next-frame" title="다음 프레임 (Right Arrow)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
                </button>

                <div class="time-display" id="frame-counter" style="margin-left: 10px; color: var(--text-dim)">Frame: 0 / 0</div>
            </div>

            <div style="display: flex; align-items: center; gap: 14px;">
                <div class="buffer-status-badge" id="buffer-status">Buffer: Ready</div>
                <div class="speed-btn-group">
                    <button class="btn-speed" data-speed="0.25">0.25x</button>
                    <button class="btn-speed" data-speed="0.5">0.5x</button>
                    <button class="btn-speed active" data-speed="1.0">1.0x</button>
                    <button class="btn-speed" data-speed="2.0">2.0x</button>
                </div>
            </div>
        </div>
    </footer>

    <!-- Three.js + URDF Application Logic -->
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
        import URDFLoader from 'urdf-loader';

        // State variables
        let scene, camera, renderer, controls, robotModel;
        let currentEpisodeId = null;
        let episodeData = null;
        let isPlaying = false;
        let playbackSpeed = 1.0;
        let currentFrameIndex = 0;
        let totalFrames = 0;
        let animationTimer = null;
        let lastFrameTime = 0;
        let lastCamIdx = -1;

        // In-Memory Fast Frame Cache
        const frameBlobCache = new Map(); // key: `${role}_${camIdx}` -> Blob URL

        // Elements
        const episodeSelect = document.getElementById('episode-select');
        const slider = document.getElementById('timeline-slider');
        const btnPlay = document.getElementById('btn-play');
        const textPlay = document.getElementById('text-play');
        const iconPlay = document.getElementById('icon-play');
        const btnPrev = document.getElementById('btn-prev-frame');
        const btnNext = document.getElementById('btn-next-frame');
        const btnResetCam = document.getElementById('btn-reset-cam');
        const btnRefresh = document.getElementById('btn-refresh');
        const timeCurrent = document.getElementById('time-current');
        const timeTotal = document.getElementById('time-total');
        const frameCounter = document.getElementById('frame-counter');
        const urdfLoading = document.getElementById('urdf-loading');
        const bufferStatus = document.getElementById('buffer-status');

        // Telemetry elements
        const valRShoulder = document.getElementById('val-r-shoulder');
        const valRElbow = document.getElementById('val-r-elbow');
        const valRWrist = document.getElementById('val-r-wrist');
        const valRGripper = document.getElementById('val-r-gripper');
        const valLShoulder = document.getElementById('val-l-shoulder');
        const valLElbow = document.getElementById('val-l-elbow');
        const valLWrist = document.getElementById('val-l-wrist');
        const valLGripper = document.getElementById('val-l-gripper');
        const valTorso1 = document.getElementById('val-torso-1');
        const valTorso2 = document.getElementById('val-torso-2');
        const valHead = document.getElementById('val-head');
        const valOdom = document.getElementById('val-odom');
        const valWheels = document.getElementById('val-wheels');
        const telemetryTime = document.getElementById('telemetry-time');

        // Camera elements
        const imgHead = document.getElementById('img-head');
        const imgLeft = document.getElementById('img-left');
        const imgRight = document.getElementById('img-right');
        const badgeHead = document.getElementById('badge-head');
        const badgeLeft = document.getElementById('badge-left');
        const badgeRight = document.getElementById('badge-right');
        const placeholderHead = document.getElementById('placeholder-head');
        const placeholderLeft = document.getElementById('placeholder-left');
        const placeholderRight = document.getElementById('placeholder-right');

        // Fetch Episode List immediately
        fetchEpisodes();

        // Initialize 3D Three.js Scene (Light Theme)
        function init3D() {
            try {
                const container = document.getElementById('urdf-canvas-container');
                const width = container.clientWidth || 600;
                const height = container.clientHeight || 400;

                scene = new THREE.Scene();
                camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
                camera.position.set(2.2, -2.5, 1.8);
                camera.up.set(0, 0, 1);

                renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                renderer.setSize(width, height);
                renderer.setPixelRatio(window.devicePixelRatio);
                renderer.shadowMap.enabled = true;
                container.appendChild(renderer.domElement);

                controls = new OrbitControls(camera, renderer.domElement);
                controls.target.set(0, 0, 0.6);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;

                // Lights for Light Mode
                const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
                scene.add(ambientLight);

                const dirLight = new THREE.DirectionalLight(0xffffff, 1.6);
                dirLight.position.set(5, 5, 10);
                dirLight.castShadow = true;
                scene.add(dirLight);

                const dirLight2 = new THREE.DirectionalLight(0x93c5fd, 0.8);
                dirLight2.position.set(-5, -5, 5);
                scene.add(dirLight2);

                // Grid Ground (Light Theme Crisp Gray Grid)
                const gridHelper = new THREE.GridHelper(10, 20, 0x4f46e5, 0xcbd5e1);
                gridHelper.rotation.x = Math.PI / 2;
                scene.add(gridHelper);

                // Coordinate axes helper
                const axesHelper = new THREE.AxesHelper(0.5);
                scene.add(axesHelper);

                // Load URDF with loader.packages
                const manager = new THREE.LoadingManager();
                manager.onLoad = () => {
                    urdfLoading.style.display = 'none';
                };
                manager.onError = (url) => {
                    console.warn('Asset loading warning:', url);
                };

                const loader = new URDFLoader(manager);
                loader.packages = {
                    '': '/models/rby1a/urdf',
                    '.': '/models/rby1a/urdf',
                    'rby1_model/rby1a': '/models/rby1a/urdf'
                };

                loader.load('/models/rby1a/urdf/model.urdf', (robot) => {
                    robotModel = robot;
                    scene.add(robotModel);
                    urdfLoading.style.display = 'none';
                    console.log('RBY1 URDF model loaded successfully!');
                    if (episodeData) updateFrame(currentFrameIndex);
                }, undefined, (err) => {
                    console.error('Error loading URDF:', err);
                    urdfLoading.style.display = 'none';
                });

                window.addEventListener('resize', onWindowResize);
                animate();
            } catch (err) {
                console.error('3D init error:', err);
                urdfLoading.style.display = 'none';
            }
        }

        function onWindowResize() {
            const container = document.getElementById('urdf-canvas-container');
            if (!container) return;
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        // Fetch Episode List
        async function fetchEpisodes() {
            try {
                const res = await fetch('/api/episodes');
                const episodes = await res.json();
                episodeSelect.innerHTML = '';

                if (episodes.length === 0) {
                    episodeSelect.innerHTML = '<option value="">No episodes found in recordings/</option>';
                    return;
                }

                episodes.forEach(ep => {
                    const opt = document.createElement('option');
                    opt.value = ep.id;
                    opt.textContent = `${ep.name} (${ep.duration_s}s, ${ep.sample_count} samples${ep.has_cameras ? ` | ${ep.cam_frames} Cams @ ${ep.cam_fps} FPS` : ''})`;
                    episodeSelect.appendChild(opt);
                });

                // Load first episode automatically
                if (episodes.length > 0) {
                    loadEpisode(episodes[0].id);
                }
            } catch (err) {
                console.error('Failed to fetch episodes:', err);
            }
        }

        // Prefetch Camera Frames into RAM Cache for smooth 30+ FPS playback
        async function prefetchFrames(episodeId, totalCamFrames) {
            frameBlobCache.clear();
            bufferStatus.textContent = 'Buffering: 0%';
            bufferStatus.style.color = 'var(--accent-amber)';

            const roles = ['head', 'left_wrist', 'right_wrist'];
            let loaded = 0;
            const totalToLoad = totalCamFrames * 3;

            // Load in concurrent batches of 15
            for (let i = 0; i < totalCamFrames; i += 5) {
                if (currentEpisodeId !== episodeId) return; // cancelled
                const promises = [];
                for (let j = i; j < Math.min(totalCamFrames, i + 5); j++) {
                    for (const role of roles) {
                        const key = `${role}_${j}`;
                        if (!frameBlobCache.has(key)) {
                            promises.push(
                                fetch(`/api/episode/${episodeId}/camera/${role}/${j}`)
                                    .then(res => res.blob())
                                    .then(blob => {
                                        frameBlobCache.set(key, URL.createObjectURL(blob));
                                        loaded++;
                                    })
                                    .catch(() => {})
                            );
                        }
                    }
                }
                await Promise.all(promises);
                const pct = Math.round((loaded / totalToLoad) * 100);
                bufferStatus.textContent = `Buffer: ${pct}% (${Math.round(loaded / 3)} / ${totalCamFrames} frames)`;
            }

            bufferStatus.textContent = `Buffer: 100% (${totalCamFrames} frames in RAM)`;
            bufferStatus.style.color = 'var(--accent-green)';
        }

        // Load Episode Data
        async function loadEpisode(id) {
            currentEpisodeId = id;
            lastCamIdx = -1;
            pause();
            try {
                const res = await fetch(`/api/episode/${id}`);
                episodeData = await res.json();

                totalFrames = episodeData.time_s.length;
                slider.max = totalFrames - 1;
                slider.value = 0;
                currentFrameIndex = 0;

                const duration = episodeData.time_s[totalFrames - 1] || 0;
                document.getElementById('meta-samples').textContent = totalFrames;
                document.getElementById('meta-duration').textContent = `${duration.toFixed(2)}s`;
                document.getElementById('meta-rate').textContent = `${(totalFrames / (duration || 1)).toFixed(1)} Hz`;
                
                const camFps = (episodeData.has_cameras && duration > 0) ? (episodeData.cam_frames / duration).toFixed(1) : '30.0';
                document.getElementById('meta-cams').textContent = episodeData.has_cameras ? `3x 640x480 @ ${camFps} FPS (${episodeData.cam_frames}f)` : 'None';
                timeTotal.textContent = formatTime(duration);

                // Reset robot position in scene to Origin (0, 0, 0)
                if (robotModel) {
                    robotModel.position.set(0, 0, 0);
                    robotModel.rotation.set(0, 0, 0);
                }

                updateFrame(0);

                // Start background prefetching for instant 30 FPS playback
                if (episodeData.has_cameras && episodeData.cam_frames > 0) {
                    prefetchFrames(id, episodeData.cam_frames);
                }
            } catch (err) {
                console.error(`Failed to load episode ${id}:`, err);
            }
        }

        // Format seconds to MM:SS.ss
        function formatTime(s) {
            const mins = Math.floor(s / 60);
            const secs = s % 60;
            return `${String(mins).padStart(2, '0')}:${secs.toFixed(2).padStart(5, '0')}`;
        }

        // Update Frame at specific index
        function updateFrame(idx) {
            if (!episodeData || idx >= totalFrames) return;
            currentFrameIndex = idx;
            slider.value = idx;

            const t = episodeData.time_s[idx] || 0;
            timeCurrent.textContent = formatTime(t);
            frameCounter.textContent = `Frame: ${idx + 1} / ${totalFrames}`;
            telemetryTime.textContent = `t = ${t.toFixed(3)}s`;

            const pos = episodeData.position ? episodeData.position[idx] : null;

            // Update 3D URDF Joint Angles
            if (robotModel && pos) {
                const jointNames = episodeData.joint_names || [];
                for (let i = 0; i < jointNames.length; i++) {
                    const jName = jointNames[i];
                    if (robotModel.joints && robotModel.joints[jName] && typeof pos[i] === 'number' && !isNaN(pos[i])) {
                        robotModel.setJointValue(jName, pos[i]);
                    }
                }

                // Update mobile base relative odometry position in 3D (Always starting at (0, 0, 0))
                if (episodeData.odometry_pose && episodeData.odometry_pose[idx]) {
                    const odom = episodeData.odometry_pose[idx]; // [x_rel, y_rel, theta_rel]
                    if (Array.isArray(odom) && typeof odom[0] === 'number' && !isNaN(odom[0])) {
                        robotModel.position.set(odom[0], odom[1] || 0, 0);
                        robotModel.rotation.z = odom[2] || 0;
                    }
                }

                // Animate Gripper Finger Prismatic Joints in 3D (0.0=Open, 1.0=Closed)
                const gripper = (episodeData.gripper_command && episodeData.gripper_command[idx]) ? episodeData.gripper_command[idx] : [0, 0];
                const rGrip = (typeof gripper[0] === 'number') ? Math.min(1.0, Math.max(0.0, gripper[0])) : 0.0;
                const lGrip = (typeof gripper[1] === 'number') ? Math.min(1.0, Math.max(0.0, gripper[1])) : 0.0;

                if (robotModel.joints) {
                    if (robotModel.joints['gripper_finger_r1']) robotModel.setJointValue('gripper_finger_r1', -0.045 * rGrip);
                    if (robotModel.joints['gripper_finger_r2']) robotModel.setJointValue('gripper_finger_r2', 0.045 * rGrip);
                    if (robotModel.joints['gripper_finger_l1']) robotModel.setJointValue('gripper_finger_l1', -0.045 * lGrip);
                    if (robotModel.joints['gripper_finger_l2']) robotModel.setJointValue('gripper_finger_l2', 0.045 * lGrip);
                }
            }

            // Update Telemetry Numbers (degrees & grippers)
            if (pos && Array.isArray(pos)) {
                const deg = pos.map(v => typeof v === 'number' && !isNaN(v) ? (v * 180 / Math.PI).toFixed(1) : '0.0');
                if (deg.length >= 22) {
                    valRShoulder.textContent = `${deg[8]}, ${deg[9]}, ${deg[10]}`;
                    valRElbow.textContent = `${deg[11]}°`;
                    valRWrist.textContent = `${deg[12]}, ${deg[13]}, ${deg[14]}`;
                    valLShoulder.textContent = `${deg[15]}, ${deg[16]}, ${deg[17]}`;
                    valLElbow.textContent = `${deg[18]}°`;
                    valLWrist.textContent = `${deg[19]}, ${deg[20]}, ${deg[21]}`;
                    valTorso1.textContent = `${deg[2]}, ${deg[3]}, ${deg[4]}`;
                    valTorso2.textContent = `${deg[5]}, ${deg[6]}, ${deg[7]}`;
                }
                if (deg.length >= 24) {
                    valHead.textContent = `${deg[22]}°, ${deg[23]}°`;
                }
                if (deg.length >= 2) {
                    valWheels.textContent = `${deg[0]}, ${deg[1]}`;
                }

                const gripper = (episodeData.gripper_command && episodeData.gripper_command[idx]) ? episodeData.gripper_command[idx] : [0, 0];
                const rGrip = (typeof gripper[0] === 'number') ? Math.min(1.0, Math.max(0.0, gripper[0])) : 0.0;
                const lGrip = (typeof gripper[1] === 'number') ? Math.min(1.0, Math.max(0.0, gripper[1])) : 0.0;
                if (valRGripper) valRGripper.textContent = `${(rGrip * 100).toFixed(1)}% (${rGrip >= 0.5 ? 'Open' : 'Closed'})`;
                if (valLGripper) valLGripper.textContent = `${(lGrip * 100).toFixed(1)}% (${lGrip >= 0.5 ? 'Open' : 'Closed'})`;
            }

            if (episodeData.odometry_pose && episodeData.odometry_pose[idx]) {
                const o = episodeData.odometry_pose[idx];
                if (Array.isArray(o) && !isNaN(o[0])) {
                    valOdom.textContent = `[${o[0].toFixed(2)}, ${o[1].toFixed(2)}, ${((o[2] || 0) * 180 / Math.PI).toFixed(1)}°]`;
                }
            }

            // Update Camera Images & Badges
            if (episodeData.has_cameras && episodeData.cam_frames > 0) {
                const camIdx = Math.min(
                    episodeData.cam_frames - 1,
                    Math.max(0, Math.floor(idx * (episodeData.cam_frames / totalFrames)))
                );

                const duration = episodeData.time_s[totalFrames - 1] || 1;
                const camFps = (episodeData.cam_frames / duration).toFixed(1);

                badgeHead.textContent = `640x480 @ ${camFps} FPS | #${camIdx + 1} / ${episodeData.cam_frames}`;
                badgeLeft.textContent = `640x480 @ ${camFps} FPS | #${camIdx + 1} / ${episodeData.cam_frames}`;
                badgeRight.textContent = `640x480 @ ${camFps} FPS | #${camIdx + 1} / ${episodeData.cam_frames}`;

                if (camIdx !== lastCamIdx) {
                    lastCamIdx = camIdx;

                    const urlHead = frameBlobCache.get(`head_${camIdx}`) || `/api/episode/${currentEpisodeId}/camera/head/${camIdx}`;
                    const urlLeft = frameBlobCache.get(`left_wrist_${camIdx}`) || `/api/episode/${currentEpisodeId}/camera/left_wrist/${camIdx}`;
                    const urlRight = frameBlobCache.get(`right_wrist_${camIdx}`) || `/api/episode/${currentEpisodeId}/camera/right_wrist/${camIdx}`;

                    imgHead.src = urlHead;
                    imgLeft.src = urlLeft;
                    imgRight.src = urlRight;
                }
                placeholderHead.style.display = 'none';
                placeholderLeft.style.display = 'none';
                placeholderRight.style.display = 'none';
            } else {
                lastCamIdx = -1;
                imgHead.src = '';
                imgLeft.src = '';
                imgRight.src = '';
                placeholderHead.style.display = 'block';
                placeholderLeft.style.display = 'block';
                placeholderRight.style.display = 'block';
                badgeHead.textContent = 'Camera Off';
                badgeLeft.textContent = 'Camera Off';
                badgeRight.textContent = 'Camera Off';
            }
        }

        // Playback Loop
        function play() {
            if (isPlaying) return;
            isPlaying = true;
            textPlay.textContent = 'Pause';
            iconPlay.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
            lastFrameTime = performance.now();

            function step(now) {
                if (!isPlaying) return;
                const dt = (now - lastFrameTime) / 1000;
                lastFrameTime = now;

                // 100 Hz dataset: advance frames by dt * 100 * playbackSpeed
                const frameStep = Math.max(1, Math.round(dt * 100 * playbackSpeed));
                currentFrameIndex += frameStep;

                if (currentFrameIndex >= totalFrames) {
                    currentFrameIndex = 0; // Loop
                }

                updateFrame(currentFrameIndex);
                animationTimer = requestAnimationFrame(step);
            }
            animationTimer = requestAnimationFrame(step);
        }

        function pause() {
            isPlaying = false;
            textPlay.textContent = 'Play';
            iconPlay.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
            if (animationTimer) cancelAnimationFrame(animationTimer);
        }

        function togglePlay() {
            if (isPlaying) pause();
            else play();
        }

        // Event Listeners
        episodeSelect.addEventListener('change', (e) => {
            if (e.target.value) loadEpisode(e.target.value);
        });

        slider.addEventListener('input', (e) => {
            pause();
            updateFrame(parseInt(e.target.value));
        });

        btnPlay.addEventListener('click', togglePlay);

        btnPrev.addEventListener('click', () => {
            pause();
            if (currentFrameIndex > 0) updateFrame(currentFrameIndex - 1);
        });

        btnNext.addEventListener('click', () => {
            pause();
            if (currentFrameIndex < totalFrames - 1) updateFrame(currentFrameIndex + 1);
        });

        btnResetCam.addEventListener('click', () => {
            camera.position.set(2.2, -2.5, 1.8);
            controls.target.set(0, 0, 0.6);
            controls.update();
        });

        btnRefresh.addEventListener('click', fetchEpisodes);

        // Speed buttons
        document.querySelectorAll('.btn-speed').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.btn-speed').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                playbackSpeed = parseFloat(btn.dataset.speed);
            });
        });

        // Keyboard Shortcuts
        window.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                e.preventDefault();
                togglePlay();
            } else if (e.code === 'ArrowLeft') {
                e.preventDefault();
                pause();
                if (currentFrameIndex > 0) updateFrame(currentFrameIndex - 1);
            } else if (e.code === 'ArrowRight') {
                e.preventDefault();
                pause();
                if (currentFrameIndex < totalFrames - 1) updateFrame(currentFrameIndex + 1);
            } else if (e.code === 'Home') {
                e.preventDefault();
                updateFrame(0);
            }
        });

        // Startup
        init3D();
    </script>
</body>
</html>
"""


class VisualizerRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for Episode Visualizer Server."""

    recordings_dir = RECORDINGS_DIR

    def log_message(self, format, *args):
        try:
            msg = format % args
            if "camera" not in msg:
                logging.info("%s - %s", self.address_string(), msg)
        except Exception:
            pass

    def _resolve_episode_files(self, episode_id: str) -> tuple[Optional[Path], Optional[Path]]:
        candidate_dirs = [self.recordings_dir, Path("/home/nvidia/recordings"), Path("/mnt/ssd/rby1-sdk/recordings")]
        for r_dir in candidate_dirs:
            if not r_dir.exists():
                continue
            npz = r_dir / f"{episode_id}.npz"
            if npz.exists():
                cam = r_dir / f"{episode_id}.cameras.h5"
                return npz, cam
        return None, None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. Main SPA Page
        if path in ("/", "/index.html"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        # 2. Static Meshes & Models Endpoint
        if path.startswith("/models/"):
            rel_path = path[len("/models/") :]
            file_path = MODELS_DIR / rel_path
            if file_path.is_file():
                self.send_response(HTTPStatus.OK)
                mime, _ = mimetypes.guess_type(str(file_path))
                if not mime:
                    mime = "application/octet-stream"
                self.send_header("Content-Type", mime)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(HTTPStatus.NOT_FOUND, f"Model file not found: {rel_path}")
                return

        # 3. API: List Episodes
        if path == "/api/episodes":
            episodes = get_episode_list(self.recordings_dir)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(episodes).encode("utf-8"))
            return

        # 4. API: Episode Kinematics
        if path.startswith("/api/episode/") and "/camera/" not in path:
            episode_id = path[len("/api/episode/") :].strip("/")
            npz_path, cam_path = self._resolve_episode_files(episode_id)

            if npz_path is None or not npz_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, f"Episode {episode_id} not found")
                return

            try:
                with np.load(npz_path, allow_pickle=True) as data:
                    time_s = data["time_s"].tolist() if "time_s" in data else []
                    position = data["position"].tolist() if "position" in data else []
                    joint_names = data["joint_names"].tolist() if "joint_names" in data else []
                    
                    odometry_pose = []
                    if "odometry_pose" in data:
                        raw_odom = np.asarray(data["odometry_pose"])
                        if raw_odom.ndim == 2 and raw_odom.shape[1] == 3 and len(raw_odom) > 0:
                            rel_odom = raw_odom - raw_odom[0]
                            odometry_pose = rel_odom.tolist()
                        else:
                            odometry_pose = raw_odom.tolist()
                    elif "odometry" in data:
                        odom_raw = np.asarray(data["odometry"])
                        if odom_raw.ndim == 3 and odom_raw.shape[1:] == (3, 3) and len(odom_raw) > 0:
                            x = odom_raw[:, 0, 2]
                            y = odom_raw[:, 1, 2]
                            yaw = np.arctan2(odom_raw[:, 1, 0], odom_raw[:, 0, 0])
                            full_odom = np.column_stack([x, y, yaw])
                            rel_odom = full_odom - full_odom[0]
                            odometry_pose = rel_odom.tolist()
                        elif odom_raw.ndim == 2 and odom_raw.shape[1] == 3:
                            rel_odom = odom_raw - odom_raw[0]
                            odometry_pose = rel_odom.tolist()

                    if not odometry_pose and position:
                        odometry_pose = [[0.0, 0.0, 0.0] for _ in range(len(position))]

                    gripper_command = []
                    if "gripper_command" in data:
                        raw_gc = np.asarray(data["gripper_command"])
                        if raw_gc.ndim == 2 and raw_gc.shape[1] == 2:
                            gripper_command = raw_gc.tolist()
                    elif "gripper" in data:
                        raw_gc = np.asarray(data["gripper"])
                        if raw_gc.ndim == 2 and raw_gc.shape[1] == 2:
                            gripper_command = raw_gc.tolist()

                    if not gripper_command and position:
                        gripper_command = [[0.0, 0.0] for _ in range(len(position))]

                cam_frames = 0
                has_cameras = cam_path is not None and cam_path.exists()
                if has_cameras and h5py is not None:
                    try:
                        with h5py.File(cam_path, "r") as h5:
                            if "cameras/head/jpeg" in h5:
                                cam_frames = len(h5["cameras/head/jpeg"])
                            elif "timestamps/unix_ns" in h5:
                                cam_frames = len(h5["timestamps/unix_ns"])
                    except Exception:
                        pass

                payload = {
                    "id": episode_id,
                    "joint_names": joint_names,
                    "time_s": time_s,
                    "position": position,
                    "odometry_pose": odometry_pose,
                    "gripper_command": gripper_command,
                    "has_cameras": has_cameras,
                    "cam_frames": cam_frames,
                }
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode("utf-8"))
                return
            except Exception as exc:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return

        # 5. API: Synchronized Camera Frame
        if "/camera/" in path:
            parts = path.strip("/").split("/")
            if len(parts) == 6:
                _, _, episode_id, _, role, idx_str = parts
                try:
                    frame_idx = int(idx_str)
                    _, cam_path = self._resolve_episode_files(episode_id)

                    if cam_path is None or not cam_path.exists() or h5py is None:
                        self.send_error(HTTPStatus.NOT_FOUND, "Camera file not available")
                        return

                    dataset_key = f"cameras/{role}/jpeg"
                    with h5py.File(cam_path, "r") as h5:
                        if dataset_key not in h5:
                            self.send_error(HTTPStatus.NOT_FOUND, f"Camera role {role} not found")
                            return

                        ds = h5[dataset_key]
                        if frame_idx < 0 or frame_idx >= len(ds):
                            frame_idx = max(0, min(frame_idx, len(ds) - 1))

                        jpeg_bytes = ds[frame_idx]

                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(bytes(jpeg_bytes))
                    return
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                    return

        self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")


def main() -> int:
    parser = argparse.ArgumentParser(description="RBY1 Web-based 3D URDF Data Analyzer Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=RECORDINGS_DIR,
        help=f"Path to recordings directory (default: {RECORDINGS_DIR})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    VisualizerRequestHandler.recordings_dir = args.recordings_dir

    server = ThreadingHTTPServer((args.host, args.port), VisualizerRequestHandler)
    logging.info("=" * 65)
    logging.info("  RBY1 3D URDF Data Analyzer & Episode Visualizer Server  ")
    logging.info("=" * 65)
    logging.info("Server running at: http://localhost:%d", args.port)
    logging.info("Network access:    http://<robot-ip>:%d", args.port)
    logging.info("Recordings dir:    %s", args.recordings_dir)
    logging.info("=" * 65)
    logging.info("Press Ctrl+C to stop the server.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("\nStopping Visualizer Server...")
    finally:
        server.server_close()
        logging.info("Server stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
