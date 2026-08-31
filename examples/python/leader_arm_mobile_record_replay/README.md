# Leader arm + mobile base + camera record/replay

This example targets the RBY1 **A model** with a two-wheel differential base.
It leaves the existing SDK examples unchanged and reuses their leader-arm and
gripper hardware drivers.

Camera support attaches to the shared memory produced by
`~/arpa_h_demo_robot_side/camera_stack/run_cameras.sh`. It does not open or
reconfigure any camera device. The recorded views are:

- head: left RGB image from the ZED stereo SHM
- right wrist: RGB image from `right_wrist` SHM
- left wrist: RGB image from `left_wrist` SHM

## Controls

- `Up` or `W`: drive forward
- `Down` or `S`: drive backward
- `Left` or `A`: rotate left
- `Right` or `D`: rotate right
- `Space`: stop the mobile base
- `Q` or `Ctrl+C`: stop the base and exit

The terminal does not report arrow-key release events. For safety, mobility uses
a dead-man timeout: if key-repeat messages stop for `--key-timeout` seconds, a
zero-velocity command is sent. Hold an arrow key to keep moving.

The program prints `Base input: ...` when it recognizes a key and
`Mobile command sent: ...` when that velocity reaches the SDK command stream.
If a terminal uses an unexpected escape sequence, run with `--keyboard-debug`
and use WASD as a fallback.

Arm and mobility commands use separate component-level SDK streams. Mobility
commands are sent immediately when the direction changes and refreshed every
0.40 seconds, instead of restarting the velocity profile at the 100 Hz arm
control rate. Startup also checks that both wheel joints report `ready=True`.

Leader-arm behavior is the same as example 35: hold the right or left leader
button while moving that arm; the triggers control the grippers.

## Start the camera producers

Run the camera stack in a separate terminal and leave it running:

```bash
cd ~/arpa_h_demo_robot_side/camera_stack
./run_cameras.sh
```

Wait until the ZED and both wrist camera nodes report that they have started.
The teleop/record programs only read these producer-owned SHM segments.

## Teleoperation

From the repository root:

```bash
.venv/bin/python examples/python/leader_arm_mobile_record_replay/teleop.py \
  --address 192.168.30.1:50051 --model a
```

Default base speeds are 0.10 m/s and 0.30 rad/s. Start lower when first testing:

```bash
.venv/bin/python examples/python/leader_arm_mobile_record_replay/teleop.py \
  --address 192.168.30.1:50051 --model a \
  --linear-speed 0.05 --angular-speed 0.15
```

To show the three camera views during teleoperation, add
`--visualize-cameras`:

```bash
.venv/bin/python examples/python/leader_arm_mobile_record_replay/teleop.py \
  --address 192.168.30.1:50051 --model a \
  --visualize-cameras
```

The preview defaults to 10 Hz so it does not add unnecessary load. Preview mode
defaults to `auto`: it first tries an OpenCV window and, if HighGUI is
unavailable, starts a browser preview at `http://127.0.0.1:8765/`.

When connected over SSH, forward that port from the client machine:

```bash
ssh -L 8765:127.0.0.1:8765 nvidia@ROBOT_IP
```

Then open `http://127.0.0.1:8765/` in the client browser. VS Code Remote users
can instead forward port 8765 from the **Ports** view. Force this backend with
`--camera-preview-mode browser`; change the port with
`--camera-preview-port`. Press `Q` in the terminal, or `Ctrl+C`, to stop the
whole teleoperation session.

## Record

`record.py` runs the same teleoperation loop and records robot data plus all
three RGB camera views. Camera sampling defaults to 30 Hz:

```bash
.venv/bin/python examples/python/leader_arm_mobile_record_replay/record.py \
  --address 192.168.30.1:50051 --model a \
  --output recordings/session_001.npz \
  --visualize-cameras
```

Omit `--visualize-cameras` to record without opening the preview window. An
existing output is not replaced unless `--overwrite` is supplied.

Two files are produced:

- `recordings/session_001.npz`: 100 Hz robot/leader/gripper/base trajectory
- `recordings/session_001.cameras.h5`: 30 Hz synchronized camera sidecar

The NPZ contains timestamps, all robot joint positions, base odometry, the
commanded `[vx, vy, wz]`, normalized gripper commands, and leader-arm positions.
New recordings use JPEG quality 90 by default so that three-camera recording
can sustain 30 Hz. The HDF5 file contains:

```text
/cameras/head/jpeg
/cameras/right_wrist/jpeg
/cameras/left_wrist/jpeg
/timestamps/monotonic_ns
/timestamps/unix_ns
/timestamps/head_source_monotonic_ns
/timestamps/head_camera_ns
```

Each `jpeg` entry is a variable-length encoded image. Robot and camera files
both contain absolute monotonic timestamps, so a camera sample can be paired
with the nearest robot sample without relying on their different sampling
rates. Lossless raw RGB remains available with `--camera-storage raw`; its
datasets are named `rgb`, and `--camera-compression` controls their HDF5
compression. Raw recording requires much more storage and may not sustain 30 Hz
on this computer.

The default head-camera recording profile is `record` (960x600), matching the
reference VR collector. The SHM source remains full-resolution. Available
profiles are:

- `--zed-record-profile full`: 1920x1200
- `--zed-record-profile record`: 960x600 (default)
- `--zed-record-profile fast`: 640x400

Camera images are large. Before a long collection, check available disk space.
At the default profile, uncompressed RGB would be about 6.0 GiB per minute.
JPEG storage is much smaller but remains scene-dependent. Use
`--zed-record-profile fast` or a lower `--camera-jpeg-quality` when storage or
write bandwidth is limited.

## Inspect a camera recording

The inspector prints frame count, effective recording rate, resolutions and
pixel statistics, then writes a middle-frame contact sheet:

```bash
.venv/bin/python \
  examples/python/leader_arm_mobile_record_replay/inspect_cameras.py \
  recordings/session_001.cameras.h5
```

The output is `recordings/session_001.contact_sheet.jpg`. It works with both
older raw-RGB files and new JPEG files. To export all three selected frames as
separate images:

```bash
.venv/bin/python \
  examples/python/leader_arm_mobile_record_replay/inspect_cameras.py \
  recordings/session_001.cameras.h5 \
  --index -1 --export-dir recordings/session_001_frames
```

To retain the previous robot-only behavior, use `--no-camera-recording`. This
also allows `record.py` to run without `run_cameras.sh`:

```bash
.venv/bin/python examples/python/leader_arm_mobile_record_replay/record.py \
  --address 192.168.30.1:50051 --model a \
  --output recordings/session_001.npz \
  --no-camera-recording
```

By default the camera worker uses
`~/openpi_rby1/.venv/bin/python`, where `h5py` and OpenCV are installed. Override
it with `--camera-python` if that environment moves. If any camera SHM stream
stays unavailable during recording, the session stops and saves the partial
robot and camera files instead of silently omitting images.

## Replay

Clear the surrounding floor area before replaying. Replay first moves the body
to the initial recorded pose, then streams arm positions, gripper targets, and
base velocity commands with the recorded timing. A three-second safety delay is
applied before motion; use `--start-delay` to change it.

```bash
.venv/bin/python examples/python/leader_arm_mobile_record_replay/replay.py \
  recordings/session_001.npz \
  --address 192.168.30.1:50051 --model a
```

Use `--skip-gripper` when the gripper is unavailable. Base replay is open-loop:
recorded odometry is retained for analysis, but replay sends the recorded base
velocities and does not close the loop on odometry. Wheel slip can therefore
cause the final base pose to differ from the recording.

Replay actuates the recorded robot, gripper, and mobile-base commands. Camera
frames are observations and are not sent to the robot during replay; they remain
available in the `.cameras.h5` sidecar for training, inspection, or timestamp
alignment.
