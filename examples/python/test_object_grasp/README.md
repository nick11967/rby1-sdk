# Object Grasp & Contact Detection (Approach A)

This directory implements **Approach A (Finger Position Stall)** for reliable object grasp detection (apples, bottles, tools, cups) on the RB-Y1 robot.

---

### How Approach A Works:
1. **Trigger Command (`target`)**: When you squeeze the trigger, you command the gripper to close (`100% closed`).
2. **Finger Encoders (`actual`)**:
   - In empty air, the fingers close completely (`100% closed`).
   - When holding an apple/object, the object physically blocks the fingers from closing all the way (e.g. fingers stop at `40% closed`).
3. **Grasp Detection**:
   $$\text{Trigger} > 35\% \quad\land\quad \text{Actual} < 88\% \quad\land\quad (\text{Trigger} - \text{Actual}) > 12\% \implies \mathbf{*** GRASPED ***}$$
   It also reports the exact object thickness/opening (e.g. `Object: 55% wide`)!
4. **Wrist FT Sensor**: Simultaneously monitored to detect **table contact** and **lift force**!

---

## 1. Quick Keyboard Test (No Leader Arm required)

Test grasping an apple/object immediately with the keyboard:

```bash
/mnt/ssd/rby1-sdk/.venv/bin/python /mnt/ssd/rby1-sdk/examples/python/test_object_grasp/quick_grasp_test.py --address 192.168.30.1:50051 --model a
```

**Controls**:
- `[C]`: Close gripper onto the apple
  - If apple is between fingers: Displays `*** GRASPED (Object: 60% wide) ***`
  - If no object is there: Displays `CLOSED / EMPTY (No Object)`
- `[O]`: Open gripper
- `[T]`: Re-tare wrist FT sensor
- `[Q]`: Quit

---

## 2. Teleoperation with Grasp Detection (Leader Arm)

Full teleoperation with real-time Approach A grasp detection and live HUD:

```bash
/mnt/ssd/rby1-sdk/.venv/bin/python /mnt/ssd/rby1-sdk/examples/python/test_object_grasp/teleop_grasp_test.py --address 192.168.30.1:50051 --model a
```

**Live HUD Preview**:
```text
[GRASP HUD] R: Trig: 90% Pos: 45% [*** GRASPED (55% wide) ***] ΔF= 1.4N
[GRASP HUD] R: Trig:100% Pos:100% [CLOSED / EMPTY            ] ΔF= 0.2N
```
If your hand touches the table, it also flags: `[Table Contact: 14.2N]`.
