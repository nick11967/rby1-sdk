# 설계 문서: 텔레옵 일시정지 및 외부 제어 연동 (Pause & Auto-Resume Architecture)

이 문서는 텔레오퍼레이션 도중 조작을 일시정지하고, 외부 정책/랜덤 액션 생성기에 로봇 제어권을 넘겨준 뒤, 다시 리더암을 안전하게 동기화하여 텔레옵을 재개하는 시스템의 설계 사양서입니다.

---

## 📌 목표 및 배경

- **목적**: 모방 학습(Imitation Learning), DAgger, 데이터 섭동(Data Perturbation), 자율 환경 리셋(Scene Reset) 등의 파이프라인에서 인간 텔레옵과 외부 자율 제어를 매끄럽게 교차 수행.
- **요구사항**:
  1. 텔레옵 중 **단축키(예: `p`)로 즉시 일시정지**하고 로봇 제어권을 해제하여 외부 스크립트가 로봇을 조작할 수 있도록 함.
  2. 외부 조작 중에는 **리더암이 가만히 있거나 힘을 빼고 대기(Relax/Hold)**하여 불필요한 충돌이나 부하 방지.
  3. 외부 작업 완료 후 **재개(예: `r`) 신호 시, 5.0초 동안 부드럽게 S-Curve Auto-Homing**하여 새로운 로봇 자세로 리더암을 안전하게 맞춘 뒤 텔레옵 재개.

---

## 🔄 상태 머신 (State Machine) 설계

```
               [ 단축키 'p' (Pause) ]
 [ 1. TELEOP_ACTIVE ] ───────────────────────────> [ 2. PAUSED_FOR_EXTERNAL ]
   - 리더암 -> 로봇 실시간 제어                       - 리더암: 토크 감발 및 안전 대기 (Relax)
   - 100 Hz 명령 스트림 송출                          - 로봇 제어권 스트림 해제 (Release)
          ▲                                           - 외부 스크립트가 자유롭게 로봇 제어
          │                                                    │
          │                                                    │ [ 단축키 'r' (Resume) ]
          │                                                    ▼
          └─────────────────────────────────────── [ 3. AUTO_HOMING_RESUME ]
                 [ 5.0초 호밍 완료 및 오차 < 0.05 rad ]     - 로봇 현재 위치 q_robot 측정 및 유지
                                                           - 리더암이 5.0초 S-Curve로 q_robot 추종
                                                           - 호밍 완료 시 텔레옵 모드로 자동 전환
```

---

## 🛠️ 세부 동작 시나리오 (Option B 채택)

### 1단계: 일시정지 트리거 (`p` 키 입력 시)
1. **로봇 명령 스트림 취소/일시정지**: 
   - `arm_stream.cancel()` 및 `mobility_stream.cancel()`을 호출하여 로봇의 제어 우선순위(Priority)를 비웁니다.
   - 로봇 Control Manager는 유지되므로, 외부 스크립트가 동일한 Control Manager를 통해 명령을 즉시 전송할 수 있습니다.
2. **리더암 힘 빼기 (Torque Relax/Standby)**:
   - 리더암의 토크를 1.0초간 부드럽게 낮추거나 중력보상/홀드 상태로 전환하여 사용자가 리더암을 거치대에 편하게 내려놓을 수 있게 합니다.
3. **터미널 상태 표시**:
   ```text
   [PAUSED] Teleop paused. Robot control released for external script.
            Leader arm is in relaxed standby mode.
            -> Press 'r' to Auto-Home & Resume Teleoperation.
   ```

### 2단계: 외부 스크립트 제어 구간
- 외부 프로세스(랜덤 액션 생성기, 강화학습 정책, 환경 리셋 스크립트 등)가 로봇에게 자유롭게 모션 명령을 보냅니다.
- 리더암은 로봇의 움직임과 간섭 없이 대기 상태를 유지합니다.

### 3단계: 텔레옵 재개 트리거 (`r` 키 입력 시)
1. **로봇 현재 관절 각도 측정**:
   - 외부 스크립트의 조작이 끝난 로봇의 현재 관절 각도 $q_{\text{robot}}$을 실시간 수신하여 목표값으로 설정합니다.
   - 로봇이 움직이지 않도록 현재 자세 유지(Hold) 명령을 전송합니다.
2. **리더암 5.0초 S-Curve Auto-Homing**:
   - 5차 다항식(Quintic polynomial) 보간을 적용하여 리더암을 현재 위치 $q_{\text{leader}}$에서 로봇 위치 $q_{\text{robot}}$으로 5.0초에 걸쳐 매우 부드럽고 안전하게 이동시킵니다.
   - 호밍 완료 조건: 리더암과 로봇의 최대 관절 오차가 `0.05 rad (약 2.8°)` 이내로 수렴할 때까지 검증.
3. **텔레옵 명령 스트림 재연결**:
   - 오차 수렴 즉시 로봇 제어 명령 스트림을 다시 생성하고, 리더암의 움직임이 로봇에 1:1 반영되는 `TELEOP_ACTIVE` 모드로 복귀합니다.
   - 작업자에게 준비 완료 알림 출력:
     ```text
     [RESUMED] Leader arm successfully homed to new robot pose (5.0s).
               Teleoperation is now ACTIVE!
     ```

---

## 📊 데이터 로거(`record_episodes.py`) 연동 방안

데이터를 수집할 때 한 에피소드 내에서 인간 조작 구간과 외부 조작 구간을 자동으로 구분할 수 있도록 필드를 추가할 수 있습니다:
- `is_teleop`: 인간이 리더암으로 조작한 구간 (`bool` 배열)
- `control_mode`: `0` = 인간 텔레옵, `1` = 일시정지/외부 액션, `2` = 호밍 전환 구간

---

## 🚀 향후 구현 시 적용 대상 파일

1. **[`teleop_autohome.py`](file:///mnt/ssd/rby1-sdk/examples/python/leader_arm_mobile_record_replay/teleop_autohome.py)**:
   - `p` / `r` 키보드 리스너 및 상태 머신(`State: ACTIVE / PAUSED / HOMING`) 추가
   - `pause_teleop()`: 리더암 토크 감발 및 로봇 스트림 cancel
   - `resume_teleop()`: 5.0초 S-Curve 호밍 루틴 재실행 후 스트림 재개
2. **외부 액션 예제 스크립트 (예: `random_action_generator.py`)**:
   - 일시정지 상태에서 로봇에 임의의 궤적을 전송하는 예시 스크립트 작성
