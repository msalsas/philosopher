"""Servo controller — head on hardware PWM (smooth, jitter-free).

The head servo (BCM12 = PWM0 ch0, enabled by `dtoverlay=pwm,pin=12,func=4` in
config.txt) is driven via the sysfs hardware-PWM interface: stable pulses, no
software-PWM tremor. The signal is released (duty 0) at rest so a servo draws no
holding current between moves — this rig's 5V rail can't sustain a held servo
plus the audio amp without browning out. Arms would need their own PWM channels
and are treated as no-ops here.
"""
from __future__ import annotations

import asyncio
import os
import random
import subprocess

_PWMCHIP = "/sys/class/pwm/pwmchip0"
_PWM0 = _PWMCHIP + "/pwm0"
_PERIOD_NS = 20_000_000          # 20 ms = 50 Hz
_CENTER_NS = 1_500_000           # 1.5 ms
_MIN_NS, _MAX_NS = 500_000, 2_500_000   # 0.5–2.5 ms = -90°..+90°
_NS_PER_DEG = 1_000_000 / 90


def _pin_env(var: str, default: int) -> int | None:
    """BCM pin from env; 0/empty/negative -> None (servo disabled, gestures no-op)."""
    raw = os.getenv(var)
    if raw is None or raw.strip() == "":
        return default
    try:
        pin = int(raw)
    except ValueError:
        return default
    return pin if pin > 0 else None


class ServoController:
    """Controls the head servo (hardware PWM). Arms are no-ops on this rig."""

    MAX_PAN = 10          # max head pan (deg) for face tracking
    PAN_DEADZONE = 5      # ignore gaze changes below this (anti-jitter)
    IDLE_DRIFT = 10       # idle "look around" amplitude (deg)
    IDLE_MIN_INTERVAL = 12.0
    IDLE_MAX_INTERVAL = 30.0

    def __init__(self, head=12, left=13, right=18, mock=False) -> None:
        candidates = {
            "head": _pin_env("PHILOSOPHER_SERVO_HEAD_PIN", head),
            "left_arm": _pin_env("PHILOSOPHER_SERVO_LEFT_ARM_PIN", left),
            "right_arm": _pin_env("PHILOSOPHER_SERVO_RIGHT_ARM_PIN", right),
        }
        self.pins = {name: pin for name, pin in candidates.items() if pin is not None}
        self.mock = mock
        self.pos = {k: 0 for k in self.pins}
        self.head_base = 0
        self._last_ns = _CENTER_NS   # for smooth ramping of the head
        self._hw = False
        self.moving = False
        self.queue = asyncio.Queue()
        self._worker_started = False

    async def init(self):
        if not self.mock and "head" in self.pins:
            # Export the HW-PWM channel and make it writable by this (non-root)
            # process; passwordless sudo is used once here. Leave it released.
            try:
                subprocess.run(
                    ["sudo", "-n", "sh", "-c",
                     f"[ -d {_PWM0} ] || (echo 0 > {_PWMCHIP}/export; sleep 0.3); "
                     f"chmod -R a+rw {_PWM0} 2>/dev/null; "
                     f"echo {_PERIOD_NS} > {_PWM0}/period; "
                     f"echo 0 > {_PWM0}/duty_cycle; "
                     f"echo 1 > {_PWM0}/enable"],
                    timeout=6, check=False,
                )
                self._hw = os.path.exists(_PWM0)
            except Exception as e:  # noqa: BLE001
                print(f"[WARNING] Hardware PWM init failed: {e}. Head servo off.")
                self._hw = False
        print("[Servo] head hardware-PWM: " + ("ready" if self._hw else "off"))
        if not self._worker_started:
            asyncio.create_task(self._movement_worker())
            self._worker_started = True
        return self

    def _write(self, attr: str, value) -> None:
        try:
            with open(f"{_PWM0}/{attr}", "w") as f:
                f.write(str(int(value)))
        except OSError:
            pass

    @staticmethod
    def _angle_to_ns(angle: float) -> int:
        return int(max(_MIN_NS, min(_MAX_NS, _CENTER_NS + angle * _NS_PER_DEG)))

    async def move(self, name, angle, duration=0.5):
        if name not in self.pins:
            return
        self.pos[name] = max(-90, min(90, angle))
        await self.queue.put((name, angle, duration))
        if self.mock:
            print(f"[Servo] {name} -> {angle}")
            await asyncio.sleep(duration)
            return
        while self.moving:
            await asyncio.sleep(0.01)

    async def _movement_worker(self):
        while True:
            name, angle, duration = await self.queue.get()
            if self.mock:
                self.moving = True
                await asyncio.sleep(0.3)
                self.moving = False
                continue
            self.moving = True
            try:
                # Only the head is on hardware PWM; arm pins have no channel here.
                if self._hw and name == "head":
                    target = self._angle_to_ns(angle)
                    start = self._last_ns
                    # Ramp in ~1.3°/step so the head pans smoothly, then release
                    # (duty 0) so it draws nothing at rest.
                    steps = max(1, int(abs(target - start) / 15_000))
                    for k in range(1, steps + 1):
                        self._write("duty_cycle", start + (target - start) * k / steps)
                        await asyncio.sleep(0.02)
                    self._last_ns = target
                    await asyncio.sleep(max(0.0, duration - 0.2))
                    self._write("duty_cycle", 0)   # release: no holding current
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] Servo movement failed: {e}")
            await asyncio.sleep(0.3)
            self.moving = False

    async def look(self, offset_x: float):
        """Point the head at a face. offset_x in [-1, 1] (0 = centered)."""
        target = max(-self.MAX_PAN, min(self.MAX_PAN, round(offset_x * self.MAX_PAN)))
        if abs(target - self.head_base) < self.PAN_DEADZONE:
            return
        self.head_base = target
        await self.move("head", target, 0.8)

    async def animate(self, emotion):
        # Emotion gestures are arm poses; arms are no-ops on this rig, so this
        # simply does nothing (kept so the WS `servo` message stays handled).
        poses = {
            "happy": [("right_arm", 45)],
            "sad": [("left_arm", -20)],
            "surprised": [("right_arm", 30), ("left_arm", 30)],
            "angry": [("right_arm", 20)],
            "fear": [("left_arm", -15)],
            "neutral": [],
        }
        pose = poses.get(emotion, [])
        for servo, angle in pose:
            await self.move(servo, angle, 0.4)
        if pose:
            await asyncio.sleep(0.4)
            for servo, _ in pose:
                await self.move(servo, 0, 0.4)

    async def react(self, emotion):
        """Brief acknowledgement of an emotion change (head shake on negatives)."""
        if emotion == "happy":
            await self._arm_bob("right_arm", 30)
        elif emotion == "surprised":
            await self._arm_bob("right_arm", 40)
        elif emotion == "fear":
            await self._arm_bob("left_arm", -12)
        elif emotion == "sad":
            await self._head_shake(amplitude=8, dur=0.35)
        elif emotion == "angry":
            await self._head_shake(amplitude=12, dur=0.18)

    async def _arm_bob(self, arm, angle):
        await self.move(arm, angle, 0.3)
        await self.move(arm, 0, 0.3)

    async def _head_shake(self, amplitude, dur, cycles=2):
        for _ in range(cycles):
            await self.move("head", self.head_base + amplitude, dur)
            await self.move("head", self.head_base - amplitude, dur)
        await self.move("head", self.head_base, dur)

    async def idle_tick(self):
        """One idle 'look around': a clear wide swing, alternating sides."""
        self._idle_dir = -getattr(self, "_idle_dir", -1)
        drift = self._idle_dir * random.randint(self.IDLE_DRIFT // 2, self.IDLE_DRIFT)
        await self.move("head", self.head_base + drift, 0.8)

    async def idle_loop(self):
        """Background loop adding subtle life while idle. Runs until cancelled."""
        while True:
            await asyncio.sleep(random.uniform(self.IDLE_MIN_INTERVAL, self.IDLE_MAX_INTERVAL))
            if not self.moving:
                await self.idle_tick()

    def close(self):
        if self._hw:
            self._write("duty_cycle", 0)
            self._write("enable", 0)
