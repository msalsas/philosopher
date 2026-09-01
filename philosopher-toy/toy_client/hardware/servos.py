"""Servo controller for head and arm movements."""
from __future__ import annotations

import asyncio
import os
import random


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
    """Controls servos for the toy's head and arms."""

    # Max head pan (degrees) for face tracking, and the deadzone (degrees) below
    # which we ignore a change to avoid jitter.
    MAX_PAN = 60
    PAN_DEADZONE = 5
    # Idle "breathing": tiny head drift around the base so the toy isn't a
    # frozen statue between events. Small amplitude, slow & randomized cadence.
    IDLE_DRIFT = 3
    IDLE_MIN_INTERVAL = 3.0
    IDLE_MAX_INTERVAL = 7.0

    def __init__(self, head=12, left=13, right=18, mock=False) -> None:
        # BCM pins, overridable per servo via env; a None pin is dropped.
        candidates = {
            "head": _pin_env("PHILOSOPHER_SERVO_HEAD_PIN", head),
            "left_arm": _pin_env("PHILOSOPHER_SERVO_LEFT_ARM_PIN", left),
            "right_arm": _pin_env("PHILOSOPHER_SERVO_RIGHT_ARM_PIN", right),
        }
        self.pins = {name: pin for name, pin in candidates.items() if pin is not None}
        self.mock = mock
        self.pos = {k: 0 for k in self.pins}
        # Base head angle the toy "looks" at (set by face tracking); emotion
        # gestures are applied as deltas on top of this.
        self.head_base = 0
        self.moving = False
        self.queue = asyncio.Queue()
        self._worker_started = False

    async def init(self):
        if not self.mock:
            try:
                # RPi.GPIO on the Pi Zero WH (provided by the rpi-lgpio drop-in
                # on modern Raspberry Pi OS). Same API as the old OPi.GPIO path.
                import RPi.GPIO as GPIO
                GPIO.setwarnings(False)
                GPIO.setmode(GPIO.BCM)
                for pin in self.pins.values():
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)
                self._pwm = {}
                for name, pin in self.pins.items():
                    pwm = GPIO.PWM(pin, 50)
                    pwm.start(7.5)
                    self._pwm[name] = pwm
            except Exception as e:
                print(f"[WARNING] GPIO init failed: {e}. Entering mock mode.")
                self.mock = True
        print("[WARNING] Servos share MicroUSB power. Only one moves at a time.")
        if not self._worker_started:
            asyncio.create_task(self._movement_worker())
            self._worker_started = True
        return self

    async def move(self, name, angle, duration=0.5):
        if name not in self.pins:
            return
        self.pos[name] = max(-90, min(90, angle))
        await self.queue.put((name, angle, duration))
        if self.mock:
            print(f"[Servo] {name} -> {angle}")
            await asyncio.sleep(duration)
            return
        # In non-mock mode, wait for worker to process
        while self.moving:
            await asyncio.sleep(0.01)

    async def _movement_worker(self):
        while True:
            name, angle, duration = await self.queue.get()
            if self.mock:
                # Mock mode handles movement synchronously in move()
                self.moving = True
                await asyncio.sleep(0.3)
                self.moving = False
                continue
            self.moving = True
            try:
                duty = 7.5 + (angle / 90.0) * 5.0
                duty = max(2.5, min(12.5, duty))
                if hasattr(self, "_pwm") and name in self._pwm:
                    self._pwm[name].ChangeDutyCycle(duty)
                    await asyncio.sleep(duration)
                    self._pwm[name].ChangeDutyCycle(0)
            except Exception as e:
                print(f"[ERROR] Servo movement failed: {e}")
            await asyncio.sleep(0.3)
            self.moving = False

    async def look(self, offset_x: float):
        """Point the head at a face. offset_x in [-1, 1] (0 = centered).

        Sets the head base position used as the reference for emotion gestures.
        Small changes within the deadzone are ignored to avoid jitter.
        """
        target = max(-self.MAX_PAN, min(self.MAX_PAN, round(offset_x * self.MAX_PAN)))
        if abs(target - self.head_base) < self.PAN_DEADZONE:
            return
        self.head_base = target
        await self.move("head", target, 0.3)

    async def animate(self, emotion):
        # Conversational emotion gesture. Expressed with the ARMS only: the head
        # stays on its gaze base (it pans, so it's reserved for looking at the
        # face — turning it away to "act" an emotion would break eye contact).
        # Each gesture raises into the pose, holds, then returns the arms to
        # rest, so arms never get "stuck" in a pose between gestures.
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
            await asyncio.sleep(0.4)  # hold the expression briefly
            for servo, _ in pose:
                await self.move(servo, 0, 0.4)  # back to rest

    async def react(self, emotion):
        """Small, brief acknowledgement of an emotion change.

        The head only pans (horizontal), so the channel depends on valence:
        - positive (happy/surprised): an ARM gesture — a horizontal head wiggle
          would read as shaking 'no' and contradict the positive feeling;
        - negative (sad/angry): a head SHAKE ('no') — an apt, legible cue on a
          pan-only head (slow = melancholy, sharper = disapproval);
        - fear: an arm recoil.
        The head always returns to its tracked gaze base afterward.
        """
        if emotion == "happy":
            await self._arm_bob("right_arm", 30)
        elif emotion == "surprised":
            await self._arm_bob("right_arm", 40)
        elif emotion == "fear":
            await self._arm_bob("left_arm", -12)
        elif emotion == "sad":
            await self._head_shake(amplitude=8, dur=0.35)    # slow -> melancholy
        elif emotion == "angry":
            await self._head_shake(amplitude=12, dur=0.18)   # sharper -> disapproval
        # neutral / unknown -> no reaction

    async def _arm_bob(self, arm, angle):
        await self.move(arm, angle, 0.3)
        await self.move(arm, 0, 0.3)  # back to rest

    async def _head_shake(self, amplitude, dur, cycles=2):
        for _ in range(cycles):
            await self.move("head", self.head_base + amplitude, dur)
            await self.move("head", self.head_base - amplitude, dur)
        await self.move("head", self.head_base, dur)  # settle back on the gaze base

    async def idle_tick(self):
        """One idle 'breath': nudge the head a few degrees around its base."""
        drift = random.randint(-self.IDLE_DRIFT, self.IDLE_DRIFT)
        await self.move("head", self.head_base + drift, 0.4)

    async def idle_loop(self):
        """Background loop adding subtle life while idle. Runs until cancelled."""
        while True:
            await asyncio.sleep(random.uniform(self.IDLE_MIN_INTERVAL, self.IDLE_MAX_INTERVAL))
            if not self.moving:
                await self.idle_tick()

    def close(self):
        if not self.mock:
            try:
                import RPi.GPIO as GPIO
                if hasattr(self, "_pwm"):
                    for pwm in self._pwm.values():
                        pwm.stop()
                GPIO.cleanup()
            except Exception:
                pass
