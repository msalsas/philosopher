#!/usr/bin/env python3
"""Standalone single-servo bring-up test for the Banana Pi M2 Zero (Allwinner H3).

OPi.GPIO's PWM class is HARDWARE PWM only — PWM(chip, pin, frequency, duty) via
the sysfs pwmchip interface — and the M2 Zero header exposes no hardware PWM
(and PA6 isn't a PWM pin anyway). So we BIT-BANG the ~50 Hz servo signal in
software on a plain GPIO using GPIO.output. It's jittery (Python timing on a
512 MB board), but enough to confirm the servo moves and that the solder joints
on the signal / 5V / GND pins actually conduct.

Wiring (one small servo):
    signal (orange/yellow) -> the chosen GPIO pin (default PA6 = physical pin 7)
    V+     (red, middle)   -> physical pin 2 (5V)
    GND    (brown/black)   -> physical pin 6 (GND)

Run on the board (needs root for GPIO). One or more SUNXI pins:
    sudo .../python scripts/servo_test.py PA6            # one servo
    sudo .../python scripts/servo_test.py PA6 PA7 PA8    # three, ONE AT A TIME

Each pin is swept center -> one end -> other end -> center, sequentially (never
two at once) — matching servos.py's brownout-safe one-servo-at-a-time design.
"""
from __future__ import annotations

import sys
import time

import OPi.GPIO as GPIO

PINS = sys.argv[1:] or ["PA6"]
PERIOD = 0.02  # 20 ms frame = 50 Hz


def hold(pin: str, pulse_ms: float, seconds: float) -> None:
    """Bit-bang a fixed-width pulse train for `seconds` to hold a servo angle."""
    pulse = pulse_ms / 1000.0
    gap = max(0.0, PERIOD - pulse)
    end = time.time() + seconds
    while time.time() < end:
        GPIO.output(pin, 1)
        time.sleep(pulse)
        GPIO.output(pin, 0)
        time.sleep(gap)


def sweep(pin: str) -> None:
    print(f"[servo_test] {pin}: bit-banging ~50 Hz — watch the horn move")
    # ~1.0 ms = -90deg, 1.5 ms = center, 2.0 ms = +90deg
    for label, pulse in (("center", 1.5), ("-90", 1.0), ("+90", 2.0), ("center", 1.5)):
        print(f"  {pin} -> {label} ({pulse} ms pulse)")
        hold(pin, pulse, 1.5)
    GPIO.output(pin, 0)


def main() -> None:
    GPIO.setmode(GPIO.SUNXI)
    GPIO.setwarnings(False)
    for pin in PINS:
        GPIO.setup(pin, GPIO.OUT)
    try:
        for pin in PINS:          # one servo at a time (brownout-safe)
            sweep(pin)
            time.sleep(0.5)
    finally:
        for pin in PINS:
            GPIO.output(pin, 0)
        GPIO.cleanup()
        print("[servo_test] done, GPIO cleaned up")


if __name__ == "__main__":
    main()
