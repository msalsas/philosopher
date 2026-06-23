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

Run on the board (needs root for GPIO):
    sudo /home/manolo/philosopher/philosopher-toy/.venv/bin/python \
        /home/manolo/philosopher/philosopher-toy/scripts/servo_test.py PA6

The servo should step: center -> one end -> other end -> center.
"""
from __future__ import annotations

import sys
import time

import OPi.GPIO as GPIO

PIN = sys.argv[1] if len(sys.argv) > 1 else "PA6"
PERIOD = 0.02  # 20 ms frame = 50 Hz


def hold(pulse_ms: float, seconds: float) -> None:
    """Bit-bang a fixed-width pulse train for `seconds` to hold a servo angle."""
    pulse = pulse_ms / 1000.0
    gap = max(0.0, PERIOD - pulse)
    end = time.time() + seconds
    while time.time() < end:
        GPIO.output(PIN, 1)
        time.sleep(pulse)
        GPIO.output(PIN, 0)
        time.sleep(gap)


def main() -> None:
    GPIO.setmode(GPIO.SUNXI)
    GPIO.setwarnings(False)
    GPIO.setup(PIN, GPIO.OUT)
    print(f"[servo_test] bit-banging {PIN} at ~50 Hz — watch the horn move")
    try:
        # ~1.0 ms = -90deg, 1.5 ms = center, 2.0 ms = +90deg
        for label, pulse in (("center", 1.5), ("-90", 1.0),
                            ("+90", 2.0), ("center", 1.5)):
            print(f"  -> {label} ({pulse} ms pulse)")
            hold(pulse, 1.5)
    finally:
        GPIO.output(PIN, 0)
        GPIO.cleanup()
        print("[servo_test] done, GPIO cleaned up")


if __name__ == "__main__":
    main()
