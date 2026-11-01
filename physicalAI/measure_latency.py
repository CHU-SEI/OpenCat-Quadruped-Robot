#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Measure how fast a PC can talk to Bittle over the serial port.

Phase 1 of the Physical AI plan. Results decide the control rate of the RL policy.

Measures
  1. 'v' round trip: request IMU (pitch, roll) and wait for the reply
  2. 'j' round trip: request all joint angles and wait for the reply
  3. 'L' round trip: send 16 joint angles as one binary packet, wait for the board's echo
  4. closed loop:    'L' then 'v' at a target rate, reports achieved rate and jitter

Safety
  By default the robot does not move: 'L' re-sends the angles read back with 'j'
  (they are not clamped, so a resting posture such as 73 / -56 is kept as it is).
  Use --amp (degrees) to add a small sine wave on the leg joints. Hold the robot
  in the air or on a stand for the first runs, and keep a hand near the power switch.

Usage
  python measure_latency.py COM5
  python measure_latency.py COM5 --amp 5 --hz 50 --seconds 10
Needs: pip install pyserial
"""
import argparse
import math
import re
import statistics
import struct
import sys
import time

import serial

BAUD = 115200
DOF = 16
LEG_JOINTS = range(8, 16)  # Bittle: 0 head, 1-7 unused/neck, 8-15 legs
MAX_ANGLE = 125  # int8 limit of the 'L' packet (the firmware range is +-125), degrees
MAX_AMP = 20  # largest --amp accepted, degrees
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def open_port(name):
    ser = serial.Serial(name, BAUD, timeout=0.05)
    # opening the port resets the board; wait for the boot messages to end with Ready!
    deadline = time.time() + 15
    buf = ""
    while time.time() < deadline:
        buf += ser.read(256).decode("ISO-8859-1")
        if "Ready!" in buf:
            break
    else:
        print("Warning: 'Ready!' not seen. Continuing anyway. Boot log:\n" + buf)
    time.sleep(0.5)
    ser.reset_input_buffer()
    return ser


def read_line(ser, timeout=1.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        line = ser.readline()
        if line:
            return line.decode("ISO-8859-1").strip()
    return None


def query(ser, cmd, want_numbers, timeout=1.0, skip=0):
    """send an ASCII command, return (numbers, seconds) from a reply line with enough numbers.
    skip: number of matching lines to ignore first ('j' prints a row of joint indices, then the angles)"""
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    ser.write(cmd.encode() + b"\n")
    end = t0 + timeout
    while time.perf_counter() < end:
        line = read_line(ser, end - time.perf_counter())
        if line is None:
            break
        nums = [float(x) for x in NUM.findall(line)]
        if len(nums) >= want_numbers:
            if skip:
                skip -= 1
                continue
            return nums[-want_numbers:], time.perf_counter() - t0
    return None, time.perf_counter() - t0


def send_joints(ser, angles, timeout=0.5):
    """send 'L' and wait for the board's echo ('L'); returns seconds, or None if no echo.
    Waiting matters: the firmware reads everything in its serial buffer as one command, so a
    command written right behind the '~' of 'L' is swallowed (found with --diag)."""
    clamped = [max(-MAX_ANGLE, min(MAX_ANGLE, int(round(a)))) for a in angles]
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    ser.write(b"L" + struct.pack("b" * DOF, *clamped) + b"~")
    end = t0 + timeout
    while time.perf_counter() < end:
        line = read_line(ser, end - time.perf_counter())
        if line == "L":
            return time.perf_counter() - t0
    return None


def summarize(name, samples_s):
    if not samples_s:
        print(f"{name}: no valid samples")
        return
    ms = sorted(s * 1000 for s in samples_s)
    p95 = ms[min(len(ms) - 1, int(len(ms) * 0.95))]
    print(f"{name}: n={len(ms)}  mean={statistics.mean(ms):.1f} ms  median={statistics.median(ms):.1f} ms"
          f"  p95={p95:.1f} ms  max={ms[-1]:.1f} ms  -> {1000 / statistics.mean(ms):.0f} Hz")


def round_trip(ser, cmd, want, n, skip=0):
    ok, bad = [], 0
    for _ in range(n):
        nums, dt = query(ser, cmd, want, skip=skip)
        if nums is None:
            bad += 1
        else:
            ok.append(dt)
    summarize(f"'{cmd}' round trip", ok)
    if bad:
        print(f"  {bad} of {n} requests got no reply")


def raw_after(ser, payload, seconds):
    """send bytes, return the (time, line) pairs the board prints within the next `seconds`"""
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    ser.write(payload)
    out = []
    while time.perf_counter() - t0 < seconds:
        line = ser.readline()
        if line:
            out.append((time.perf_counter() - t0, line.decode("ISO-8859-1").strip()))
    return out


def diagnose(ser, joints):
    """why does 'v' get no reply after 'L'? print exactly what the board sends back"""
    packet = b"L" + struct.pack("b" * DOF, *[max(-MAX_ANGLE, min(MAX_ANGLE, int(round(a)))) for a in joints]) + b"~"
    print("packet:", packet)
    print("-- 'L' alone, listen 0.5 s")
    for t, line in raw_after(ser, packet, 0.5):
        print(f"  {t * 1000:7.1f} ms  {line!r}")
    for gap in (0.0, 0.01, 0.03, 0.1):
        print(f"-- 'L', wait {gap * 1000:.0f} ms, then 'v', listen 0.5 s")
        ser.reset_input_buffer()
        t0 = time.perf_counter()
        ser.write(packet)
        time.sleep(gap)
        ser.write(b"v\n")
        while time.perf_counter() - t0 < gap + 0.5:
            line = ser.readline()
            if line:
                print(f"  {(time.perf_counter() - t0) * 1000:7.1f} ms  {line.decode('ISO-8859-1').strip()!r}")
    print("-- 'v' alone, listen 0.5 s")
    for t, line in raw_after(ser, b"v\n", 0.5):
        print(f"  {t * 1000:7.1f} ms  {line!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", help="serial port, e.g. COM5")
    ap.add_argument("--n", type=int, default=100, help="requests per round-trip test")
    ap.add_argument("--hz", type=float, default=50, help="target rate of the closed loop")
    ap.add_argument("--seconds", type=float, default=10, help="length of the closed loop")
    ap.add_argument("--amp", type=float, default=0, help="sine amplitude on leg joints in degrees (0 = do not move)")
    ap.add_argument("--diag", action="store_true", help="only print the raw replies after 'L' and 'v', then exit")
    args = ap.parse_args()
    if not 0 <= args.amp <= MAX_AMP:
        sys.exit(f"--amp must be between 0 and {MAX_AMP} degrees")

    ser = open_port(args.port)
    try:
        print("--- 1. IMU ('v')")
        nums, _ = query(ser, "v", 2)
        print("sample (pitch, roll):", nums)
        round_trip(ser, "v", 2, args.n)

        print("--- 2. joint angles ('j')")
        joints, _ = query(ser, "j", DOF, skip=1)  # first matching line is the index row 0..15
        if joints is None:
            sys.exit("Could not read joint angles with 'j'. Check the reply format in the serial monitor.")
        print("current angles:", [int(a) for a in joints])
        if [int(a) for a in joints] == list(range(DOF)):
            sys.exit("'j' returned the index row (0..15), not angles. Not sending anything to the robot.")
        if args.diag:
            diagnose(ser, joints)
            return
        round_trip(ser, "j", DOF, args.n, skip=1)

        print("--- 3. 'L' round trip (send joints, wait for the echo)")
        acks = [send_joints(ser, joints) for _ in range(args.n)]
        summarize("'L' round trip", [a for a in acks if a is not None])
        if None in acks:
            print(f"  {acks.count(None)} of {args.n} packets got no echo")

        print(f"--- 4. closed loop at {args.hz:.0f} Hz for {args.seconds:.0f} s (amp={args.amp} deg)")
        period = 1.0 / args.hz
        start = time.perf_counter()
        next_t = start
        ticks, late, missed = [], 0, 0
        last = start
        while time.perf_counter() - start < args.seconds:
            t = time.perf_counter() - start
            target = list(joints)
            if args.amp:
                for j in LEG_JOINTS:
                    target[j] = joints[j] + args.amp * math.sin(2 * math.pi * 0.5 * t)
            if send_joints(ser, target) is None:
                missed += 1
            nums, _ = query(ser, "v", 2, timeout=max(period, 0.2))
            if nums is None:
                missed += 1
            now = time.perf_counter()
            ticks.append(now - last)
            last = now
            next_t += period
            wait = next_t - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                late += 1
        summarize("loop period", ticks)
        print(f"  deadline misses: {late} of {len(ticks)}, missed replies ('L' echo or IMU): {missed}")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
