#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Move one leg joint at a time so you can see which joint a firmware index is, and which way it turns.

Why not the serial monitor: 'i 8 60' runs on top of the last skill with the gyro balance on, so the IMU
moves the other legs too. 'L' bypasses that, so only the joint you send moves.

The robot stands (kbalance), then each of joints 8..15 is moved by --delta degrees from the standing
angle, held for a moment, and moved back. Press Enter between joints and write down what you saw
in physicalAI/joint_check.md.

Keep the robot on a flat floor with room around it, the battery connected, a hand near the power switch.
Usage:  python joint_check.py COM4  [--delta 25] [--hold 1.5]
"""
import argparse
import sys
import time

import serial

from measure_latency import DOF, open_port, query, send_joints

PRED = {
    8: "left front shoulder", 9: "right front shoulder", 10: "right back shoulder", 11: "left back shoulder",
    12: "left front knee", 13: "right front knee", 14: "right back knee", 15: "left back knee",
}
# firmware angleLimit (deg) for joints 8..15
LIMITS = {8: (-200, 80), 9: (-200, 80), 10: (-80, 200), 11: (-80, 200),
          12: (-80, 200), 13: (-80, 200), 14: (-80, 200), 15: (-80, 200)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port")
    ap.add_argument("--delta", type=float, default=25, help="degrees added to the joint (default 25, max 40)")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds to hold the moved joint")
    args = ap.parse_args()
    if not 0 < abs(args.delta) <= 40:
        sys.exit("--delta must be between -40 and 40 degrees, not 0")

    ser = open_port(args.port)
    try:
        ser.write(b"kbalance\n")
        print("standing up (kbalance) ...")
        time.sleep(3.0)
        ser.reset_input_buffer()

        base, _ = query(ser, "j", DOF, skip=1)
        if base is None:
            sys.exit("could not read the joint angles with 'j'")
        base = [int(a) for a in base]
        print("standing angles:", base)

        for idx in range(8, 16):
            lo, hi = LIMITS[idx]
            target = base[idx] + args.delta
            if not lo < target < hi:
                target = base[idx] - args.delta  # the other way, so it stays inside the firmware limit
            moved = list(base)
            moved[idx] = int(round(target))
            input(f"\nEnter to move joint {idx} from {base[idx]} to {moved[idx]} deg  (guess: {PRED[idx]})  ")
            if send_joints(ser, moved) is None:
                print("  no echo from the board, skipping")
                continue
            time.sleep(args.hold)
            send_joints(ser, base)
            time.sleep(0.8)
            print(f"  joint {idx} moved back. Which leg and joint moved? Which way (front/back)? Note: "
                  f"{'+' if moved[idx] > base[idx] else '-'}{abs(moved[idx] - base[idx])} deg was sent.")
        print("\ndone. standing again.")
        ser.write(b"kbalance\n")
        time.sleep(1.0)
    finally:
        ser.close()


if __name__ == "__main__":
    main()
