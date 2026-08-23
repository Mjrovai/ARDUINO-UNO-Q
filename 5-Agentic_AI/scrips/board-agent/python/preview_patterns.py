#!/usr/bin/env python3
"""
preview_patterns.py — render the sketch's LED matrix patterns in the terminal.

Keep these coordinates identical to the ones in sketch.ino. If you nudge a
pixel in one place, nudge it in the other.

    python3 preview_patterns.py            # all patterns
    python3 preview_patterns.py check x    # only the ones you name
    python3 preview_patterns.py 7          # a digit
"""

import sys

ROWS, COLS = 8, 13


def blank():
    return [[0] * COLS for _ in range(ROWS)]


def render(name, frame):
    print(f"\n  {name}")
    print("     " + "".join(str(c % 10) for c in range(COLS)))
    for r, row in enumerate(frame):
        print(f"   {r} " + "".join("#" if v else "." for v in row))


def check():
    f = blank()
    for r, c in [(5, 2), (6, 3), (7, 4), (6, 5), (5, 6),
                 (4, 7), (3, 8), (2, 9), (1, 10)]:
        f[r][c] = 1
    return f


def cross():
    f = blank()
    for i in range(ROWS):
        f[i][2 + i] = 1
        f[i][10 - i] = 1
    return f


def smiley():
    f = blank()
    for r, c in [(1, 3), (1, 4), (1, 8), (1, 9),
                 (5, 2), (6, 3), (6, 4), (6, 5), (6, 6),
                 (6, 7), (6, 8), (6, 9), (5, 10)]:
        f[r][c] = 1
    return f


DIGIT_FONT = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b110, 0b010, 0b010, 0b111],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b010, 0b010, 0b010],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
]

ROW_OFFSET, COL_OFFSET = 1, 5


def digit(d):
    f = blank()
    for row in range(5):
        for col in range(3):
            if (DIGIT_FONT[d][row] >> (2 - col)) & 1:
                f[row + ROW_OFFSET][col + COL_OFFSET] = 1
    return f


NAMED = {"off": blank, "check": check, "x": cross, "smiley": smiley}


def main():
    wanted = sys.argv[1:] or list(NAMED) + [str(d) for d in range(10)]
    for name in wanted:
        if name in NAMED:
            render(name, NAMED[name]())
        elif name.isdigit() and len(name) == 1:
            render(f"digit {name}", digit(int(name)))
        else:
            print(f"\n  unknown pattern: {name}", file=sys.stderr)
    print()


if __name__ == "__main__":
    main()
