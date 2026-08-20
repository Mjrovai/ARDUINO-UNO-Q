#!/usr/bin/env python3
import time
from pathlib import Path

# Based on your output: mapss_thermal → temp1_input
TEMP_PATH = Path("/sys/class/hwmon/hwmon0/temp1_input")

def read_temp_c():
    raw = int(TEMP_PATH.read_text().strip())
    return raw / 1000.0

if __name__ == "__main__":
    try:
        while True:
            t = read_temp_c()
            print(f"{t:.1f} °C")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
