"""
Lightweight monitoring script — polls /metrics every 30s and logs to file.
Run alongside the API: python monitoring/monitor.py
"""

import time
import json
import datetime
import urllib.request
import urllib.error
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")
LOG_FILE = os.path.join(os.path.dirname(__file__), "monitor.log")
INTERVAL = int(os.getenv("MONITOR_INTERVAL", "30"))


def fetch_metrics() -> dict:
    url = f"{API_URL}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": str(e)}


def log(data: dict):
    entry = {"timestamp": datetime.datetime.utcnow().isoformat(), **data}
    line = json.dumps(entry)
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def check_health():
    try:
        with urllib.request.urlopen(f"{API_URL}/health", timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


def main():
    print(f"Monitoring {API_URL} every {INTERVAL}s → {LOG_FILE}")
    consecutive_failures = 0
    while True:
        if not check_health():
            consecutive_failures += 1
            log({"alert": "API_DOWN", "consecutive_failures": consecutive_failures})
            if consecutive_failures >= 3:
                print("ALERT: API has been down for 3+ consecutive checks!")
        else:
            consecutive_failures = 0
            metrics = fetch_metrics()
            log(metrics)

            # Threshold alerts
            avg_latency = metrics.get("avg_latency_ms", 0)
            if avg_latency > 500:
                log({"alert": "HIGH_LATENCY", "avg_latency_ms": avg_latency})

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
