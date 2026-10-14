"""Generate bounded demonstration traffic; does not measure model quality."""

import argparse
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.requests < 1 or args.interval < 0:
        parser.error("requests must be positive and interval must be nonnegative")
    with httpx.Client(base_url=args.url, timeout=30, trust_env=False) as client:
        for index in range(args.requests):
            invalid = index % 10 == 0
            text = "   " if invalid else "Clinical study of coronary artery disease."
            response = client.post("/predict", json={"text": text})
            expected = 422 if invalid else 200
            if response.status_code != expected:
                raise RuntimeError(f"Expected {expected}, got {response.status_code}")
            time.sleep(args.interval)
    print(f"Completed {args.requests} requests, including intentional 422 responses.")


if __name__ == "__main__":
    main()
