"""
Latency benchmark for HaMeR / WiLoR remote inference server.

Sends N requests with a test image and reports:
  - RTT       : total round-trip time (client-measured)
  - infer_ms  : server-side GPU inference (returned by server)
  - network_ms: RTT - infer_ms  (TCP+TLS+upload+download overhead)

Usage:
    python latency_bench.py --url https://xxxx.trycloudflare.com
    python latency_bench.py --url https://... --image /path/to/hand.jpg --n 30
"""

import argparse
import time
import statistics

import cv2
import numpy as np
import requests


def make_test_image(path: str | None, size: int = 256) -> bytes:
    if path:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.resize(img, (size, size))
    else:
        # Synthetic: gray gradient — valid JPEG, no real hand
        img = np.random.randint(60, 200, (size, size, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def bench(url: str, image_bytes: bytes, n: int, is_right: int = 1):
    endpoint = url.rstrip("/") + "/infer"
    session = requests.Session()

    rtts, infers, networks = [], [], []

    print(f"\nSending {n} requests to {endpoint}")
    print(f"Image size : {len(image_bytes)/1024:.1f} KB\n")

    for i in range(n):
        files = {"crop": ("crop.jpg", image_bytes, "image/jpeg")}
        data  = {"is_right": str(is_right)}

        t0 = time.perf_counter()
        try:
            resp = session.post(endpoint, files=files, data=data, timeout=10)
            rtt_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            body = resp.json()
        except Exception as e:
            print(f"  [{i+1:3d}] ERROR: {e}")
            continue

        infer_ms   = body.get("inference_ms", float("nan"))
        network_ms = rtt_ms - infer_ms

        rtts.append(rtt_ms)
        infers.append(infer_ms)
        networks.append(network_ms)

        print(f"  [{i+1:3d}]  RTT={rtt_ms:6.1f}ms  infer={infer_ms:6.1f}ms  network={network_ms:6.1f}ms")

    if not rtts:
        print("No successful requests.")
        return

    def stats(vals, label):
        print(f"\n{label}")
        print(f"  mean ± std : {statistics.mean(vals):.1f} ± {statistics.stdev(vals) if len(vals)>1 else 0:.1f} ms")
        print(f"  min / max  : {min(vals):.1f} / {max(vals):.1f} ms")
        print(f"  median     : {statistics.median(vals):.1f} ms")

    print("\n" + "="*50)
    stats(rtts,     "RTT (total round-trip)")
    stats(infers,   "Server inference (GPU)")
    stats(networks, "Network overhead (RTT - infer)")
    print("="*50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",     required=True, help="Server URL (cloudflare tunnel)")
    parser.add_argument("--image",   default=None,  help="Path to test image (default: synthetic)")
    parser.add_argument("--n",       type=int, default=20, help="Number of requests (default: 20)")
    parser.add_argument("--size",    type=int, default=256, help="Crop size in pixels (default: 256)")
    parser.add_argument("--left",    action="store_true",  help="Send as left hand")
    args = parser.parse_args()

    image_bytes = make_test_image(args.image, size=args.size)
    bench(args.url, image_bytes, n=args.n, is_right=0 if args.left else 1)


if __name__ == "__main__":
    main()
