#!/usr/bin/env python3
"""
Quick Capture Example

Simple script showing basic usage of the Scene Capture Service API.
"""

import sys
import requests

# Configuration
BASE_URL = "http://localhost:8095"
NUM_CAPTURES = 1000
INTERVAL = 0.6  # seconds between captures

print("Scene Capture Quick Start")
print("=" * 50)

try:
    # Step 1: Create a session
    print("\n1. Creating capture session...")
    response = requests.post(
        f"{BASE_URL}/sessions/create",
        json={"name": "Quick Capture Demo"}
    )
    response.raise_for_status()
    session_data = response.json()
    session_id = session_data["session_id"]
    print(f"   Session ID: {session_id}")
    print(f"   Output: {session_data['output_dir']}")

    # Step 2: Auto-capture frames
    print(f"\n2. Capturing {NUM_CAPTURES} frames...")
    print(f"   This will take approximately {(NUM_CAPTURES * INTERVAL) / 60:.1f} minutes...")
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/auto",
        json={
            "num_captures": NUM_CAPTURES,
            "interval": INTERVAL
        },
        timeout=NUM_CAPTURES * INTERVAL + 60  # Add 60s buffer
    )

    if response.status_code != 200:
        print(f"   Error: HTTP {response.status_code}")
        print(f"   Response: {response.text}")
        sys.exit(1)

    capture_data = response.json()
    print(f"   Captured: {capture_data['total_captured']} frames")

    # Step 3: Check status
    print("\n3. Checking session status...")
    response = requests.get(f"{BASE_URL}/sessions/{session_id}/status")
    response.raise_for_status()
    status = response.json()
    print(f"   Status: {status['status']}")
    print(f"   Total captures: {status['capture_count']}")

    # Step 4: Finalize session
    print("\n4. Finalizing session...")
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/finalize")
    response.raise_for_status()
    final_data = response.json()
    print(f"   {final_data['message']}")

    print("\n" + "=" * 50)
    print("✓ Done! Check the output directory for captured images.")
    print(f"  Output: {final_data['output_dir']}")

except requests.exceptions.RequestException as e:
    print(f"\n✗ Error: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status: {e.response.status_code}")
        print(f"Response text: {e.response.text}")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Unexpected error: {e}")
    sys.exit(1)
