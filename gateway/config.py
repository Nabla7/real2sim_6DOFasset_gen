"""
Gateway configuration for Spatial Memory Service.
"""

import os

# Service URLs (internal Docker network or localhost)
SAM3_URL = os.getenv("SAM3_URL", "http://localhost:8091")
SAM3D_URL = os.getenv("SAM3D_URL", "http://localhost:8092")
FOUNDATIONPOSE_URL = os.getenv("FOUNDATIONPOSE_URL", "http://localhost:8093")

# Gateway settings
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8080"))

# Timeouts (seconds)
SAM3_TIMEOUT = int(os.getenv("SAM3_TIMEOUT", "60"))
SAM3D_TIMEOUT = int(os.getenv("SAM3D_TIMEOUT", "300"))  # Mesh reconstruction is slow
FOUNDATIONPOSE_TIMEOUT = int(os.getenv("FOUNDATIONPOSE_TIMEOUT", "120"))

# Weights paths (mounted volume)
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "/weights")
SAM3_WEIGHTS = os.path.join(WEIGHTS_DIR, "sam3")
SAM3D_WEIGHTS = os.path.join(WEIGHTS_DIR, "sam3d-objects")
FOUNDATIONPOSE_WEIGHTS = os.path.join(WEIGHTS_DIR, "foundationpose")

# Mesh output directory
MESH_OUTPUT_DIR = os.getenv("MESH_OUTPUT_DIR", "/tmp/spatial_memory/meshes")


