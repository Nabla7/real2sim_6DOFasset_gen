#!/bin/bash
# Test script for GraspGen service integration

set -e

GATEWAY_URL="http://localhost:8080"
GRASPGEN_URL="http://localhost:8094"

echo "=== Testing GraspGen Service Integration ==="
echo ""

# Test 1: GraspGen service health
echo "1. Testing GraspGen service health..."
curl -s "$GRASPGEN_URL/health" | jq .
echo ""

# Test 2: Gateway health (should include all services)
echo "2. Testing Gateway health..."
curl -s "$GATEWAY_URL/health" | jq .
echo ""

echo "=== Manual Test Instructions ==="
echo ""
echo "To test grasp generation, you need:"
echo "  • RGB image (base64 encoded PNG)"
echo "  • Depth image (base64 encoded float32 array)"
echo "  • Camera intrinsics K"
echo "  • Object label and bounding box"
echo ""
echo "Test grasp-only endpoint:"
echo '  curl -X POST http://localhost:8080/grasp \'
echo '    -H "Content-Type: application/json" \'
echo '    -d @test_grasp_request.json'
echo ""
echo "Test full pipeline with grasps:"
echo '  curl -X POST http://localhost:8080/process \'
echo '    -H "Content-Type: application/json" \'
echo '    -d @test_process_request_with_grasps.json'
echo ""
echo "Example request format in test_grasp_request.json:"
echo '{'
echo '  "image_rgb_b64": "...",,'
echo '  "depth_b64": "...",''  
echo '  "K": [[615, 0, 320], [0, 615, 240], [0, 0, 1]],'
echo '  "label": "bottle",'
echo '  "bbox": [100, 150, 200, 300],'
echo '  "filter_collisions": true,'
echo '  "gripper_type": "robotiq_2f_140"'
echo '}'
echo ""

