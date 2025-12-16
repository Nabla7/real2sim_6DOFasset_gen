#!/bin/bash
# Test script to verify all services are running correctly

set -e

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
SAM3_URL="${SAM3_URL:-http://localhost:8091}"
SAM3D_URL="${SAM3D_URL:-http://localhost:8092}"
FP_URL="${FP_URL:-http://localhost:8093}"

echo "=== Testing Spatial Memory Service Deployment ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_service() {
    local name=$1
    local url=$2

    printf "Testing %-20s ... " "$name"

    if response=$(curl -sf "$url/health" 2>&1); then
        if echo "$response" | grep -q "ok"; then
            printf "${GREEN}✓ OK${NC}\n"
            return 0
        else
            printf "${YELLOW}⚠ WARN${NC} (unexpected response)\n"
            echo "  Response: $response"
            return 1
        fi
    else
        printf "${RED}✗ FAIL${NC}\n"
        echo "  Error: $response"
        return 1
    fi
}

# Test all services
echo "Health Checks:"
test_service "Gateway" "$GATEWAY_URL"
test_service "SAM3" "$SAM3_URL"
test_service "SAM3D" "$SAM3D_URL"
test_service "FoundationPose" "$FP_URL"

echo ""
echo "=== Summary ==="

# Check if gateway is accessible
if curl -sf "$GATEWAY_URL/health" > /dev/null 2>&1; then
    echo "${GREEN}✓ Service is operational${NC}"
    echo ""
    echo "API endpoint: $GATEWAY_URL"
    echo "Example usage:"
    echo "  curl -X POST $GATEWAY_URL/process -H 'Content-Type: application/json' -d @request.json"
    exit 0
else
    echo "${RED}✗ Service is not operational${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if container is running: docker ps"
    echo "  2. View logs: docker-compose logs -f"
    echo "  3. Check health endpoints individually"
    exit 1
fi

