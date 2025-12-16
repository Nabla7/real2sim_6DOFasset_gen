#!/bin/bash
# Stop all spatial memory services

LOG_DIR=/workspace/logs

echo "=== Stopping Spatial Memory Services ==="

# Function to stop a service
stop_service() {
  local name=$1
  local pid_file="$LOG_DIR/${name}.pid"
  local port=$2
  
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (PID: $pid)..."
      kill "$pid" 2>/dev/null
      sleep 1
      # Force kill if still running
      if kill -0 "$pid" 2>/dev/null; then
        echo "  Force killing $name..."
        kill -9 "$pid" 2>/dev/null
      fi
      echo "  ✓ $name stopped"
    else
      echo "  $name not running (stale PID file)"
    fi
    rm -f "$pid_file"
  else
    echo "  $name: No PID file found"
  fi
}

# Stop services in reverse order (FoundationPose first, Gateway last)
stop_service "foundationpose" 8093
stop_service "sam3d" 8092
stop_service "sam3" 8091
stop_service "gateway" 8080

# Also kill any remaining python processes on these ports
echo ""
echo "Checking for remaining processes on service ports..."
for port in 8080 8091 8092 8093; do
  pid=$(lsof -ti :$port 2>/dev/null)
  if [ -n "$pid" ]; then
    echo "  Killing process on port $port (PID: $pid)"
    kill -9 $pid 2>/dev/null
  fi
done

echo ""
echo "=== All services stopped ==="
echo ""
echo "To view logs:"
echo "  cat $LOG_DIR/gateway.log"
echo "  cat $LOG_DIR/sam3.log"
echo "  cat $LOG_DIR/sam3d.log"
echo "  cat $LOG_DIR/foundationpose.log"

