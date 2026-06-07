#!/bin/bash

# Default values
KILL_SIGNAL="KILL"   # Changed default to force kill (SIGKILL)
SHOW_ONLY=false      # Default to actually killing processes

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -g, --graceful      Use SIGTERM instead of SIGKILL for graceful termination"
    echo "  -s, --show-only     Only show processes, don't kill them"
    echo "  -h, --help          Show this help message"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--graceful)
            KILL_SIGNAL="TERM"
            shift
            ;;
        -s|--show-only)
            SHOW_ONLY=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Find all OneSim processes
echo "Finding OneSim processes..."
MASTER_PROCESSES=$(pgrep -f "python src/main.py.*--mode master" || echo "")
WORKER_PROCESSES=$(pgrep -f "python src/main.py.*--mode worker" || echo "")

# Show all processes
if [ -n "$MASTER_PROCESSES" ]; then
    echo "Master processes:"
    ps -f -p $MASTER_PROCESSES
else
    echo "No master processes found."
fi

if [ -n "$WORKER_PROCESSES" ]; then
    echo "Worker processes:"
    ps -f -p $WORKER_PROCESSES
else
    echo "No worker processes found."
fi

# Count processes
MASTER_COUNT=$(echo $MASTER_PROCESSES | wc -w)
WORKER_COUNT=$(echo $WORKER_PROCESSES | wc -w)
TOTAL_COUNT=$((MASTER_COUNT + WORKER_COUNT))

# If showing only, exit here
if $SHOW_ONLY; then
    echo "Found $MASTER_COUNT master and $WORKER_COUNT worker processes."
    echo "Total: $TOTAL_COUNT processes"
    echo "Run without --show-only to kill these processes."
    exit 0
fi

# If no processes found, exit
if [ $TOTAL_COUNT -eq 0 ]; then
    echo "No OneSim processes found to kill."
    exit 0
fi

# Confirm before killing
read -p "Kill $TOTAL_COUNT OneSim processes with SIG$KILL_SIGNAL? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Kill processes
KILLED_COUNT=0

if [ -n "$WORKER_PROCESSES" ]; then
    echo "Killing worker processes..."
    for PID in $WORKER_PROCESSES; do
        echo "Killing worker process $PID with SIG$KILL_SIGNAL"
        kill -$KILL_SIGNAL $PID
        KILLED_COUNT=$((KILLED_COUNT + 1))
    done
fi

if [ -n "$MASTER_PROCESSES" ]; then
    echo "Killing master processes..."
    for PID in $MASTER_PROCESSES; do
        echo "Killing master process $PID with SIG$KILL_SIGNAL"
        kill -$KILL_SIGNAL $PID
        KILLED_COUNT=$((KILLED_COUNT + 1))
    done
fi

echo "Killed $KILLED_COUNT OneSim processes."

# Verify all processes are gone
sleep 2
REMAINING_PROCESSES=$(pgrep -f "python src/main.py.*(--mode master|--mode worker)" || echo "")

if [ -n "$REMAINING_PROCESSES" ]; then
    echo "Warning: Some processes are still running:"
    ps -f -p $REMAINING_PROCESSES
    echo ""
    echo "Try running the script again to kill remaining processes."
else
    echo "All OneSim processes have been terminated successfully."
fi