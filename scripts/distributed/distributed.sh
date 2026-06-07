#!/bin/bash

# Default values
MASTER_ADDRESS="127.0.0.1"
MASTER_PORT="10051"
NUM_WORKERS=2
CONFIG_PATH="config/config.json"
MODEL_CONFIG_PATH="config/model_config.json"
BASE_WORKER_PORT=10052

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -a, --address MASTER_ADDRESS    Master address (default: 127.0.0.1)"
    echo "  -p, --port MASTER_PORT          Master port (default: 50051)"
    echo "  -w, --workers NUM_WORKERS       Number of worker nodes (default: 2)"
    echo "  -c, --config CONFIG_PATH        Path to config file (default: config/hy_config.json)"
    echo "  -m, --model MODEL_CONFIG_PATH   Path to model config (default: /data/wl/OneSim/config/llm/model_config_7b.json)"
    echo "  -h, --help                      Show this help message"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--address)
            MASTER_ADDRESS="$2"
            shift 2
            ;;
        -p|--port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -w|--workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_CONFIG_PATH="$2"
            shift 2
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

echo "Starting OneSim with $NUM_WORKERS workers"
echo "Master address: $MASTER_ADDRESS"
echo "Master port: $MASTER_PORT"
echo "Config: $CONFIG_PATH"
echo "Model config: $MODEL_CONFIG_PATH"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start master
echo "Starting master node..."
nohup python src/main.py \
    --config "$CONFIG_PATH" \
    --model_config "$MODEL_CONFIG_PATH" \
    --mode master \
    --master_port "$MASTER_PORT" \
    --expected_workers "$NUM_WORKERS" \
    > logs/master.log 2>&1 &

MASTER_PID=$!
echo "Master started with PID: $MASTER_PID"

# Give master a moment to start up
sleep 2

# Start workers
for ((i=0; i<NUM_WORKERS; i++)); do
    WORKER_PORT=$((BASE_WORKER_PORT + i))
    echo "Starting worker $((i+1)) on port $WORKER_PORT..."
    WORKER_CMD="python src/main.py --config $CONFIG_PATH --model_config $MODEL_CONFIG_PATH --mode worker --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --worker_port $WORKER_PORT"
    echo "Executing worker command:"
    echo "$WORKER_CMD"
    nohup python src/main.py \
        --config "$CONFIG_PATH" \
        --model_config "$MODEL_CONFIG_PATH" \
        --mode worker \
        --master_address "$MASTER_ADDRESS" \
        --master_port "$MASTER_PORT" \
        --worker_port "$WORKER_PORT" \
        > "logs/worker_${i}_port_${WORKER_PORT}.log" 2>&1 &
    
    WORKER_PID=$!
    echo "Worker $((i+1)) started with PID: $WORKER_PID"
    echo "Worker $((i+1)) log: logs/worker_${i}_port_${WORKER_PORT}.log" 
    
    # Small delay between starting workers
    sleep 0.1
done

echo "All processes started. Use 'tail -f logs/*.log' to monitor logs."
echo "To check running processes: ps aux | grep src/main.py"