#!/bin/bash
set -e

# Default values
DEV_MODE=false
PORT=8000

if [ -f .env ]; then
    # Load environment variables from .env file
    source .env
fi

# Function to display help information
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dev              Run in development mode using 'fastapi dev'"
    echo "  --port NUMBER      Specify the port number (default: 8000)"
    echo "  --help             Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Run in production mode on port 8000"
    echo "  $0 --dev           # Run in development mode on port 8000"
    echo "  $0 --port 9000     # Run in production mode on port 9000"
    echo "  $0 --dev --port 5000  # Run in development mode on port 5000"
    echo ""
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dev) DEV_MODE=true; shift ;;
        --port)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --port requires a numeric argument"
                show_help
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: port must be a valid number"
                show_help
                exit 1
            fi
        PORT="$2"; shift 2 ;;
        --help)
            show_help
            exit 0
        ;;
        *)
            echo "Error: Unknown parameter: $1"
            show_help
            exit 1
        ;;
    esac
done

# Choose command based on dev mode
if [ "$DEV_MODE" = true ]; then
    echo "Starting in development mode on port $PORT"
    uv run fastapi dev ./src/transcribo_backend/app.py --port "$PORT"
else
    echo "Starting in production mode on port $PORT"
    uv run fastapi run ./src/transcribo_backend/app.py --port "$PORT"
fi
