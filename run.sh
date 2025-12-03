#!/bin/sh
set -e

if [ -f .env ]; then
    # Load environment variables from .env file
    . .env
fi

# Ensure PORT has a value (in case .env sets it to empty)
PORT="${PORT:-8000}"

# Function to display help information
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --port NUMBER      Specify the port number (default: 8000)"
    echo "  --help             Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Run in production mode on port 8000"
    echo "  $0 --port 9000     # Run in production mode on port 9000"
    echo ""
}

# Parse command line arguments
while [ "$#" -gt 0 ]; do
    case $1 in
        --port)
            if [ -z "$2" ] || echo "$2" | grep -q "^-"; then
                echo "Error: --port requires a numeric argument"
                show_help
                exit 1
            fi
            if ! echo "$2" | grep -Eq "^[0-9]+$"; then
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

echo "Starting in production mode on port $PORT"
fastapi run ./src/transcribo_backend/app.py --port "$PORT"