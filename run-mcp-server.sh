#!/bin/bash
# Wrapper script to run the MCP server with uv
# This ensures the virtual environment is always properly managed

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the project directory
cd "$SCRIPT_DIR"

# Find uv in PATH or common locations
find_uv() {
    # Check if uv is in PATH
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi

    # Check common installation locations
    local common_paths=(
        "$HOME/.local/bin/uv"
        "$HOME/.cargo/bin/uv"
        "/usr/local/bin/uv"
        "/opt/homebrew/bin/uv"
    )

    for path in "${common_paths[@]}"; do
        if [ -x "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    return 1
}

UV_PATH=$(find_uv) || {
    echo "Error: uv not found. Please install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
}

# Ensure dependencies are synced (this is fast if already synced)
"$UV_PATH" sync --quiet 2>/dev/null || true

# Run the MCP server with stdio transport (default for MCP)
# Use Python 3.12 explicitly to avoid compatibility issues with newer Python versions
# Pass through any additional arguments from the command line
exec "$UV_PATH" run --python 3.12 python -m prompt_saver_mcp.server stdio "$@"

