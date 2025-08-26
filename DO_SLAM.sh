#!/bin/bash
# Robot + SLAM only (no camera). Run from repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start_full_system.sh" --no-camera "$@"
