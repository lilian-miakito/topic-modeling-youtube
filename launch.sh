#!/bin/bash
# =============================================================================
# To-Peek Launch Script
# =============================================================================
# Launches frontend in Docker + backend locally (for MPS GPU acceleration)
#
# Usage:
#   ./launch.sh          # Start both services
#   ./launch.sh stop     # Stop all services
#   ./launch.sh logs     # Show logs
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PID_FILE="$PROJECT_ROOT/.backend.pid"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------------------------------------------------------------------------
# Stop function (aggressive kill of uvicorn and subprocesses)
# ---------------------------------------------------------------------------
stop_services() {
    log_info "Stopping services..."
    
    # Stop frontend Docker
    cd "$PROJECT_ROOT"
    docker compose -f docker-compose.frontend.yml down 2>/dev/null || true
    
    # Stop backend from PID file
    if [ -f "$BACKEND_PID_FILE" ]; then
        PID=$(cat "$BACKEND_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            # Kill process group (parent + children)
            pkill -9 -P "$PID" 2>/dev/null || true
            kill -9 "$PID" 2>/dev/null || true
            log_success "Backend process tree killed (PID $PID)"
        fi
        rm -f "$BACKEND_PID_FILE"
    fi
    
    # Kill ALL uvicorn processes
    log_info "Killing all uvicorn processes..."
    pkill -9 -f "uvicorn.*app.main" 2>/dev/null || true
    pkill -9 -f "uvicorn.*to-peek" 2>/dev/null || true
    
    # Kill processes on port 8000
    log_info "Clearing port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    
    # Kill any Python processes in to-peek-backend
    log_info "Killing related Python processes..."
    pkill -9 -f "python.*to-peek-backend" 2>/dev/null || true
    
    # Kill multiprocessing workers (HDBSCAN, joblib, etc.)
    pkill -9 -f "multiprocessing.*spawn" 2>/dev/null || true
    pkill -9 -f "loky" 2>/dev/null || true
    
    # Wait a moment for cleanup
    sleep 1
    
    # Verify nothing left on port 8000
    if lsof -ti:8000 > /dev/null 2>&1; then
        log_warn "Some processes still on port 8000, force killing..."
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
    
    # Final check
    if lsof -ti:8000 > /dev/null 2>&1; then
        log_error "Could not free port 8000. Check manually with: lsof -i:8000"
    else
        log_success "All services stopped, port 8000 free"
    fi
}

# ---------------------------------------------------------------------------
# Logs function
# ---------------------------------------------------------------------------
show_logs() {
    docker compose -f docker-compose.frontend.yml logs -f
}

# ---------------------------------------------------------------------------
# Main start function
# ---------------------------------------------------------------------------
start_services() {
    log_info "Starting To-Peek (Frontend: Docker, Backend: Local + MPS)"
    echo ""
    
    # Check if already running
    if [ -f "$BACKEND_PID_FILE" ] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
        log_warn "Backend already running. Use './launch.sh stop' first."
        exit 1
    fi
    
    # ---------------------------------------------------------------------------
    # 1. Start Backend (local uvicorn with MPS)
    # ---------------------------------------------------------------------------
    log_info "Starting backend (local, MPS GPU enabled)..."
    cd "$PROJECT_ROOT/to-peek-backend"
    
    # Activate venv if exists
    if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    fi
    
    # Create data directory
    mkdir -p "$PROJECT_ROOT/to-peek-backend/data"
    
    # Start uvicorn in background
    nohup uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 \
        > "$PROJECT_ROOT/.backend.log" 2>&1 &
    echo $! > "$BACKEND_PID_FILE"
    
    log_success "Backend started (PID $(cat $BACKEND_PID_FILE))"
    log_info "Backend logs: tail -f $PROJECT_ROOT/.backend.log"
    
    # Wait for backend to be ready
    log_info "Waiting for backend..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Backend ready at http://localhost:8000"
            break
        fi
        sleep 1
    done
    
    # ---------------------------------------------------------------------------
    # 2. Start Frontend (Docker)
    # ---------------------------------------------------------------------------
    log_info "Starting frontend (Docker)..."
    cd "$PROJECT_ROOT"
    docker compose -f docker-compose.frontend.yml up -d --build
    
    log_success "Frontend started at http://localhost:3000"
    
    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    echo ""
    echo "=============================================="
    echo -e "${GREEN}To-Peek is running!${NC}"
    echo "=============================================="
    echo "Frontend:  http://localhost:3000"
    echo "Backend:   http://localhost:8000"
    echo "API Docs:  http://localhost:8000/docs"
    echo ""
    echo "Commands:"
    echo "  ./launch.sh stop     Stop all services"
    echo "  ./launch.sh logs     Frontend logs"
    echo "  tail -f .backend.log Backend logs"
    echo "=============================================="
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-start}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    logs)
        show_logs
        ;;
    restart)
        stop_services
        sleep 2
        start_services
        ;;
    *)
        echo "Usage: $0 {start|stop|logs|restart}"
        exit 1
        ;;
esac

