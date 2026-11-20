#!/bin/bash
set -e

MODE=$1

CONFIG_PORT=$(node -p "require('./config.json').server.port")
WS_DEV=$(node -p "require('./config.json').ws.dev")
WS_PROD=$(node -p "require('./config.json').ws.prod")

echo "📦 Config loaded:"
echo "PORT=$CONFIG_PORT"
echo "WS_DEV=$WS_DEV"
echo "WS_PROD=$WS_PROD"

if lsof -iTCP:"$CONFIG_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "❌ port $CONFIG_PORT is already in use: stop the old server first (lsof -iTCP:$CONFIG_PORT -sTCP:LISTEN)"
    exit 1
fi

# =====================================================
# DEV MODE
# =====================================================
if [ "$MODE" = "dev" ]; then
    echo "🚀 DEV MODE STARTING"

    # -------------------------
    # backend (C++)
    # -------------------------
    make server consumer   # the server spawns ./consumer for each MFF-lab session
    ./server &
    SERVER_PID=$!

    # -------------------------
    # frontend (VITE)
    # IMPORTANT: inject env here
    # -------------------------
    cd client

    export VITE_WS_URL=$WS_DEV

    npm install
    npm run dev &
    VITE_PID=$!

    cd ..

    echo "⚡ DEV READY"
    echo "Frontend: http://localhost:5173"
    echo "Backend:  http://localhost:$CONFIG_PORT"

    trap "kill $SERVER_PID $VITE_PID" EXIT
    wait

# =====================================================
# PROD MODE
# =====================================================
elif [ "$MODE" = "prod" ]; then
    echo "🚀 PROD MODE STARTING"

    # -------------------------
    # build frontend
    # -------------------------
    cd client

    export VITE_WS_URL=$WS_PROD

    npm install
    npm run build

    cd ..

    # -------------------------
    # build backend
    # -------------------------
    make server consumer   # the server spawns ./consumer for each MFF-lab session
    ./server &
    SERVER_PID=$!

    echo "🌍 Server running on http://localhost:$CONFIG_PORT"
    echo "🌐 ngrok should point to:"
    echo "   ngrok http $CONFIG_PORT"

    trap "kill $SERVER_PID" EXIT
    wait

else
    echo "Usage: ./run.sh dev | ./run.sh prod"
    exit 1
fi