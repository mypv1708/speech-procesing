#!/bin/bash
# Script để khởi động hệ thống với cấu hình robot server

# Cấu hình Robot Server
export ROBOT_SERVER_HOST=192.168.0.111
export ROBOT_SERVER_PORT=12345

echo "============================================================"
echo "Robot Server Configuration:"
echo "  Host: $ROBOT_SERVER_HOST"
echo "  Port: $ROBOT_SERVER_PORT"
echo "============================================================"
echo ""

# Kích hoạt virtual environment nếu có
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Chạy main application
python main.py

