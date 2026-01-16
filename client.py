"""
Robot Control Client - Gửi lệnh đến server và nhận phản hồi
Giữ kết nối để có thể gửi nhiều lệnh liên tiếp
"""
import socket
import logging
import sys
import time
import os

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Cấu hình từ environment variable
HOST = os.getenv('ROBOT_SERVER_HOST', 'localhost')
DEFAULT_PORT = 8888
PORT = int(os.getenv('ROBOT_SERVER_PORT', DEFAULT_PORT))


class RobotClient:
    """Client giữ kết nối với server để gửi nhiều lệnh"""
    
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self) -> bool:
        """Kết nối đến server"""
        try:
            if self.connected and self.socket:
                return True
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)  # Timeout 10 giây
            
            logger.info(f"Connecting to server {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info("✓ Connected to server")
            return True
            
        except ConnectionRefusedError:
            logger.error(f"Connection refused - is server running on {self.host}:{self.port}?")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Error connecting to server: {e}", exc_info=True)
            self.connected = False
            return False
    
    def send_command(self, command: str) -> bool:
        """
        Gửi lệnh đến server và nhận phản hồi
        
        Args:
            command: Lệnh cần gửi (ví dụ: "$SEQ;FWD,4;TR,90;STOP\n")
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            # Đảm bảo đã kết nối
            if not self.connected:
                if not self.connect():
                    return False
            
            # Gửi lệnh
            logger.info(f"Sending command: {repr(command)}")
            self.socket.sendall(command.encode('utf-8'))
            
            # Nhận phản hồi
            response = self.socket.recv(1024).decode('utf-8')
            logger.info(f"Received response: {repr(response)}")
            
            # Kiểm tra phản hồi (hỗ trợ nhiều format response)
            response_clean = response.strip()
            
            # Nếu response rỗng, có thể server không trả về gì (vẫn coi là thành công)
            if not response_clean:
                logger.info("✓ Command sent (no response from server, assuming success)")
                return True
            
            # Kiểm tra các format response hợp lệ
            if (response_clean == "$DONE,SEQ" or 
                response_clean == "Đã nhận payload!" or 
                "nhận" in response_clean.lower() or 
                "payload" in response_clean.lower() or
                "done" in response_clean.lower() or
                "success" in response_clean.lower() or
                "ok" in response_clean.lower()):
                logger.info("✓ Command executed successfully!")
                return True
            else:
                logger.warning(f"Unexpected response: {response}")
                # Vẫn coi là thành công nếu đã gửi được
                logger.info("Command was sent, treating as success")
                return True
                
        except socket.timeout:
            logger.error("Connection timeout - server did not respond")
            self.connected = False
            return False
        except (ConnectionResetError, BrokenPipeError):
            logger.warning("Connection lost, attempting to reconnect...")
            self.connected = False
            if self.connect():
                return self.send_command(command)  # Thử lại
            return False
        except Exception as e:
            logger.error(f"Error sending command: {e}", exc_info=True)
            self.connected = False
            return False
    
    def close(self):
        """Đóng kết nối"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        self.connected = False
        logger.info("Connection closed")


# Global client instance
_client = None


def get_client(host=None, port=None) -> RobotClient:
    """
    Lấy hoặc tạo client instance
    
    Args:
        host: Server host (mặc định từ env hoặc 'localhost')
        port: Server port (mặc định từ env hoặc 8888)
    """
    global _client
    if _client is None:
        _host = host if host is not None else HOST
        _port = port if port is not None else PORT
        _client = RobotClient(host=_host, port=_port)
    return _client


def send_command(command: str, host=None, port=None) -> bool:
    """
    Gửi lệnh đến server (giữ kết nối)
    
    Args:
        command: Lệnh cần gửi (ví dụ: "$SEQ;FWD,4;TR,90;STOP\n")
        host: Server host (tùy chọn, mặc định từ env)
        port: Server port (tùy chọn, mặc định từ env)
        
    Returns:
        True nếu thành công, False nếu thất bại
    """
    client = get_client(host=host, port=port)
    return client.send_command(command)


def main():
    """Main function - giữ kết nối và gửi lệnh"""
    # Parse command line arguments
    # Format: python client.py [port] [command]
    #        python client.py [host:port] [command]
    #        python client.py [command]
    
    host = HOST
    port = PORT
    command = None
    
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        
        # Kiểm tra format host:port
        if ':' in arg1:
            parts = arg1.split(':')
            host = parts[0]
            try:
                port = int(parts[1])
            except ValueError:
                logger.error(f"Invalid port in '{arg1}'. Using default port {PORT}")
                port = PORT
            # Command là argument tiếp theo nếu có
            command = sys.argv[2] if len(sys.argv) > 2 else None
        # Kiểm tra nếu là số (port)
        elif arg1.isdigit():
            port = int(arg1)
            # Command là argument tiếp theo nếu có
            command = sys.argv[2] if len(sys.argv) > 2 else None
        else:
            # Là command
            command = arg1
    
    logger.info(f"Connecting to server at {host}:{port}")
    client = RobotClient(host=host, port=port)
    
    if not client.connect():
        logger.error(f"Failed to connect to server at {host}:{port}")
        sys.exit(1)
    
    try:
        if command:
            # Gửi lệnh từ command line
            if not command.endswith('\n'):
                command += '\n'
            
            success = client.send_command(command)
            if success:
                logger.info("✓ Operation completed successfully")
            else:
                logger.error("✗ Operation failed")
        else:
            # Chế độ interactive - giữ kết nối và gửi nhiều lệnh
            logger.info("Interactive mode - keeping connection alive")
            logger.info("Enter commands (or 'quit' to exit):")
            
            # Gửi lệnh mặc định đầu tiên
            default_command = "$SEQ;FWD,4;TR,90;STOP\n"
            logger.info(f"Sending default command: {repr(default_command)}")
            client.send_command(default_command)
            
            # Giữ kết nối và chờ lệnh tiếp theo
            while True:
                try:
                    user_input = input("\nEnter command (or 'quit'): ").strip()
                    if user_input.lower() == 'quit':
                        break
                    
                    if user_input:
                        if not user_input.endswith('\n'):
                            user_input += '\n'
                        client.send_command(user_input)
                    else:
                        logger.info("Empty command, keeping connection alive...")
                        time.sleep(1)
                        
                except KeyboardInterrupt:
                    logger.info("\nInterrupted by user")
                    break
                except EOFError:
                    break
                    
    finally:
        client.close()
        logger.info("Client shutdown")


if __name__ == "__main__":
    main()

