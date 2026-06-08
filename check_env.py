import shutil
import socket

def check_sox():
    sox_path = shutil.which("sox")
    if sox_path:
        print(f"[OK] SoX found at: {sox_path}")
        return True
    else:
        print("[WARNING] SoX not found in PATH!") 
        print("  - This is required for some audio operations.")
        print("  - Installing Python bindings (pip install sox) is NOT enough.")
        print("  - Please download SoX from https://sourceforge.net/projects/sox/")
        print("  - Add it to your System PATH environment variable.")
        return False

def check_port(port=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', port)) == 0:
            print(f"[ERROR] Port {port} is ALREADY IN USE.")
            print("  - Please kill the process using this port.")
            return False
        else:
            print(f"[OK] Port {port} is free.")
            return True

if __name__ == "__main__":
    print("--- Environment Check ---")
    check_sox()
    check_port()
    print("-------------------------")
