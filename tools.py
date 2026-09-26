

from datetime import datetime

def get_system_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
