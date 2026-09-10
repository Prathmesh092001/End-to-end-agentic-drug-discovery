import os
import sys
import logging

LOG_DIR = "logs"
LOG_FILE_PATH = os.path.join(LOG_DIR, "running_logs.log")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT = "[%(asctime)s] %(levelname)s | %(name)s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stderr),
    ],
)

logger = logging.getLogger("partex_agent")
