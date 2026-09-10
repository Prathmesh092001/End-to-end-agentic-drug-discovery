import os
import sys
import logging

LOG_DIR = "logs"
LOG_FILE_PATH = os.path.join(LOG_DIR, "running_logs.log")
os.makedirs(LOG_DIR, exist_ok=True)

# Reconfigure stderr stream for Windows console Unicode / UTF-8 compatibility
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)

logger = logging.getLogger("partex_agent")