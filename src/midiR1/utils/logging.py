import logging
import os

def init_logger():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler('logs/train.log'),
            logging.StreamHandler()
        ]
    )