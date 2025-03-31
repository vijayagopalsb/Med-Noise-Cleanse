# File: src/core/train.py

# Import Libraries

# Add this BEFORE importing tensorflow
import os
import sys
import absl.logging

# Suppress all types messages tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress absl warnings
absl.logging.set_verbosity(absl.logging.ERROR)

# Add project root to sys.path to resolve module imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import Custom App Libraries

from src.core.training import Trainer
from config.settings import DATASET_PATH
from config.logging_config import logger

if __name__ == "__main__":
  
    trainer = Trainer(DATASET_PATH)
    # Start Train
    logger.info("Start Trainig ...")
    trainer.train()
    logger.info("Successfully Completed the traning.")
