# File: src/ports/data_loader.py

# Data Handling Interface

# Import Libraries

# Add this BEFORE importing tensorflow
import os
import sys


# Suppress all types messages tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from abc import ABC, abstractmethod
import numpy as np

class DataLoaderPort(ABC):
    """Interface for Data Handling"""

    @abstractmethod
    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load traning and tesing images."""
        pass



