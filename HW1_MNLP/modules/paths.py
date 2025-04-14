"""
Module to define paths used in the whole project.

USEFUL CONSTANTS:
- TRAINING_SET: Path to the training set file.
- UPDATED_TRAINING_SET: Path to the updated training set file.
"""

from pathlib import Path

DATA_FOLDER: Path = Path('dataset')
UPDATED_TRAIN_SET: Path = DATA_FOLDER / 'updated_train.csv'
UPDATED_VALID_SET: Path = DATA_FOLDER / 'updated_valid.csv'

# Create the dataset folder if it doesn't exist
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
