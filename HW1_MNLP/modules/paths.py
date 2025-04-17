"""
Module to define paths used in the whole project.

Useful constants:
- UPDATED_TRAINING_SET: Path to the updated training set file.
- UPDATED_VALIDATION_SET: Path to the updated validation set file.
"""

from pathlib import Path


DATA_FOLDER: Path = Path('dataset')
UPDATED_TRAIN_SET: Path = DATA_FOLDER / 'updated_train.csv'
UPDATED_VALID_SET: Path = DATA_FOLDER / 'updated_valid.csv'

# Create the needed folders if they don't exist
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
