"""
Module to define paths used in the whole project.

USEFUL CONSTANTS:
- TRAINING_SET: Path to the training set file.
- UPDATED_TRAINING_SET: Path to the updated training set file.
"""

from pathlib import Path

DATA_FOLDER: Path = Path('dataset')
TRAINING_SET: Path = DATA_FOLDER / 'training.tsv'
UPDATED_TRAINING_SET: Path = DATA_FOLDER / 'updated_training.csv'
