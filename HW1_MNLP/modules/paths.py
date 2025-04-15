"""
Module to define paths used in the whole project.

Useful constants:
- UPDATED_TRAINING_SET: Path to the updated training set file.
- UPDATED_VALIDATION_SET: Path to the updated validation set file.
- MATRIX_SIMILARITY_FOLDER: Path to the folder containing the matrix similarity files.
- GRAPH_SIMILARITY_FOLDER: Path to the folder containing the graph similarity files.
"""

from pathlib import Path


DATA_FOLDER: Path = Path('dataset')
UPDATED_TRAIN_SET: Path = DATA_FOLDER / 'updated_train.csv'
UPDATED_VALID_SET: Path = DATA_FOLDER / 'updated_valid.csv'

SIMILARITY_FOLDER: Path = Path('similarity')
MATRIX_SIMILARITY_FOLDER: Path = SIMILARITY_FOLDER / 'matrix'
GRAPH_SIMILARITY_FOLDER: Path = SIMILARITY_FOLDER / 'graph'

# Create the needed folders if they don't exist
MATRIX_SIMILARITY_FOLDER.mkdir(parents=True, exist_ok=True)
GRAPH_SIMILARITY_FOLDER.mkdir(parents=True, exist_ok=True)
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
