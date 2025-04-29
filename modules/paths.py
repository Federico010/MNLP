"""
Module to define paths used in the whole project.

Useful constants:
- DATA_DIR: Path to the data directory.
- TRANSFORMER_MODEL_DIR: Path to everything related to the transformer model.
- GRAPH_MODEL_DIR: Path to everything related to the graph model.
- UPDATED_TRAINING_SET: Path to the updated training set file.
- UPDATED_VALIDATION_SET: Path to the updated validation set file.
"""

from pathlib import Path


DATA_DIR: Path = Path('data')
TRANSFORMER_MODEL_DIR: Path = DATA_DIR / 'transformer_model'
GRAPH_MODEL_DIR: Path = DATA_DIR / 'graph_model'
DATASET_DIR: Path = DATA_DIR / 'dataset'
UPDATED_TRAIN_SET: Path = DATASET_DIR / 'updated_train.csv'
UPDATED_VALID_SET: Path = DATASET_DIR / 'updated_valid.csv'

# Create the needed folders if they don't exist
DATA_DIR.mkdir(parents = True, exist_ok = True)
TRANSFORMER_MODEL_DIR.mkdir(parents = True, exist_ok = True)
GRAPH_MODEL_DIR.mkdir(parents = True, exist_ok = True)
DATASET_DIR.mkdir(parents = True, exist_ok = True)
