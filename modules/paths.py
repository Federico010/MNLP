"""
Module to define paths used in the whole project.

Useful constants:
- DATA_DIR: Path to the data directory.
- TRANSFORMER_MODEL_DIR: Path to everything related to the transformer model.
- GRAPH_MODEL_DIR: Path to everything related to the graph model.
- TRAINING_SET: URI to the training set file. It is downloaded from the Hugging Face Hub.
- VALIDATION_SET: URI to the validation set file. It is downloaded from the Hugging Face Hub.
- TEST_SET: Path to the test set file.
- UPDATED_TRAINING_SET: Path to the updated training set file.
- UPDATED_VALIDATION_SET: Path to the updated validation set file.
- UPDATED_TEST_SET: Path to the updated test set file.
- TRANSFORMER_PREDICTIONS: Path to the transformer model predictions.
- GRAPH_PREDICTIONS: Path to the graph model predictions.
"""

from pathlib import Path


DATA_DIR: Path = Path('data')
TRANSFORMER_MODEL_DIR: Path = DATA_DIR / 'transformer_model'
GRAPH_MODEL_DIR: Path = DATA_DIR / 'graph_model'
DATASET_DIR: Path = DATA_DIR / 'dataset'
DATASET_URI: str = 'hf://datasets/sapienzanlp/nlp2025_hw1_cultural_dataset'
TRAIN_SET: str = f'{DATASET_URI}/train.csv'
VALIDATION_SET: str = f'{DATASET_URI}/valid.csv'
TEST_SET: Path = DATASET_DIR / 'test.csv'
UPDATED_TRAIN_SET: Path = DATASET_DIR / 'updated_train.csv'
UPDATED_VALIDATION_SET: Path = DATASET_DIR / 'updated_validation.csv'
UPDATED_TEST_SET: Path = DATASET_DIR / 'updated_test.csv'
TRANSFORMER_PREDICITONS: Path = Path('DerfMax_output_model1.csv')
GRAPH_PREDICTIONS: Path = Path('DerfMax_output_model2.csv')

# Create the needed folders if they don't exist
TRANSFORMER_MODEL_DIR.mkdir(parents = True, exist_ok = True)
GRAPH_MODEL_DIR.mkdir(parents = True, exist_ok = True)
