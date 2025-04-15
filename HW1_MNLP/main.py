"""
Starting point for the script.

Imports: dataset, graph
"""

import pandas as pd

from modules import dataset, graph


def main() -> None:
    """
    Main function to run the script.
    """

    # Prepare the training set
    train_set: pd.DataFrame = dataset.prepare_dataset('train')
    train_x: pd.DataFrame = train_set.drop(columns=['label']).sample(frac=1, random_state=42)
    train_y: pd.Series = train_set['label']

    for mode, treshold in (('iou', 0.5), ('correlation', 0.45), ('filtered correlation', 0.4)):
        # Calculate the similarity graph
        graph.get_similarity_graph(train_x, similarity_threshold=treshold, mode=mode, save_fig=True)

    # Prepare the validation set
    dataset.prepare_dataset('valid')


if __name__ == '__main__':
    main()
