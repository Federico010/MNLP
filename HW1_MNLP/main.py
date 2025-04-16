"""
Starting point for the script.

Imports: dataset, graph
"""

from typing import Literal

import networkx as nx
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from modules import dataset, graph


def main() -> None:
    """
    Main function to run the script.
    """

    # Parameters for dataset creation
    mode: Literal['iou', 'correlation', 'filtered correlation'] = 'iou'
    treshold: float = 0.5
    
    # Prepare the training set
    train_set: pd.DataFrame = dataset.prepare_dataset('train')
    train_x: pd.DataFrame = train_set.drop(columns=['label']).sample(frac=1, random_state=42)
    train_y: pd.Series = train_set['label']

    G: nx.Graph = graph.get_similarity_graph(train_x, similarity_threshold=treshold, mode=mode, save_fig=True)
    torch_graph: Data = from_networkx(G)

    # Prepare the validation set
    dataset.prepare_dataset('valid')


if __name__ == '__main__':
    main()
