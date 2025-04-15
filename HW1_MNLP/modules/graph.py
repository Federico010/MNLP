"""
Module to handle the graph data structure.

Useful functions:
- get_similarity_graph

Imports: paths
"""

from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from modules import paths


def _iou(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Intersection over Union (IoU) matrix for the given DataFrame.   
    """

    # Initialize an empty IoU matrix
    iou_matrix: pd.DataFrame = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)

    # Calculate the IoU matrix
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2:
                iou_matrix.loc[col1, col2] = 1.0  # IoU is 1.0 with itself
            else:
                # Calculate intersection and union
                intersection: int = ((df[col1] != 0) & (df[col2] != 0)).sum()
                union: int = ((df[col1] != 0) | (df[col2] != 0)).sum()

                # Calculate IoU
                if union > 0:
                    iou_matrix.loc[col1, col2] = intersection / union
                else:
                    iou_matrix.loc[col1, col2] = 0.0  # IoU is 0 if there is no union

    return iou_matrix


def _filtered_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the filtered correlation matrix (obtained by avoiding to consider datas missing for both parts) for the given DataFrame.
    """

    # Initialize an empty correlation matrix
    correlation_matrix = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)

    # Calculate the correlation matrix, ignoring rows where both columns are 0
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2:
                correlation_matrix.loc[col1, col2] = 1.0  # Correlazione perfetta con se stessi
            else:
                # Filtra le righe in cui entrambe le colonne sono 0
                filtered_df: pd.DataFrame = df[(df[col1] != 0) | (df[col2] != 0)]
                if not filtered_df.empty:
                    correlation_matrix.loc[col1, col2] = filtered_df[col1].corr(filtered_df[col2])
                else:
                    correlation_matrix.loc[col1, col2] = 0.0  # Nessuna correlazione se non ci sono dati validi

    return correlation_matrix


def get_similarity_graph(df: pd.DataFrame,
                         similarity_threshold: float = 0.5,
                         mode: Literal['iou', 'correlation', 'filtered correlation'] = 'iou',
                         save_fig: bool = False
                         ) -> nx.Graph:
    """
    Create a similarity graph from the given DataFrame.

    Args:
        df: DataFrame containing the data.
        similarity_threshold: Threshold for similarity.
        mode: Mode of similarity calculation ('iou', 'correlation', 'filtered correlation').
        save_fig: Whether to save the figure or not.
    """

    similarity_matrix: pd.DataFrame

    if mode == 'iou':
        similarity_matrix = _iou(df)
    elif mode == 'correlation':
        similarity_matrix = df.corr()
    else:
        similarity_matrix = _filtered_correlation(df)
    
    similarity_matrix.to_csv(paths.MATRIX_SIMILARITY_FOLDER / f'{mode}_matrix.csv')

    # Create a graph from the correlation matrix
    G: nx.Graph = nx.Graph()
    for i, row in similarity_matrix.iterrows():
        for j, value in row.items():
            if i != j and value >= similarity_threshold:
                G.add_edge(i, j)
    
    # Draw the graph
    if save_fig:
        plt.figure(figsize=(10, 8))
        nx.draw(
            G, nx.spring_layout(G), with_labels=True, node_color='skyblue', font_size=10,
            node_size=1000, edge_color='grey', alpha=0.7
        )
        plt.title(f"Similarity Graph ({mode}, threshold: {similarity_threshold})")
        plt.savefig(paths.GRAPH_SIMILARITY_FOLDER / f'{mode}_graph.png')

    return G
