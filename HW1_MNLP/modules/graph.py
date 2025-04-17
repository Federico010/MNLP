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
                intersection: int = ((pd.notna(df[col1])) & (pd.notna(df[col2]))).sum()
                union: int = ((pd.notna(df[col1])) | (pd.notna(df[col2]))).sum()

                # Calculate IoU
                if union > 0:
                    iou_matrix.loc[col1, col2] = intersection / union
                else:
                    iou_matrix.loc[col1, col2] = 0.0  # IoU is 0 if there is no union

    return iou_matrix


def get_similarity_graphs(df: pd.DataFrame,
                         similarity_threshold: float = 0.5,
                         mode: Literal['iou', 'correlation'] = 'iou',
                         show: bool = False
                         ) -> list[nx.Graph]:
    """
    Create similarity graphs from the given DataFrame and populate them. They all share the same structure.

    Args:
        df: DataFrame containing the data.
        similarity_threshold: Threshold for similarity.
        mode: Mode of similarity calculation ('iou', 'correlation').
        show: Whether to show the figure or not.
    """

    similarity_matrix: pd.DataFrame

    if mode == 'iou':
        similarity_matrix = _iou(df)
    else:
        similarity_matrix = df.corr()
    
    # Add edges based on the similarity threshold
    egdes: list[tuple[str, str]] = []
    for i, row in similarity_matrix.iterrows():
        for j, value in row.items():
            if i != j and value >= similarity_threshold:
                egdes.append((str(i), str(j)))
    
    # Create the graphs
    graphs: list[nx.Graph] = []
    for _, row in df.iterrows():
        graph: nx.Graph = nx.Graph()
        for col, value in row.items():
            graph.add_node(str(col), x=(value,))
        graph.add_edges_from(egdes)
        graphs.append(graph)
    
    # Draw the first graph
    G: nx.Graph = graphs[0]
    if show:
        plt.figure(figsize=(10, 8))
        nx.draw(
            G, nx.spring_layout(G), with_labels=True, node_color='skyblue', font_size=10,
            node_size=1000, edge_color='grey', alpha=0.7
        )
        plt.title(f"Similarity Graph ({mode}, threshold: {similarity_threshold})")
        plt.show()

    return graphs
