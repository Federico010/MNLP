"""
Module to handle the graph data structure.

Useful classes:
- SimilarityGraph
"""

from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch


class SimilarityGraph:
    """
    Class to handle the similarity graph.

    Useful methods:
    - get_graphs
    """

    def __init__(self, df: pd.DataFrame, threshold: float = 0.5, connected: bool = False, show: bool = False) -> None:
        """
        Initialize the similarity graph.

        Args:
            df: DataFrame containing the data.
            threshold: Threshold for the correlation.
            connected: if True, the graph will be connected.
            show: Whether to show the figure or not.
        """

        # Create the similarity matrix
        similarity_matrix: pd.DataFrame = self._get_iou_matrix(df)
        
        # Set the edges of the graph
        self._set_edges(similarity_matrix, threshold, connected)

        # Show the graph if required
        if show:
            # Create the graph
            G: nx.Graph = nx.Graph()
            G.add_nodes_from(df.columns)
            G.add_edges_from(self._edges)
            
            # Draw the graph
            self._show_graph(G)

  
    def _get_iou_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the IoU matrix.
        """

        # Create the IoU matrix
        iou_matrix: pd.DataFrame = pd.DataFrame(index = df.columns, columns = df.columns, dtype = float)

        for col in df.columns:
            iou_matrix.at[col, col] = 1.

        for col1, col2 in combinations(df.columns, 2):
            series1: pd.Series[float] = df[col1]
            series2: pd.Series[float] = df[col2]

            # Calculate the IoU
            intersection: float = ((series1 != 0) & (series2 != 0)).sum()
            union: float = ((series1 != 0) | (series2 != 0)).sum()
            iou: float = intersection / union if union > 0 else 0
            iou_matrix.at[col1, col2] = iou
            iou_matrix.at[col2, col1] = iou
        
        return iou_matrix


    def _set_edges(self, similarity_matrix: pd.DataFrame, threshold: float, connected: bool = False) -> None:
        """
        Set the edges of the graph from the similarity matrix.

        Args:
            similarity_matrix: Similarity matrix.
            threshold: Threshold for the correlation.
            connected: if True, the graph will be connected.
        """

        self._nodes: set[str] = set(similarity_matrix.columns)
        self._edges: set[tuple[str, str]]

        if connected:
            # Create a fully connected graph
            all_weighted_edges: set[tuple[str, str, float]] = {(col1, col2, similarity_matrix.at[col1, col2])
                                                            for col1, col2 in combinations(similarity_matrix.columns, 2)
                                                            }
            fully_connected_graph: nx.Graph = nx.Graph()
            fully_connected_graph.add_weighted_edges_from(all_weighted_edges)

            # Find a maximum spanning tree
            spanning_tree: nx.Graph = nx.maximum_spanning_tree(fully_connected_graph)
            self._edges = set(spanning_tree.edges())
        else:
            self._edges = set()
    
        # Add the edges above the threshold
        rows: NDArray[np.intp]
        cols: NDArray[np.intp]
        rows, cols = np.where(similarity_matrix >= threshold)
        for i, j in zip(rows, cols):
            if i < j:
                self._edges.add((similarity_matrix.columns[i], similarity_matrix.columns[j]))


    def _show_graph(self, G: nx.Graph) -> None:
        """
        Show the graph.
        """

        plt.figure(figsize=(10, 8))
        nx.draw(G,
                nx.spring_layout(G),
                with_labels = True,
                node_color = 'skyblue',
                font_size = 10,
                node_size = 1000,
                edge_color = 'grey',
                alpha = 0.7
                )
        plt.title("Similarity Graph")
        plt.show()

    
    def get_graphs(self, df: pd.DataFrame) -> list[nx.Graph]:
        """
        Populate the graphs with the data. Each node will be associated with the values in  the columns that start with the same name.
        """

        graphs: list[nx.Graph] = []

        # Take a list of suffixes
        suffixes: set[str] = set()
        for node in self._nodes:
            node_suffixes: set[str] = set()
            for col in df.columns:
                if col.startswith(node):
                    node_suffixes.add(col[len(node):])
            suffixes = suffixes.intersection(node_suffixes) if suffixes else node_suffixes

        # Create the graphs
        for _, row in df.iterrows():
            graph: nx.Graph = nx.Graph()
            for node in self._nodes:
                graph.add_node(node, x_graph = torch.tensor([row[node + suffix] for suffix in suffixes], dtype = torch.float32))
            graph.add_edges_from(self._edges)
            graphs.append(graph)
        
        return graphs
