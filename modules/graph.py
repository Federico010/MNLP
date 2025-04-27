"""
Module to handle the graph data structure.

Useful classes:
- SimilarityGraph
"""

from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch


class SimilarityGraph:
    """
    Class to handle the similarity graph.

    Useful methods:
    - get_graphs
    """

    def __init__(self, df: pd.DataFrame, threshold: float = 0.5, min_arity: int = 0, show: bool = False) -> None:
        """
        Initialize the similarity graph.

        Args:
            df: DataFrame containing the data.
            threshold: Threshold for the correlation.
            min_arity: Minimum arity for the nodes.
            show: Whether to show the figure or not.
        """

        # Create the similarity matrix
        similarity_matrix: pd.DataFrame = self._get_iou_matrix(df)
        
        # Set the edges of the graph
        self._set_edges(similarity_matrix, threshold, min_arity)

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


    def _set_edges(self, similarity_matrix: pd.DataFrame, threshold: float, min_arity: int) -> None:
        """
        Set the edges of the graph from the similarity matrix.

        Args:
            similarity_matrix: Similarity matrix.
            threshold: Threshold for the correlation.
            min_arity: Minimum arity for the nodes.
        """

        self._edges: set[tuple[str, str]] = set()
        self._nodes: set[str] = set(similarity_matrix.columns)
        node_arities: Counter = Counter({node: 0 for node in self._nodes})

        for col1, col2 in combinations(similarity_matrix.columns, 2):
            value: float = similarity_matrix.at[col1, col2]
            if value >= threshold:
                self._edges.add((col1, col2))
                node_arities[col1] += 1
                node_arities[col2] += 1

        # Add edges to get the minimum arity
        for node1, arity in node_arities.items():
            if arity < min_arity:
                # add the best edges to the node
                for node2 in similarity_matrix.nlargest(min_arity + 1, node1, 'all').index[1:]:
                    self._edges.add((node1, node2))
                    node_arities[node1] += 1
                    node_arities[node2] += 1
            

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
