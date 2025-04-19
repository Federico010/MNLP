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
from typing import Optional, List

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
            series1: pd.Series = df[col1]
            series2: pd.Series = df[col2]

            # Calculate the IoU
            intersection: float = (series1.notna() & series2.notna()).sum()
            union: float = (series1.notna() | series2.notna()).sum()
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

        self._edges = set()
        node_arities: Counter = Counter({col: 0 for col in similarity_matrix.columns})

        for col1, col2 in combinations(similarity_matrix.columns, 2):
            value: float = similarity_matrix.at[col1, col2]
            if value >= threshold:
                self._edges.add((str(col1), str(col2)))
                node_arities[col1] += 1
                node_arities[col2] += 1

        # Add edges to get the minimum arity
        for node1, arity in node_arities.items():
            if arity < min_arity:
                # add the best edges to the node
                for node2 in similarity_matrix.nlargest(min_arity + 1, node1, 'all').index[1:]:
                    self._edges.add((str(node1), str(node2)))
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

    
    def get_graphs(self, df: pd.DataFrame, global_features: Optional[List[float]] = None) -> list[nx.Graph]:
        """
        Populate the graphs with the data.
        """
        graphs: list[nx.Graph] = []
        for i, (_, row) in enumerate(df.iterrows()):
            graph: nx.Graph = nx.Graph()
            for col, value in row.items():
                if global_features is not None:
                    graph.add_node(col, x=(value, global_features[i]))
                else:
                    graph.add_node(col, x=(value,))
            graph.add_edges_from(self._edges)
            graphs.append(graph)
        return graphs
