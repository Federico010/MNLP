"""
Module to handle the graph data structure.

Useful classes:
- SimilarityGraph
"""

from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


class SimilarityGraph:
    """
    Class to handle the similarity graph.

    Useful methods:
    - get_graphs
    """

    def __init__(self, df: pd.DataFrame, threshold: float = 0.5, show: bool = False) -> None:
        """
        Initialize the similarity graph.

        Args:
            df: DataFrame containing the data.
            threshold: Threshold for similarity.
            show: Whether to show the figure or not.
        """

        # Initialize the edges
        self._edges: set[tuple[str, str]] = set()

        # Create the similarity matrix
        similarity_matrix: pd.DataFrame = df.corr()

        # Add edges based on the similarity threshold
        for col1, col2 in combinations(similarity_matrix.columns, 2):
            value: float = similarity_matrix.at[col1, col2]
            if value >= threshold:
                self._edges.add((col1, col2))
        
        # Print the percentage of edges
        percentage: float = len(self._edges) / (len(df.columns) * (len(df.columns) - 1) / 2)
        print(f"Percentage of edges: {percentage:.2%}")

        # Show the graph if required
        if show:
            # Create the graph
            G: nx.Graph = nx.Graph()
            G.add_nodes_from(df.columns)
            G.add_edges_from(self._edges)
            
            # Draw the graph
            plt.figure(figsize=(10, 8))
            nx.draw(
                G, nx.spring_layout(G), with_labels=True, node_color='skyblue', font_size=10,
                node_size=1000, edge_color='grey', alpha=0.7
            )
            plt.title("Similarity Graph")
            plt.show()

    
    def get_graphs(self, df: pd.DataFrame) -> list[nx.Graph]:
        """
        Populate the graphs with the data.
        """

        graphs: list[nx.Graph] = []

        # Create the graphs
        for _, row in df.iterrows():
            graph: nx.Graph = nx.Graph()
            for col, value in row.items():
                graph.add_node(col, x=(value,))
            graph.add_edges_from(self._edges)
            graphs.append(graph)
        
        return graphs
