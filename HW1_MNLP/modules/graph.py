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

    def __init__(self, df: pd.DataFrame, k: int = 3, show: bool = False) -> None:
        """
        Initialize the similarity graph.

        Args:
            df: DataFrame containing the data.
            k: Number of edges to keep for each node, excluding itself.
            show: Whether to show the figure or not.
        """

        # Create the correlation matrix
        correlation_matrix: pd.DataFrame = self._get_correlation_matrix(df)
        
        # Find the top k edges for each column (+ itself)
        self._set_edges(correlation_matrix, k)

        # Show the graph if required
        if show:
            # Create the graph
            G: nx.DiGraph = nx.DiGraph()
            G.add_nodes_from(df.columns)
            G.add_weighted_edges_from(self._edges)
            
            # Draw the graph
            self._show_graph(G)


    def _get_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the correlation matrix.
        """

        # Create the correlation matrix
        correlation_matrix: pd.DataFrame = pd.DataFrame(index = df.columns, columns = df.columns, dtype = float)

        for col in df.columns:
            correlation_matrix.at[col, col] = 1.

        for col1, col2 in combinations(df.columns, 2):
            series1: pd.Series = df[col1]
            series2: pd.Series = df[col2]

            # Replace NaNs if only one of the series has NaN
            series1 = series1.where(series1.notna() | series2.isna(), other = 0)
            series2 = series2.where(series2.notna() | series1.isna(), other = 0)

            # Calculate the correlation
            correlation: float = series1.corr(series2)
            correlation_matrix.at[col1, col2] = correlation
            correlation_matrix.at[col2, col1] = correlation
        
        return correlation_matrix
    

    def _set_edges(self, correlation_matrix: pd.DataFrame, k: int) -> None:
        """
        Set the edges of the graph from the correlation matrix.

        Args:
            correlation_matrix: Correlation matrix.
            k: Number of edges to keep for each node, excluding itself.
        """

        self._edges = set()
        for col1, row in correlation_matrix.iterrows():
            top_k: pd.Series = row.nlargest(k + 1)
            for col2, value in top_k.items():
                self._edges.add((str(col1), str(col2), value))


    def _show_graph(self, G: nx.DiGraph) -> None:
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

    
    def get_graphs(self, df: pd.DataFrame) -> list[nx.DiGraph]:
        """
        Populate the graphs with the data.
        """

        graphs: list[nx.DiGraph] = []

        # Create the graphs
        for _, row in df.iterrows():
            graph: nx.DiGraph = nx.DiGraph()
            for col, value in row.items():
                graph.add_node(col, x=(value,))
            graph.add_weighted_edges_from(self._edges)
            graphs.append(graph)
        
        return graphs
