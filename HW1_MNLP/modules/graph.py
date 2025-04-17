"""
Module to handle the graph data structure.

Useful functions:
- get_similarity_graph

Imports: paths
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def get_similarity_graphs(df: pd.DataFrame,
                         similarity_threshold: float = 0.5,
                         show: bool = False
                         ) -> list[nx.Graph]:
    """
    Create similarity graphs from the given DataFrame and populate them. They all share the same structure.

    Args:
        df: DataFrame containing the data.
        similarity_threshold: Threshold for similarity.
        show: Whether to show the figure or not.
    """

    similarity_matrix: pd.DataFrame = df.corr()
    
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
        plt.title(f"Similarity Graph, threshold: {similarity_threshold})")
        plt.show()

    return graphs
