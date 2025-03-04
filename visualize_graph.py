import networkx as nx
import matplotlib.pyplot as plt

# Load the GraphML file
graph = nx.read_graphml("/Users/mahirjain/Desktop/Personal Projects/matchmind-included-infoleague/bedrock_example/graph_chunk_entity_relation.graphml")

# Draw the graph
plt.figure(figsize=(10, 7))
nx.draw(graph, with_labels=True, node_size=500, node_color="lightblue", edge_color="gray")
plt.show()