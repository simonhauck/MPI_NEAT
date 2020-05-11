from typing import Dict, List

import networkx as nx

from neat_core.models.genome import Genome
from neat_core.models.node import NodeType, Node


class NetworkXGenomeGraph(object):

    def draw_genome_graph(self, genome: Genome, draw_labels=True) -> None:
        """
        Parse the genome into a networkx graph, which can be plotted with matplotlib, if the command plt.show()
        is invoked after this method
        :param genome: the genome that should be displayed
        :param draw_labels: True if the labels of the connections should be printed
        :return: None
        """
        graph = nx.DiGraph()

        node_colors = []

        for x_position, node_list in sorted(self._sort_nodes_in_layers(genome).items()):
            size = len(node_list)
            step_size = 1 / (size + 2)

            for node, i in zip(node_list, range(1, len(node_list) + 1)):
                y_position = i * step_size
                # Store color for later drawing
                node_colors.append(self._get_color_for_node_type(node.node_type))
                # Add node
                graph.add_node(node.innovation_number, pos=(x_position, y_position))

        disabled_connections = []
        enabled_connections = []
        for connection in genome.connections:
            graph.add_edge(connection.input_node, connection.output_node, label=connection.weight)
            if connection.enabled:
                enabled_connections.append((connection.input_node, connection.output_node))
            else:
                disabled_connections.append((connection.input_node, connection.output_node))

        # Extract required fields
        node_positions = nx.get_node_attributes(graph, 'pos')
        connection_labels = nx.get_edge_attributes(graph, 'label')

        # Draw graph edges
        nx.draw_networkx_edges(graph, node_positions, edgelist=enabled_connections,
                               connectionstyle='arc3, rad=0.1')
        collection = nx.draw_networkx_edges(graph, node_positions, edgelist=disabled_connections,
                                            alpha=0.5,
                                            edge_color='b',
                                            connectionstyle='arc3, rad=0.1')
        # Bug in the library, can't set the line style.:
        # https://stackoverflow.com/questions/51138059/no-dotted-line-with-networkx-drawn-on-basemap/51148746#51148746?s=e8a8f3c423e84da9aa77b1259b3ad829
        for patch in collection:
            patch.set_linestyle('dashed')

        # Draw graph edges labels
        if draw_labels:
            nx.draw_networkx_edge_labels(graph, node_positions, edge_labels=connection_labels, label_pos=0.7,
                                         rotate=False)

        # Draw graph node labels
        nx.draw_networkx_labels(graph, node_positions)

        # Draw graph nodes
        # nx.draw_net(graph, pos=node_positions, node_color=node_colors, connectionstyle='arc3, rad=0.1', with_labels=True)
        nx.draw_networkx_nodes(graph, pos=node_positions, node_color=node_colors, with_labels=True)

    def _get_color_for_node_type(self, node_type: NodeType) -> str:
        """
        Get a color for each node type
        :return: the color as hex value
        """
        colors = {
            NodeType.INPUT: '#ff4105',
            NodeType.HIDDEN: '#4ada76',
            NodeType.OUTPUT: '#002fa7'
        }
        return colors[node_type]

    def _sort_nodes_in_layers(self, genome: Genome) -> Dict[float, List[Node]]:
        """
        Sort the nodes in the genome into layers according to the x position
        :param genome: the genome that should be displayed
        :return: a dictionary with the x position as key and the value is a list with the nodes in the layer
        """
        layers = {}
        for node in genome.nodes:
            if node.x_position not in layers:
                layers[node.x_position] = [node]
            else:
                layers[node.x_position].append(node)
        return layers
