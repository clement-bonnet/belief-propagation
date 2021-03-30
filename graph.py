from functools import reduce

import numpy as np
from scipy.special import softmax
import networkx as nx


# TODO : gerer MTT _

class Node:
    def __init__(self, name, node_type, dist=None, dist_index=None, nb_states=None):
        """
        node_type in {"F", "V"} for factor and variable
        if factor node you must provide
            dist : numpy array which is a tensor whose dimension is the number of variables
                It represents the distribution.
            dist_index : list of the name of each variable node 
                associated with each dimension of the distribution
        if variable node you must provide:
            nb_states : int.
        """
        self.name = name
        self.node_type = node_type
        assert(node_type in ["F", "V"])
        if node_type == "F":
            if type(dist) is not str:  # "MTT"
                assert dist is not None and dist_index is not None
                assert dist.ndim == len(dist_index)
                self.dist = np.log(dist)
                self.dist_index = dist_index
            else:
                self.dist = dist  # "MTT"
        else:
            assert nb_states is not None
            self.nb_states = nb_states


class Edge:
    """
    One edge for each direction.
    Private Class
    """

    def __init__(self, node_1, node_2):
        self.nodes = [node_1, node_2]
        assert node_1.node_type != node_2.node_type, "The graph must be bipartite."

        # if F_to_V : bool factor to variable or variable to factor
        self.F_to_V = node_1.node_type == "F" and node_2.node_type == "V"

        self.nb_variable = node_2.nb_states if self.F_to_V else node_1.nb_states
        # Convention: messages are log_messages
        self.message = np.zeros(self.nb_variable)

    def __repr__(self):
        node1, node2 = self.nodes
        return f"({node1.name} -> {node2.name})"


class Graph:
    def __init__(self, dic_nodes, edge_names):
        """
        dic_nodes : dictionnary mapping the name of each node to the node.
        edge_name : list containing the list of the form [node_name_1,node_name_2]
                    There is no need to put the nodes in both directions.
        """
        # We work with node NAMES and edge POINTERS
        self.dic_nodes = dic_nodes  # Maps names to nodes
        self.edges = [Edge(dic_nodes[node_name_1], dic_nodes[node_name_2])
                      for (node_name_1, node_name_2) in edge_names]
        self.edges += [Edge(dic_nodes[node_name_2], dic_nodes[node_name_1])
                       for (node_name_1, node_name_2) in edge_names]

        # Construction of the neighboring_nodes for all nodes
        # Because the graph is bipartite, the neighboring nodes of
        # V nodes are F nodes and reciprocaly.

        # Vocab : INCOMING of a NODE
        # and ANTECEDENT of an EDGE

        # self.incoming_edges = {}  # maps name of the node to list of incoming edges
        # for name, node in dic_nodes.items():
        #     incoming = [e for e in self.edges if e.nodes[1].name == name]
        #     self.incoming_edges[name] = incoming
        self.incoming_edges = {
            name: [e for e in self.edges if e.nodes[1].name == name]
            for name, _ in dic_nodes.items()}
        self.antecedant_edges = [
            [
                ant_edge for ant_edge in self.edges
                if (  # antecedent property
                    ant_edge.nodes[1].name == edge.nodes[0].name and
                    # Not inverse edge
                    ant_edge.nodes[0].name != edge.nodes[1].name)
            ]
            for edge in self.edges
        ]

##########################################
#       Old way of doing it
        # self.antecedant_edges = []  # list of list of edges
        # for edge in self.edges:
        #     self.antecedant_edges.append([
        #         ant_edge for ant_edge in self.edges
        #         if (
        #             # antecedent property
        #             ant_edge.nodes[1].name == edge.nodes[0].name and
        #             # Not inverse edge
        #             ant_edge.nodes[0].name != edge.nodes[1].name
        #         )
        #     ])
        #     if edge.nodes[0].node_type == "F":
        #         if edge.nodes[0].dist != "MTT":
        #             # check number of antecedent + 1 equals tensor dimension of distribution
        #             assert len(
        #                 self.antecedant_edges[-1]) + 1 == len(edge.nodes[0].dist_index)
        #             assert set(
        #                 self.antecedant_edges[-1]).issubset(set(edge.nodes[0].dist_index))

    def belief_propagation(self):
        # TODO: scheduling

        for num_edge, edge in enumerate(self.edges):

            antecedant_edges = self.antecedant_edges[num_edge]

            if not antecedant_edges:
                print(num_edge, edge)
                continue

            # A message from a variable node v to a factor node f
            # is the product of the messages from all other neighboring factor nodes
            if not edge.F_to_V:
                # We iterate implicitly on all the possible values of the variable
                messages = map(lambda e: e.message, antecedant_edges)
                edge.message = reduce(
                    lambda x, y: x + y,
                    messages,
                    np.zeros(edge.nodes[0].nb_states))

            # A message from a factor node f to a variable node v
            # is the product of the factor with messages from all other nodes,
            # marginalized over all variables except the one associated with v
            else:
                if edge.nodes[0].dist == "MTT":
                    continue
                dist = edge.nodes[0].dist
                dist_index = edge.nodes[0].dist_index

#                 edge.message = np.zeros_like(edge.message)
                target_name = edge.nodes[1].name

                # TODO : convert dist_index into a dictionnary
                target_index = dist_index.index(target_name)
                for index_dist, f in np.ndenumerate(dist):
                    messages = map(
                        # index_dist[dist_index.index(e.nodes[0].name)]
                        # is the indice of the state of the marginal conterfactual
                        lambda e: e.message[index_dist[dist_index.index(
                            e.nodes[0].name)]],
                        antecedant_edges
                    )
                    target_state = index_dist[target_index]
                    edge.message[target_state] += (
                        reduce(lambda x, y: x + y, messages, f)
                    )

    def compute_belief_nodes(self):
        self.beliefs = {}
        for name, node in self.dic_nodes.items():
            if node.node_type == "V":
                # We consider all the factor nodes
                incoming_edges = self.incoming_edges[name]
                # Extracting the messages.
                # Implicitly parallel for each state of the variable
                messages = map(lambda e: e.message, incoming_edges)
                beliefs = reduce(lambda x, y: x + y, messages,
                                 np.zeros(node.nb_states))
                self.beliefs[name] = np.log(softmax(beliefs))
                assert np.isclose(np.exp(self.beliefs[name]).sum(), 1)
        return self.beliefs

    def max_likelihood(self):
        max_likelihood = list(map(lambda e: max(e), self.beliefs.values()))
        # maybe 0 > np.zeros()
        return reduce(lambda x, y: x + y, max_likelihood, 0)

    def proba_state(self, states, bal_coef=1):
        """
        states : dictionnary mapping the name of the nodes 
                to the index of the state activivated for this node

        bal_coef : weights of true link. weight of false negatifs compared to false positifs

        returns : log proba
        """
        log_proba = 0
        for name, belief in self.beliefs.items():
            coef = bal_coef if states[name] else 1
            log_proba += coef * belief[states[name]]
        return log_proba

    def __repr__(self):
        print("Use plot method instead.")
        raise NotImplementedError

    def plot(self):
        G = nx.Graph()
        G.add_nodes_from(self.dic_nodes.keys())
        for edge in self.edges:
            node1, node2 = edge.nodes
            G.add_edge(node1.name, node2.name)
        nx.draw(G, with_labels=True)
