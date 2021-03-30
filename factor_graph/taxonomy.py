import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import nevergrad as ng
from tqdm.notebook import tqdm

from graph import Node, Edge, Graph
from processing import process_embedding_from_words


class NoPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def phi_edge(i, j, w, embedding, y=1):
    """
    i and j position of the pair of world in df_cross_embedding
    y = y_ij 1 or 0
    w = numpy array

    Interpretation physique: ?
    """
    if y:
        return np.exp(np.dot(w, embedding[i, j]))
    else:
        return 1


def build_graph(w, words, embedding):
    dic_nodes = {}
    edge_names = []

    # T is a node with special distribution
    # TODO : CHeck TUTTE 1984
    dic_nodes["T"] = Node(name="T", node_type="F", dist="MTT")

    for i, wordi in enumerate(words):
        for j, wordj in enumerate(words):
            # TODO: Rajouter les siblings
            if i == j:
                continue
            name_V = f"V_{wordi}_{wordj}"
            node_V = Node(name_V, "V", nb_states=2)
            dic_nodes[name_V] = node_V
            dist = np.array([phi_edge(i, j, w, embedding, y) for y in [0, 1]])
            name_F = f"F_{wordi}_{wordj}"
            node_F = Node(name_F, "F", dist=dist, dist_index=[name_V])
            dic_nodes[name_F] = node_F
            edge_names.append((name_V, name_F))
            edge_names.append((name_V, 'T'))

    graph = Graph(dic_nodes, edge_names)
    return graph


def compute_belief_from_w(w, words, embedding):
    graph = build_graph(w, words, embedding)
    graph.belief_propagation()
    belief = graph.compute_belief_nodes()
    return belief


def build_state(words, df_graph):
    state = {}
    for i, wordi in enumerate(words):
        for j, wordj in enumerate(words):
            # TODO: Rajouter les siblings
            if i == j:
                continue
            name_V = f"V_{wordi}_{wordj}"
            state[name_V] = df_graph.iloc[i, j]
    return state


def proba_state(w, state, words, embedding, bal_coef=1):
    graph = build_graph(w, words, embedding)
    graph.belief_propagation()
    graph.compute_belief_nodes()
    return graph.proba_state(state, bal_coef=bal_coef)


def gradient_from_w(w, state, words, embedding, eps=1e-3, verbose=True, bal_coef=1):
    """
    Compute the gradient of the log max_likelihood with respect to w.
    output:
        grad: np.array
        likelihood: float
    """
    grad = np.zeros_like(w, dtype=np.float32)
    likelihood = proba_state(w, state, words, embedding, bal_coef=bal_coef)

    for i, _ in enumerate(tqdm(grad, total=len(w), disable=not verbose, leave=False)):
        w_eps = w.copy()
        w_eps[i] += eps
        likelihood_eps = proba_state(w_eps, state, words, embedding, bal_coef=bal_coef)
        grad[i] = (likelihood_eps - likelihood)/eps
    return grad, likelihood


def plot_state_likelihood_and_gradient(state, words, embedding, W_X, W_Y, bal_coef=1):
    # plot offset embedding[0] and feature embedding[1]
    embedding = embedding[:, :, :2].copy()

    @np.vectorize
    def likelihood_from_coordinates(*w_i):
        w = np.array(w_i)
        likelihood = proba_state(w, state, words, embedding, bal_coef=bal_coef)
        return likelihood

    @np.vectorize
    def norm_of_gradient_from_coordinates(*w_i):
        w = np.array(w_i)
        grad, _ = gradient_from_w(w, state, words, embedding, verbose=False, bal_coef=bal_coef)
        return np.linalg.norm(grad)

    def plot_function(W_X, W_Y, fig, ax, func):
        if func == "likelihood":
            L = likelihood_from_coordinates(W_X, W_Y)
            title = 'Log-likelihood of the state as a function of w'
        elif func == "gradient":
            L = norm_of_gradient_from_coordinates(W_X, W_Y)
            title = 'Gradient of the log-likelihood of the graph'
        else:
            raise ValueError(
                "func: {} not in ['likelihood', 'gradient'].".format(func))
        L = L[:-1, :-1]
        left_x, right_x = W_X.min(), W_X.max()
        left_y, right_y = W_Y.min(), W_Y.max()
        left_l, right_l = -np.abs(L).max(), np.abs(L).max()
        left_l, right_l = L.min(), L.max()
        L = ax.pcolormesh(W_X, W_Y, L, cmap='hot', vmin=left_l, vmax=right_l)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("$W_0$: offset", fontsize=14)
        ax.set_ylabel("$W_1$: feature", fontsize=14)
        ax.axis([left_x, right_x, left_y, right_y])
        fig.colorbar(L, ax=ax)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_function(W_X, W_Y, figure, axes[0], "likelihood")
    plot_function(W_X, W_Y, figure, axes[1], "gradient")
    plt.tight_layout()
    plt.show()


def plot_nx_graph(w, words, embedding, verbose=False):
    G = nx.Graph()
    G.add_nodes_from(words)
    belief = compute_belief_from_w(w, words, embedding)
    for i, wordi in enumerate(words):
        for j, wordj in enumerate(words):
            if i == j:
                continue
            name_V = f"V_{wordi}_{wordj}"
            proba_link = np.exp(belief[name_V][1])
            if verbose:
                print(name_V, proba_link)
            if proba_link > 0.5:
                G.add_edge(wordi, wordj)
    nx.draw(G, with_labels=True, pos=nx.spring_layout(G))
    return G


def check_process_embedding_from_words(embedd, words, verbose=False):
    if embedd is None:
        return process_embedding_from_words(words)
    else:
        if verbose:
            print("Embedding already charged")
        return embedd


class TaxonomyModule:
    def __init__(self, nb_features):
        self.w = np.random.randn(nb_features)   # already includes the offset
        self.train_embedd = None
        self.test_embedd = None
        self.nb_features = nb_features
        self.losses = []

    def train(self, train_df_graph, nb_epochs, lr, bal_coef=1, verbose=True):
        """
        if lr = None : uses nevergrad
        """
        train_words = list(train_df_graph.columns)

        with NoPrint():
            self.train_embedd = check_process_embedding_from_words(
                self.train_embedd, train_words)

        train_state = build_state(train_words, train_df_graph)

        if lr is not None:
            self.losses = []
            for epoch in tqdm(range(nb_epochs), disable=not verbose):
                grad, likelihood = gradient_from_w(
                    self.w, train_state, train_words, self.train_embedd,
                    bal_coef=bal_coef, verbose=verbose)
                self.w += lr*grad
                self.losses.append(likelihood)
                if verbose:
                    print("[Epoch {}]: {:.3f}".format(epoch + 1, likelihood))
        else:
            def f(w):
                return - proba_state(
                    w, train_state, train_words, self.train_embedd, bal_coef=bal_coef)
            with NoPrint():
                optimizer = ng.optimizers.NGOpt(
                    parametrization=self.nb_features, budget=nb_epochs, num_workers=4)
            try:
                verbosity = 2 if verbose else 0
                recommendation = optimizer.minimize(f, verbosity=verbosity)
                self.w = recommendation.value
            except KeyboardInterrupt:
                self.w = optimizer.provide_recommendation()


    def f1_score(self, test_df_graph, eps=1e-8):
        test_words = list(test_df_graph.columns)

        with NoPrint():
            self.test_embedd = check_process_embedding_from_words(
                self.test_embedd, test_words)

        belief = compute_belief_from_w(self.w, test_words, self.test_embedd)
        truth_graph = np.array(test_df_graph, dtype=bool)
        inferred_graph = np.zeros(
            (len(test_words), len(test_words)), dtype=bool)
        for i, wordi in enumerate(test_words):
            for j, wordj in enumerate(test_words):
                if i == j:
                    continue
                name_V = f"V_{wordi}_{wordj}"
                proba_link = np.exp(belief[name_V][1])
                if proba_link > 0.5:
                    inferred_graph[i, j] = 1
        tp = np.sum((inferred_graph == 1) & (truth_graph == 1))
        fp = np.sum((inferred_graph == 1) & (truth_graph == 0))
        fn = np.sum((inferred_graph == 0) & (truth_graph == 1))
        precision = tp/(tp+fp+eps)
        recall = tp/(tp+fn+eps)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)
        return f1_score

    def infer_graph(self, test, verbose=False):
        """
        input: either a dataframe df_graph or a list of words
        verbose : output probabilities of each link
        """
        if type(test) is pd.DataFrame:
            test_words = list(test.columns)
        elif type(test) is list:
            test_words = test
        else:
            raise TypeError(
                "input must be either a dataframe or a list of words")

        with NoPrint():
            self.test_embedd = check_process_embedding_from_words(
                self.test_embedd, test_words)

        return plot_nx_graph(self.w, test_words, self.test_embedd, verbose=verbose)
