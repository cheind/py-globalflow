from functools import partial

import networkx as nx

from .mot import EdgeType, FlowDict, FlowGraph, int_to_float


def draw_graph(flowgraph: FlowGraph, ax=None):
    """Draws the graphical representation of the assignment problem."""
    pos = nx.multipartite_layout(flowgraph, align="vertical")

    def _filter_edges(etype: EdgeType):
        edges = flowgraph.edges()
        subedges = [(u, v) for u, v in edges if flowgraph[u][v]["etype"] == etype]
        return subedges

    nx.draw_networkx_edges(
        flowgraph,
        pos,
        _filter_edges(EdgeType.ENTER),
        edge_color="purple",
        connectionstyle="arc3,rad=-0.3",
    )
    nx.draw_networkx_edges(
        flowgraph,
        pos,
        _filter_edges(EdgeType.EXIT),
        edge_color="green",
        ax=ax,
        connectionstyle="arc3,rad=0.3",
    )
    nx.draw_networkx_edges(
        flowgraph,
        pos,
        _filter_edges(EdgeType.OBS),
        edge_color="blue",
        style="dashed",
        ax=ax,
    )
    nx.draw_networkx_edges(
        flowgraph,
        pos,
        _filter_edges(EdgeType.TRANSITION),
        edge_color="black",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        flowgraph,
        pos,
        node_size=600,
        node_color="white",
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_labels(
        flowgraph,
        pos,
        font_size=8,
    )
    i2f = partial(int_to_float, scale=flowgraph.graph["cost_scale"])
    nx.draw_networkx_edge_labels(
        flowgraph,
        pos,
        edge_labels={
            (u, v): f'{i2f(flowgraph[u][v]["weight"]):.2f}'
            for u, v in flowgraph.edges()
        },
        font_size=8,
        font_color="k",
        label_pos=0.5,
        verticalalignment="top",
        ax=ax,
    )


def draw_flowdict(flowgraph: FlowGraph, flowdict: FlowDict, ax=None):
    """Draws the solution of the assignment problem."""
    pos = nx.multipartite_layout(flowgraph, align="vertical")

    edges = flowgraph.edges()
    edges_with_flow = [(u, v) for u, v in edges if flowdict[u][v] > 0]
    nx.draw_networkx_edges(
        flowgraph,
        pos,
        edge_color="gray",
        width=1,
        style="dashed",
        ax=ax,
    )
    nx.draw_networkx_edges(
        flowgraph,
        pos,
        edgelist=edges_with_flow,
        edge_color="green",
        width=2,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        flowgraph,
        pos,
        node_size=600,
        node_color="white",
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_labels(
        flowgraph,
        pos,
        font_size=8,
        ax=ax,
    )
