import networkx as nx
from .mot import FlowDict, GlobalFlowMOT


def draw_graph(flowmot: GlobalFlowMOT, ax=None):
    """Draws the graphical representation of the assignment problem."""
    pos = nx.multipartite_layout(flowmot.graph, align="vertical")

    def _filter_edges(color):
        edges = flowmot.graph.edges()
        subedges = [(u, v) for u, v in edges if flowmot.graph[u][v]["color"] == color]
        return subedges

    nx.draw_networkx_edges(
        flowmot.graph,
        pos,
        _filter_edges("purple"),
        edge_color="purple",
        connectionstyle="arc3,rad=-0.3",
    )
    nx.draw_networkx_edges(
        flowmot.graph,
        pos,
        _filter_edges("green"),
        edge_color="green",
        ax=ax,
        connectionstyle="arc3,rad=0.3",
    )
    nx.draw_networkx_edges(
        flowmot.graph,
        pos,
        _filter_edges("blue"),
        edge_color="blue",
        style="dashed",
        ax=ax,
    )
    nx.draw_networkx_edges(
        flowmot.graph,
        pos,
        _filter_edges("black"),
        edge_color="black",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        flowmot.graph,
        pos,
        node_size=600,
        node_color="white",
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_labels(
        flowmot.graph,
        pos,
        font_size=8,
    )
    nx.draw_networkx_edge_labels(
        flowmot.graph,
        pos,
        edge_labels={
            (u, v): f'{flowmot._i2f(flowmot.graph[u][v]["weight"]):.2f}'
            for u, v in flowmot.graph.edges()
        },
        font_size=8,
        font_color="k",
        label_pos=0.5,
        verticalalignment="top",
        ax=ax,
    )


def draw_flowdict(flowmot: GlobalFlowMOT, flowdict: FlowDict, ax=None):
    """Draws the solution of the assignment problem."""
    pos = nx.multipartite_layout(flowmot.graph, align="vertical")

    edges = flowmot.graph.edges()
    edges_with_flow = [(u, v) for u, v in edges if flowdict[u][v] > 0]
    nx.draw_networkx_edges(
        flowmot.graph,
        pos,
        edge_color="gray",
        width=1,
        style="dashed",
        ax=ax,
    )
    nx.draw_networkx_edges(
        flowmot.graph,
        pos,
        edgelist=edges_with_flow,
        edge_color="green",
        width=2,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        flowmot.graph,
        pos,
        node_size=600,
        node_color="white",
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_labels(
        flowmot.graph,
        pos,
        font_size=8,
        ax=ax,
    )
