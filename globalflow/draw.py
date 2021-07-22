import networkx as nx
from .mot import GlobalFlowMOT, Trajectories


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
        # connectionstyle="arc3,rad=-0.3",
    )

    nx.draw_networkx_edges(
        flowmot.graph,
        pos,
        _filter_edges("green"),
        edge_color="green",
        ax=ax,
        # connectionstyle="arc3,rad=0.3",
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
        node_size=400,
        alpha=0.5,
        node_color="white",
        edgecolors="black",
        ax=ax,
    )

    nx.draw_networkx_labels(flowmot.graph, pos, font_size=5)
    nx.draw_networkx_edge_labels(
        flowmot.graph,
        pos,
        edge_labels={
            (u, v): f'{flowmot._i2f(flowmot.graph[u][v]["weight"]):.2f}'
            for u, v in flowmot.graph.edges()
        },
        font_size=6,
        label_pos=0.5,
        font_color="black",
        ax=ax,
    )


def draw_trajectories(flowmot: GlobalFlowMOT, trajectories: Trajectories, ax=None):
    """Draws the solution of the assignment problem."""
    pos = nx.multipartite_layout(flowmot.graph, align="vertical")

    nx.draw_networkx_edges(flowmot.graph, pos, edge_color="black", ax=ax)

    edges = []
    for t in trajectories:
        for i in range(len(t)):
            if i == 0:
                edges.append((GlobalFlowMOT.START_NODE, t[i]))
            else:
                edges.append((t[i - 1].with_tag("v"), t[i]))
            edges.append((t[i], t[i].with_tag("v")))
        edges.append((t[-1].with_tag("v"), GlobalFlowMOT.END_NODE))

    nx.draw_networkx_edges(
        flowmot.graph, pos, edgelist=edges, edge_color="red", width=2, ax=ax
    )

    nx.draw_networkx_nodes(
        flowmot.graph,
        pos,
        node_size=400,
        alpha=0.5,
        node_color="white",
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_labels(flowmot.graph, pos, font_size=5, ax=ax)
