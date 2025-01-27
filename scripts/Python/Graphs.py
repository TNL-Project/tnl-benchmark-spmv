import matplotlib.pyplot as plt
import numpy as np

#                                                                        brown
matplotlib_fixed_colors = ["blue", "green", "red", "cyan", "magenta", "#663300"]


def draw_graphs(
    graph_labels,
    graphs,
    xlabel,
    ylabel,
    filename,
    legend_loc="upper right",
    bar="none",
    yscale="linear",
    latex_labels={},
):
    """
    Draw several graphs into one figure
    graph_labels - list of labels of graphs to be drawn
    graphs - dictionary with graphs where key is the label of the graph and value is the data
    xlabel - label of x axis
    ylabel - label of y axis
    filename - name of the output file
    legend_loc - location of the legend
    bar - name of the bar to be drawn. Bar is drawn as a line with value 1.
        It serves for better visualization of speed-up (i.e. where it is larger or smaller than one).
        If bar is set to 'none', no bar is drawn.
    yscale - scale of y axis ('linear' or 'log')
    latex_labels - dictionary with labels in latex format where key is the label of the graph and value is the latex label
    """
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    latexNames = []
    size = 1
    color_idx = 0
    for label in graph_labels:
        if not label in graphs:
            raise RuntimeError(f"Graph {label} not found in graphs")
        t = np.arange(len(graphs[label]))
        if color_idx < len(matplotlib_fixed_colors):
            axs.plot(
                t,
                graphs[label],
                "-o",
                ms=1,
                lw=1,
                color=matplotlib_fixed_colors[color_idx],
            )
            color_idx = color_idx + 1
        else:
            axs.plot(t, graphs[label], "-o", ms=1, lw=1)
        size = len(graphs[label])
        if not latex_labels:
            latexNames.append(label)
        else:
            latexNames.append(latex_labels[label])
    if bar != "none":
        bar_data = np.full(size, 1)
        axs.plot(t, bar_data, "-", ms=1, lw=1.5)
        if bar != "":
            latexNames.append(bar)

    axs.legend(latexNames, loc=legend_loc)
    axs.yaxis.set_label_position("right")
    axs.yaxis.tick_right()
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_yscale(yscale)
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            #"font.sans-serif": ["Helvetica"],
            "font.size": 22,
        }
    )
    plt.savefig(filename)
    plt.close(fig)


def draw_dual_graphs(
    graph_labels,
    graphs,
    xlabel,
    ylabels,
    filename,
    legend_loc="upper right",
    bar="none",
    yscales=["linear","linear"],
    left_y_limits =[0,0],
    right_y_limits=[0,0],
    fig_size=(8, 5),
    latex_labels={},
):
    """
    Draw two graphs into one figure. One has yaxis label on the left and the other on the right.
    graph_labels - list of labels of graphs to be drawn
    graphs - dictionary with graphs where key is the label of the graph and value is the data
    xlabel - label of x axis
    ylabel - label of y axis
    filename - name of the output file
    legend_loc - location of the legend
    bar - name of the bar to be drawn. Bar is drawn as a line with value 1.
        It serves for better visualization of speed-up (i.e. where it is larger or smaller than one).
        If bar is set to 'none', no bar is drawn.
    yscale - scale of y axis ('linear' or 'log')
    latex_labels - dictionary with labels in latex format where key is the label of the graph and value is the latex label
    """
    fig, ax1 = plt.subplots(1, 1, figsize=fig_size)
    ax2 = ax1.twinx()
    latexNames = []
    size = 1
    if not len(graph_labels) == 2:
        raise RuntimeError("Only two graphs are supported")
    if not graph_labels[0] in graphs:
        raise RuntimeError(f"Graph {graph_labels[0]} not found in graphs")
    if not graph_labels[1] in graphs:
        raise RuntimeError(f"Graph {graph_labels[1]} not found in graphs")
    if not len(graphs[graph_labels[0]]) == len(graphs[graph_labels[1]]):
        raise RuntimeError(
            f"Graphs must have the same size ({len(graphs[graph_labels[0]])} != {len(graphs[graph_labels[1]])}"
        )

    t = np.arange(len(graphs[graph_labels[0]]))
    ax1.plot(
        t,
        graphs[graph_labels[0]],
        "-o",
        ms=1,
        lw=1,
        color=matplotlib_fixed_colors[0],
    )
    ax2.plot(
        t,
        graphs[graph_labels[1]],
        "o",
        ms=1,
        lw=1,
        color=matplotlib_fixed_colors[1],
    )

    size = max(len(graphs[graph_labels[0]]), len(graphs[graph_labels[1]]))
    # if not latex_labels:
    #    latexNames.append(graph_labels[0])
    #    latexNames.append(graph_labels[1])
    # else:
    #    latexNames.append(latex_labels[graph_labels[0]])
    #    latexNames.append(latex_labels[graph_labels[1]])

    if bar != "none":
        bar_data = np.full(size, 1)
        ax1.plot(t, bar_data, "-", ms=1, lw=1.5)
        if bar != "":
            latexNames.append(bar)

    if legend_loc != "none":
        ax1.legend(latexNames, loc=legend_loc)
    ax1.set_xlabel(xlabel)
    ax1.xaxis.set_label_position("top")
    ax1.set_ylabel(ylabels[0])
    ax1.set_yscale(yscales[0])
    if left_y_limits[0] != 0 or left_y_limits[1] != 0:
        ax1.set_ylim(left_y_limits[0], left_y_limits[1])
    ax2.set_ylabel(ylabels[1])
    ax2.set_yscale(yscales[1])
    if right_y_limits[0] != 0 or right_y_limits[1] != 0:
        ax2.set_ylim(right_y_limits[0], right_y_limits[1])
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Tahoma"],
            "font.size": 22,
        }
    )
    plt.savefig(filename)
    plt.close(fig)
