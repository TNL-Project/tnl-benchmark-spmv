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
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_yscale(yscale)
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
        }
    )
    plt.savefig(filename)
    plt.close(fig)
