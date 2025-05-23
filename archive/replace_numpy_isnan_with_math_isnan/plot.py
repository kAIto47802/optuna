import numpy as np
import plotly.graph_objects as go


def plot_data(
    mean1: dict[str, np.ndarray],
    std1: dict[str, np.ndarray],
    mean2: dict[str, np.ndarray],
    std2: dict[str, np.ndarray],
    n_trials: int,
    n_seeds: int,
    colors: list[str],
    linestyles: dict[str, str],
    names: dict[str, str],
    labels: list[str],
    xlabel: str,
    ylabel: str,
) -> go.Figure:
    # fig, ax = plt.subplots(figsize=(10, 5))
    fig = go.Figure()
    dx = np.arange(n_trials) + 1
    for color, label, mean, std in zip(colors, labels, [mean1, mean2], [std1, std2]):
        for name in mean.keys():
            for show_legend, sign in zip([False, True], [1, -1]):
                fig.add_trace(
                    go.Scatter(
                        x=dx,
                        y=mean[name],
                        mode="lines",
                        name=names[name] + " " + label,
                        line=dict(color=color, dash=linestyles[name], width=2),
                        showlegend=show_legend,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=dx,
                        y=mean[name] + sign * std[name] / np.sqrt(n_seeds),
                        mode="lines",
                        fill="tonexty",
                        line=dict(color=color, width=0),
                        showlegend=False,
                    )
                )
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=800,
        height=500,
        template="simple_white",
        font=dict(family="Computer Modern", size=16),
        legend=dict(
            x=0.01,
            y=0.95,
            xanchor="left",
            yanchor="top",
            bordercolor="black",
            borderwidth=1,
            tracegroupgap=0,
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        mirror=True,
        gridcolor="gray",
        griddash="solid",
        range=[0, None],
    )
    fig.update_yaxes(
        showgrid=True,
        mirror=True,
        gridcolor="gray",
        griddash="solid",
        range=[0, None],
    )

    return fig


if __name__ == "__main__":
    mean1 = np.load("mean1.npz")
    std1 = np.load("std1.npz")
    mean2 = np.load("mean2.npz")
    std2 = np.load("std2.npz")

    experimental_settings1 = np.load("experimental_settings1.npz")
    experimental_settings2 = np.load("experimental_settings2.npz")
    assert experimental_settings1 == experimental_settings2

    # colors = ["#0072B2", "#D55E00"]
    # colors = ["#56B4E9", "#E69F00"]
    colors = ["#0072B2", "#CC79A7"]
    linestyles = {"tpe": "solid", "brute-force": "dash"}
    names = {"tpe": "TPE", "brute-force": "Brute Force"}
    labels = ["(With this PR)", "(Original)"]
    xlabel = "Number of Trials"
    ylabel = "Elapsed Time / s"

    # Plot the data
    fig = plot_data(
        mean1=mean1,
        std1=std1,
        mean2=mean2,
        std2=std2,
        n_trials=experimental_settings1["n_trials"],
        n_seeds=experimental_settings1["n_seeds"],
        colors=colors,
        linestyles=linestyles,
        names=names,
        labels=labels,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    fig.write_image("benchmark_time5.png")
