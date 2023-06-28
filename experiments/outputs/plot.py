import argparse
import glob
import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set(
    style="darkgrid",
    rc={
        "figure.figsize": (7.2, 4.45),
        "text.usetex": False,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "font.size": 15,
        "figure.autolayout": True,
        "axes.titlesize": 16,
        "axes.labelsize": 17,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.fontsize": 15,
    },
)
colors = sns.color_palette("colorblind", 4)
# colors = sns.color_palette("Set1", 2)
# colors = ['#FF4500','#e31a1c','#329932', 'b', 'b', '#6a3d9a','#fb9a99']
dashes_styles = cycle(["-", "-.", "--", ":"])
sns.set_palette(colors)
colors = cycle(colors)


def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def plot_df(df, color, xaxis, yaxis, ma=1, label=""):
    df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")  # convert NaN string to NaN value

    mean = df.groupby(xaxis).mean()[yaxis]
    std = df.groupby(xaxis).std()[yaxis]
    if ma > 1:
        mean = moving_average(mean, ma)
        std = moving_average(std, ma)

    x = df.groupby(xaxis)[xaxis].mean().keys().values
    plt.plot(x, mean, label=label, color=color, linestyle=next(dashes_styles))
    plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)

    # plt.ylim([0,200])
    # plt.xlim([40000, 70000])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Plot Traffic Signal Metrics"""
    )
    parser.add_argument("--file", nargs="+", help="Measures files\n")
    parser.add_argument("--folder", help="Folder with output files\n")
    parser.add_argument("-l", nargs="+", default=None, help="File's legends\n")
    parser.add_argument("-t", type=str, default="", help="Plot title\n")
    parser.add_argument("-yaxis", type=str, help="The column to plot.\n")
    parser.add_argument("-xaxis", type=str, default="step", help="The x axis.\n")
    parser.add_argument("-ma", type=int, default=1, help="Moving Average Window.\n")
    parser.add_argument("-sep", type=str, default=",", help="Values separator on file.\n")
    parser.add_argument("-xlabel", type=str, default="Time step (seconds)", help="X axis label.\n")
    parser.add_argument("-ylabel", type=str, default="Total travel time (s)", help="Y axis label.\n")
    parser.add_argument("-output", type=str, default=None, help="PDF output filename.\n")

    args = parser.parse_args()

    if (args.file is None and args.folder is None):
        parser.error("Either file or folder with files must be provided")
    if not (args.file is None):
        output = args.file
    else:
        csv_files = os.listdir(args.folder)
        output = [os.path.join(args.folder, filename) for filename in csv_files if filename.endswith('.csv')]
        print(output)
    
    labels = cycle(args.l) if args.l is not None else cycle([str(i) for i in range(len(output))])

    plt.figure()

    # File reading and grouping
    for file in output:
        main_df = pd.DataFrame()
        for f in glob.glob(file + "*"):
            df = pd.read_csv(f, sep=args.sep)
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df))

        # Plot DataFrame
        plot_df(main_df, xaxis=args.xaxis, yaxis=args.yaxis, label=next(labels), color=next(colors), ma=args.ma)
    plt.legend()
    plt.title(args.t)
    plt.ylabel(args.ylabel)
    plt.xlabel(args.xlabel)
    plt.ylim(bottom=0)

    if args.output is not None:
        plt.savefig(args.output, bbox_inches="tight")

    plt.show()
