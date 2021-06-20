from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns


ticksize = 5
markersize = 6
markeredgewidth = 1.
figure_width = 7.08
labelsize = 10
titlesize = 12
fontsize = 10
ticklabelsize = 8
linewidth = 1.5
# figsize = (figure_width, figure_width*0.75)
figsize = [6.85039, 5.1375]


def set_figuresize():
    """
Set the size of a figure.
Default is (180mm, 180,,*0.75)
    """

    params = {"figure.figsize": figsize}

    plt.rcParams.update(params)


def set_legendstyle():
    """
Set legend options.
    """
    params = {
        "legend.numpoints": 1,
        "legend.fontsize": fontsize,
        "legend.handlelength": 2.2,
        "legend.borderpad": 0.5,
        "legend.framealpha": 1,
        "legend.fancybox": True
    }
    plt.rcParams.update(params)


def set_font():
    """Set font options."""
    params = {"text.antialiased": True,
              "font.family": "sans-serif",
              "font.sans-serif": "Arial",
              "font.weight": "normal",
              }

    plt.rcParams.update(params)


def set_latex_font():
    """
Set font options. Note, uses latex.
    """
    params = {"text.usetex": True,
              "text.latex.preamble": "\\usepackage{amsmath}, \\usepackage{amssymb}",
              "text.antialiased": True,
              #   "font.family": "lmodern",
              #   "font.weight": "normal"
              }

    plt.rcParams.update(params)

def set_linestyle():
    """
Set line style options
    """

    params = {"lines.linewidth": linewidth,
              "lines.linestyle": "solid",
              "lines.marker": None,
              "lines.antialiased": True,
              "lines.markersize": markersize,
              "lines.markeredgewidth": markeredgewidth,
              "lines.antialiased": True
             }

    plt.rcParams.update(params)


def set_tickstyle():
    """
Set tick style options
    """

    params = {'xtick.direction': 'out',
              'ytick.direction': 'out',
              "xtick.major.size": ticksize,
              "ytick.major.size": ticksize,
              "xtick.labelsize": ticklabelsize,
              "ytick.labelsize": ticklabelsize,
              }

    plt.rcParams.update(params)


def set_axestyle():
    """
Set tick style options
    """

    params = {"axes.titlesize": titlesize,
              "axes.labelsize": labelsize,
              "axes.labelcolor": "black",
              "axes.linewidth": 1,
              "axes.spines.right": False,
              "axes.spines.top": False,
              "axes.unicode_minus": True
            }

    plt.rcParams.update(params)


def reset_style():
    plt.rcdefaults()


def set_grid(ax, bgcolor="#EAEAF2", linecolor="w", linestyle="-", linewidth=1.3):
    """
Set background color and grid line options

Parameters
----------
Required arguments
ax : matplotlib.axis
    axis object where the background color and grid line options are set
bgcolor : str
    background color
    Default is "#EAEAF2"
linecolor : str
    linecolor color
    Default is "w"
linestyle : str
    linestyle
    Default is "-"
linewidth : float
    linewidth
    Default is 1.3
    """
    ax.set_axis_bgcolor(bgcolor)
    ax.set_axisbelow(True)
    ax.grid(True, color=linecolor, linestyle=linestyle, linewidth=linewidth,
            zorder=-10)


def spines_color(ax, edges={"top": "None", 
                            "right": "None"}):
    """
Set spines color

Parameters
----------
Required arguments
ax : matplotlib.axis
    axis object where the spine colors are set
edges : dictionary
    edges as keys with colors as key values
    """

    for edge, color in edges.items():
        ax.spines[edge].set_edgecolor(color)


def remove_ticks(ax):
    """
Remove ticks from right y axis and top x axis

Parameters
----------
Required arguments
ax : matplotlib.axis
    axis object where the ticks are removed
    """
    ax.tick_params(axis="x", which="both", bottom=True, top=False,
                   labelbottom=True, labelcolor="black",
                   labelsize=labelsize)

    ax.tick_params(axis="y", which="both", right=False, left=True,
                   labelleft=True, labelcolor="black",
                   labelsize=labelsize)


def get_colormap_tableu20(color=None):
    """
Get the Tableau20 colormap.

Parameters
----------
Optinal arguments
color : int
    returns #color from the Tableau20 colormap

Returns
----------
color : rgb tuple
    if color != None, then this color is returned
color : list of rgb tuples
    if color == None, then a list of rgb tuples is returned
    """

    tableau20 = [(31, 119, 180), (14, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    if color is None:
        return tableau20
    else:
        color = color % len(tableau20)

        return tableau20[color]



def get_colormap(palette="hls", nr_colors=6):
    """
Get the color palette from seaborn

Parameters
----------
Optional arguments
palette : str
    the color palette to get
    Default is "hsl"
nr_colors : int
    the number of colors in the seaborn color palette.
    Default is 6


Returns
----------
color : list of rgb tuples
    """
    return sns.color_palette(palette, nr_colors)



def get_current_colormap():
    """
Get the current color palette

Returns
----------
color : list of rgb tuples
    """
    return sns.color_palette()


def set_title(title, ax=None, **kwargs):
    """
Set the title

Parameters
----------
Required arguments

title : str
   Title of the plot

Optional arguments

ax : matplotlib.axis object
    Axis object where to put the title
**kwargs : named arguments
    Matplotlib title() and set_title() arguments
    """
    if ax is None:
        plt.title(title, fontsize=titlesize, **kwargs)
    else:
        ax.set_title(title, fontsize=titlesize, **kwargs)


def set_xlabel(xlabel, ax=None, color="black", lblsize=labelsize, **kwargs):
    """
Set the x label

Parameters
----------
Required arguments

xlabel : str
   xlabel of the plot

Optional arguments

ax : matplotlib.axis object
    Axis object where to put the xlabel
color : matplotlib accepted color
    color of the label
    Default is "black"
**kwargs : named arguments
    Matplotlib xlabel() and set_xlabel() arguments
    """
    if ax is None:
        plt.xlabel(xlabel, fontsize=lblsize, color=color, **kwargs)
    else:
        ax.set_xlabel(xlabel, fontsize=lblsize, color=color, **kwargs)


def set_ylabel(ylabel, ax=None, color="black", lblsize=labelsize, **kwargs):
    """
Set the y label

Parameters
----------
Required arguments

ylabel : str
   ylabel of the plot

Optional arguments

ax : matplotlib.axis object
    Axis object where to put the ylabel
color : matplotlib accepted color
    color of the label
    Default is "black"
**kwargs : named arguments
    Matplotlib ylabel() and set_ylabel() arguments
    """
    if ax is None:
        plt.ylabel(ylabel, fontsize=lblsize, color=color, **kwargs)
    else:
        ax.set_ylabel(ylabel, fontsize=lblsize, color=color, **kwargs)


def set_style(style="seaborn-darkgrid", nr_colors=6, palette="hls", custom=True):
    """
Set the style of a plot

Optional arguments

style : str
    ["classic" | "dark_background" | "seaborn-pastel" |
    "seaborn" | "seaborn-deep" | "seaborn-colorblind" |
    "bmh" | "seaborn-white" | "seaborn-dark" | "seaborn-poster" |
    "seaborn-ticks" | "seaborn-bright" | "seaborn-whitegrid" |
    "seaborn-notebook" | "fivethirtyeight" | "seaborn-muted" |
    "grayscale" | "ggplot" | "seaborn-talk" | "seaborn-darkgrid" |
    "seaborn-dark-palette" | "seaborn-paper"]
    which matplotlib style to use as base.
    Default is "seaborn-darkgrid"
nr_colors : int
    the number of colors in the seaborn color palette.
    Default is 6
palette : hls | husl | matplotlib colormap | seaborn color palette
    Set the matplotlib color cycle using a seaborn palette.
    Available seaborn palette names:
        deep | muted | bright | pastel | dark | colorblind
    Other options:
        hls | husl | any named matplotlib palette | list of colors
    Matplotlib pallettes can be specified as reversed palettes by appending "_r"
    to the name or as dark palettes by appending "_d" to the name.
    (These options are mutually exclusive, but the resulting list of colors
    can also be reversed).
    Default is "hsl"
custom : bool
    If custom style should be used in addition to the standard style.
    Default is True
    """
    plt.style.use(style)
    sns.set_palette(palette, n_colors=nr_colors)


    if custom:
        set_font()
        set_linestyle()
        set_tickstyle()
        set_legendstyle()
        set_figuresize()
        set_axestyle()




def create_figure(style="seaborn-darkgrid", nr_colors=6, palette="hls", custom_style=True):
    """
Create a new figure

Optional arguments

style : str
    ["classic" | "dark_background" | "seaborn-pastel" |
    "seaborn" | "seaborn-deep" | "seaborn-colorblind" |
    "bmh" | "seaborn-white" | "seaborn-dark" | "seaborn-poster" |
    "seaborn-ticks" | "seaborn-bright" | "seaborn-whitegrid" |
    "seaborn-notebook" | "fivethirtyeight" | "seaborn-muted" |
    "grayscale" | "ggplot" | "seaborn-talk" | "seaborn-darkgrid" |
    "seaborn-dark-palette" | "seaborn-paper"]
    which matplotlib style to use as base.
    Default is "seaborn-darkgrid"
nr_colors : int
    the number of colors in the seaborn color palette.
    Default is 6
palette : hls | husl | matplotlib colormap | seaborn color palette
    Set the matplotlib color cycle using a seaborn palette.
    Available seaborn palette names:
        deep | muted | bright | pastel | dark | colorblind
    Other options:
        hls | husl | any named matplotlib palette | list of colors
    Matplotlib palettes can be specified as reversed palettes by appending "_r"
    to the name or as dark palettes by appending "_d" to the name.
    (These options are mutually exclusive, but the resulting list of colors
    can also be reversed).
    Default is "hsl"

Returns
----------
ax : matplotlib.axis object
    """

    plt.close()

    set_style(style=style, nr_colors=nr_colors, palette=palette, custom=custom_style)

    plt.figure()
    ax = plt.subplot(111)

    return ax



def set_legend(labels, ax=None):

    if ax is None:
        ax = plt.gca()

    color = ax.get_facecolor()

    legend = plt.legend(labels)

    #frame = legend.get_frame()
    #frame.set_facecolor(color)


def prettyPlot(x=[], y=None,
               title="",
               xlabel="",
               ylabel="",
               color=None,
               style="classic",
               custom_style=True,
               palette="hls",
               nr_colors=6,
               ax=None,
               new_figure=True,
               zorder=3,
               yerr=None,
               xerr=None,
               ecolor=None,
               labelfontsize=labelsize,
               ffontsize=fontsize,
               capsize=5,
               capthick=2,
               **kwargs):
    """
prettyPlot

Creates pretty line plots, just a wrapper around matplotlib and seaborn.
Customizing several matplotlib options

Parameters
----------
Required arguments

x : list | array
   x values to plot

Optional arguments

y : list | array
   y values to plot
title : str
    Title of the plot.
    Default is ""
xlabel : str
    Xlabel of the plot.
    Default is ""
ylabel : str
    Ylabel of the plot.
    Default is ""
color : int | matplotlib color arg
    Color of the line. If given as a int uses #color from the current colormap.
    Else it is a matplotlib color argument.
style : str
    ["classic" | "dark_background" | "seaborn-pastel" |
    "seaborn" | "seaborn-deep" | "seaborn-colorblind" |
    "bmh" | "seaborn-white" | "seaborn-dark" | "seaborn-poster" |
    "seaborn-ticks" | "seaborn-bright" | "seaborn-whitegrid" |
    "seaborn-notebook" | "fivethirtyeight" | "seaborn-muted" |
    "grayscale" | "ggplot" | "seaborn-talk" | "seaborn-darkgrid" |
    "seaborn-dark-palette" | "seaborn-paper"]
    which matplotlib style to use as base.
    Default is "seaborn-darkgrid"
custom_style : bool
    If custom style should be used in addition to the standard style.
    Default is True
palette : hls | husl | matplotlib colormap | seaborn color palette
    Set the matplotlib color cycle using a seaborn palette.
    Available seaborn palette names:
        deep, muted, bright, pastel, dark, colorblind
    Other options:
        hls, husl, any named matplotlib palette, list of colors
    Matplotlib palettes can be specified as reversed palettes by appending "_r"
    to the name or as dark palettes by appending "_d" to the name.
    (These options are mutually exclusive, but the resulting list of colors
    can also be reversed).
    Default is "hsl"
nr_colors : int
    the number of colors to be used with the seaborn color palette
    #colors different colors
ax : matplotlib.axis object
    Axis object where to plot
    Default is None
new_figure : bool
    If a new figure should be made, or if the plot should be made
    on top of the last existing plot.
    Default is True.
linestyle: str
    ["solid" | "dashed", "dashdot", "dotted" | (offset, on-off-dash-seq) | "-" | "--" | "-." | ":" | "None" | " " | ""]
linewidth : int
    width of the plotted lines.
    Default is 2
marker : accepted matplotlib marker
    marker for each plot point
xerr/yerr : scalar or array-like, shape(n,1) or shape(2,n), optional
    If a scalar number, len(N) array-like object,
    or an Nx1 array-like object, errorbars are drawn at +/-value relative to the
    data.
    If a sequence of shape 2xN, errorbars are drawn at -row1 and +row2 relative
    to the data.
    Default is None.
ecolor : int | matplotlib color arg
    Color of the line. If given as a int uses #color from the current colormap.
    Else it is a matplotlib color argument.
    If None, use the color of the line connecting the markers.
    Default is None.
capsize : int
    The length of the error bar caps in points;
    if None, it will take the value from errorbar.capsize rcParam.
    Default is 5.
capthick : int
    Controls the thickness of the error bar cap in points.
    Default is 2.
**kwargs : named arguments
    arguments sent directly to matplotlib plot(**kwargs)

Returns
----------
ax : matplotlib.axis object
    """
    global labelsize
    global fontsize
    if labelfontsize != labelsize:
        labelsize = labelfontsize
    if ffontsize != fontsize:
        fontsize = ffontsize

    set_style(style, nr_colors=nr_colors, palette=palette, custom=custom_style)

    if ax is None:
        if new_figure:
            ax = create_figure(style=style,
                               nr_colors=nr_colors,
                               palette=palette)
        else:
            ax = plt.gca()


    if len(x) == 0:
        return ax


    # spines_color(ax)
    if custom_style:
        remove_ticks(ax)

    set_title(title, ax)
    set_xlabel(xlabel, ax, lblsize=labelfontsize)
    set_ylabel(ylabel, ax, lblsize=labelfontsize)

    if y is None:
        y = x
        x = range(len(y))


    if color is not None:
        if isinstance(color, int):
            colors = sns.color_palette()
            color = colors[color]

    p = ax.plot(x, y, color=color, zorder=zorder, **kwargs)

    if xerr is not None or yerr is not None:
        # Get the last plotted color
        color = p[-1].get_color()
        if ecolor is not None:
            ecolors = sns.color_palette()
            ecolor = ecolors[ecolor]
        else:
            ecolor = color
        print("E-Color {} with type {}".format(ecolor, type(ecolor)))

        ax.errorbar(x, y,
                    color=color,
                    zorder=zorder,
                    xerr=xerr,
                    yerr=yerr,
                    ecolor=ecolor,
                    capsize=capsize,
                    capthick=capthick,
                    **kwargs)



    if custom_style:
        ax.yaxis.offsetText.set_fontsize(labelfontsize)
        ax.yaxis.offsetText.set_color("black")
        # ax.ticklabel_format(useOffset=False)


    # ax.set_xlim([min(x), max(x)])
    # ax.set_ylim([min(y), max(y)])

    return ax




# TODO updated doc string
def prettyBar(x, error=None,
              index=None,
              color=None,
              title="",
              xlabels=[],
              xticks=None,
              ylabel="",
              custom_style=True,
              width=0.2,
              linewidth=0,
              ax=None,
              new_figure=True,
              style="seaborn-dark",
              palette="hls",
              nr_colors=6,
              align="center",
              labelfontsize=labelsize,
              ffontsize=fontsize,
              error_kw={"lw": 2,
                        "capsize": 10,
                        "capthick": 2},
              **kwargs):
    """
Creates pretty bar plots, just a wrapper around matplotlib and seaborn.
Customizing several matplotlib options

Parameters
----------
Required arguments

x : list | array
   x values to plot

Optional arguments

error : list | array
    error for each x value
index : list | array
    x position of each bar
color : int
    Color of the line, given as a int.
    Uses #color from the current colormap
title : str
    Title of the plot.
    Default is ""
xlabels : list
    list of xlabels for the plot.
    Default is []
xticks : list | array
    position of each x label
ylabel : str
    ylabel of the plot.
    Default is ""
custom_style : bool
    If custom style should be used in addition to the standard style.
    Default is True
width : int
    width of each bar.
    Default is 0.2
linewidth : int
    width of line around each bar.
    Default is 0
ax : matplotlib.axis object
    Axis object where to plot
    Default is None
new_figure : bool
    If a new figure should be made, or if the plot should be made
    ontop of the last existing plot.
    Default is True.
style : str
    ["classic" | "dark_background" | "seaborn-pastel" |
    "seaborn" | "seaborn-deep" | "seaborn-colorblind" |
    "bmh" | "seaborn-white" | "seaborn-dark" | "seaborn-poster" |
    "seaborn-ticks" | "seaborn-bright" | "seaborn-whitegrid" |
    "seaborn-notebook" | "fivethirtyeight" | "seaborn-muted" |
    "grayscale" | "ggplot" | "seaborn-talk" | "seaborn-darkgrid" |
    "seaborn-dark-palette" | "seaborn-paper"]
    which matplotlib style to use as base.
    Default is "seaborn-dark"
palette : hls | husl | matplotlib colormap | seaborn color palette
    Set the matplotlib color cycle using a seaborn palette.
    Available seaborn palette names:
        deep, muted, bright, pastel, dark, colorblind
    Other options:
        hls, husl, any named matplotlib palette, list of colors
    Matplotlib palettes can be specified as reversed palettes by appending "_r"
    to the name or as dark palettes by appending "_d" to the name.
    (These options are mutually exclusive, but the resulting list of colors
    can also be reversed).
    Default is "hsl"
nr_colors : int
    The number of colors to be used with the seaborn color palette
error_kw : dict
    Dictionary of kwargs to be passed to errorbar method.
    ecolor and capsize may be specified here rather than as independent kwargs.
**kwargs : unpacked dict
    arguments sent directly to matplotlib.bar(**kwargs)

Returns
----------
ax : matplotlib ax Object
    """

    global labelsize
    if labelfontsize != labelsize: 
        labelsize = labelfontsize
    global fontsize
    if ffontsize != fontsize:
        fontsize = ffontsize
    set_style(style, nr_colors=nr_colors, palette=palette, custom=custom_style)

    if ax is None:
        if new_figure:
            plt.figure()

            ax = plt.subplot(111)
        else:
            ax = plt.gca()

    spines_color(ax)
    remove_ticks(ax)

    if index is None:
        try:
            index = np.arange(len(x))
        except TypeError:
            index = [0]

    if xticks is None:
        xticks = index


    if error_kw is None:
        error_kw = dict(lw=2, capsize=10, capthick=2)

    if color is None:
        colors = sns.color_palette()
    elif isinstance(color, int):
        colors = sns.color_palette()[color]

    ax.bar(index, x, yerr=error, color=colors, width=width,
           align=align, linewidth=linewidth, error_kw=error_kw,
           **kwargs)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=0)

    # ax.set_xlim([min(x), max(x)])
    # ax.set_ylim([min(y), max(y)])

    set_title(title, ax)
    set_ylabel(ylabel, ax, lblsize=labelsize)


    return ax
