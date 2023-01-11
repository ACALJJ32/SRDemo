import matplotlib.pyplot as plt
from colour.utilities import first_item, tstack, normalise_maximum, is_string, as_float_array
from colour.plotting import filter_cmfs, XYZ_to_plotting_colourspace, CONSTANTS_COLOUR_STYLE, filter_RGB_colourspaces
from colour.constants import EPSILON
from colour.models import XYZ_to_xy, xy_to_XYZ, RGB_to_XYZ
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from colour.algebra import normalise_vector
import numpy as np
from threading import Lock, RLock
import itertools
import bisect

constants = CONSTANTS_COLOUR_STYLE
style = {
        'figure.dpi': 100,
        'savefig.dpi': 100,

        # Spine Settings
        'axes.linewidth': constants.geometry.short,
        'axes.edgecolor': constants.colour.dark,

        # Axes Settings
        # 'axes.facecolor': constants.colour.brightest,
        'axes.facecolor': 'black',
        'axes.grid': False,
        'axes.grid.which': 'major',
        'axes.grid.axis': 'both',

        # Lines
        'lines.linewidth': constants.geometry.short * 0.5,
        'lines.markersize': constants.geometry.short * 3,
        'lines.markeredgewidth': constants.geometry.short * 0.75,
    }

lock = Lock()
rlock = RLock()

class Structure(dict):
    def __init__(self, *args, **kwargs):
        super(Structure, self).__init__(*args, **kwargs)
        self.__dict__ = self


def plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
                                                            RGB,
                                                            filename=None,
                                                            colourspace='ITU-R BT.709',
                                                            colourspaces=['ITU-R BT.2020','ITU-R BT.709']):

    # create figure and axes 
    plt.rcParams.update(style)
    width, _ = plt.rcParams['figure.figsize']
    plt.figure(figsize=(width, width))

    # 调整画布，去除留白
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    _, axes = plt.gcf(), plt.gca()
    # end create figure and axes

    # 1. plot spectral locus colours
    spectral_locus_colours = 'RGB'

    cmfs = first_item(filter_cmfs('CIE 1931 2 Degree Standard Observer').values())

    illuminant = CONSTANTS_COLOUR_STYLE.colour.colourspace.whitepoint

    wavelengths = cmfs.wavelengths
    equal_energy = np.array([1 / 3] * 2)

    ij = XYZ_to_xy(cmfs.values, illuminant)
    labels = (390, 460, 470, 480, 490, 500, 510, 520, 540, 560, 580, 600, 620, 700)

    pl_ij = tstack([
        np.linspace(ij[0][0], ij[-1][0], 20),
        np.linspace(ij[0][1], ij[-1][1], 20)
    ]).reshape(-1, 1, 2)
    sl_ij = np.copy(ij).reshape(-1, 1, 2)

    if spectral_locus_colours.upper() == 'RGB':
        spectral_locus_colours = normalise_maximum(
            XYZ_to_plotting_colourspace(cmfs.values), axis=-1)
        XYZ = xy_to_XYZ(pl_ij)
        
        purple_line_colours = normalise_maximum(
            XYZ_to_plotting_colourspace(XYZ.reshape(-1, 3)), axis=-1)
    else:
        purple_line_colours = spectral_locus_colours
    
    for slp_ij, slp_colours in ((pl_ij, purple_line_colours),
                                (sl_ij, spectral_locus_colours)):
        line_collection = LineCollection(
            np.concatenate([slp_ij[:-1], slp_ij[1:]], axis=1),
            colors=slp_colours)
        axes.add_collection(line_collection)

    wl_ij = dict(tuple(zip(wavelengths, ij)))

    for label in labels:
        ij = wl_ij.get(label)

        if ij is None:
            continue

        i, j = ij
        ij = np.array([ij])

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = (wavelengths[index]
                if index < len(wavelengths) else wavelengths[-1])

        dx = wl_ij[right][0] - wl_ij[left][0]
        dy = wl_ij[right][1] - wl_ij[left][1]

        direction = np.array([-dy, dx])

        normal = (np.array([-dy, dx]) if np.dot(
            normalise_vector(ij - equal_energy), normalise_vector(direction)) >
                0 else np.array([dy, -dx]))
        normal = normalise_vector(normal) / 30

        label_colour = (spectral_locus_colours
                        if is_string(spectral_locus_colours) else
                        spectral_locus_colours[index])
        axes.plot(
            (i, i + normal[0] * 0.75), (j, j + normal[1] * 0.75),
            color=label_colour)
    # end plot spectral locus colours

    # 2. plot colour spaces
    colourspaces = filter_RGB_colourspaces(colourspaces).values()

    def xy_to_ij(xy):
        """
        Converts given *CIE xy* chromaticity coordinates to *ij*
        chromaticity coordinates.
        """
        return xy

    x_limit_min, x_limit_max = [-0.1], [0.9]
    y_limit_min, y_limit_max = [-0.1], [0.9]

    samples = np.linspace(0, 1, len(CONSTANTS_COLOUR_STYLE.colour.cycle))
    if isinstance(CONSTANTS_COLOUR_STYLE.colour.map, LinearSegmentedColormap):
        cycle = CONSTANTS_COLOUR_STYLE.colour.map(samples)
    else:
        cycle = getattr(plt.cm, CONSTANTS_COLOUR_STYLE.colour.map)(samples)

    cycle = itertools.cycle(cycle)

    plotting_colourspace = CONSTANTS_COLOUR_STYLE.colour.colourspace

    plot_settings_collection = [{
        'label': '{0}'.format(colourspace.name),
        'marker': 'o',
        'color': next(cycle)[:3]
    } for colourspace in colourspaces]

    for i, colourspace in enumerate(colourspaces):
        plot_settings = plot_settings_collection[i]

        P = np.where(
            colourspace.primaries == 0,
            EPSILON,
            colourspace.primaries)
            
        P = xy_to_ij(P)
        W = xy_to_ij(colourspace.whitepoint)

        P_p = np.vstack([P, P[0]])
        axes.plot(P_p[..., 0], P_p[..., 1], **plot_settings)

        x_limit_min.append(np.amin(P[..., 0]) - 0.1)
        y_limit_min.append(np.amin(P[..., 1]) - 0.1)
        x_limit_max.append(np.amax(P[..., 0]) + 0.1)
        y_limit_max.append(np.amax(P[..., 1]) + 0.1)
    # end plot colour spaces

    # 3. start plot rgb points
    RGB = as_float_array(RGB).reshape(-1, 3)
    scatter_settings = {
        's': 1,
        'c': 'RGB',
        'marker': 'o',
        'alpha': 0.85,
    }
    colourspace = first_item(filter_RGB_colourspaces(colourspace).values())
    XYZ = RGB_to_XYZ(RGB, colourspace.whitepoint, colourspace.whitepoint,
                    colourspace.matrix_RGB_to_XYZ)  
    ij = XYZ_to_xy(XYZ, colourspace.whitepoint)  
    RGB = normalise_maximum(XYZ_to_plotting_colourspace(XYZ, colourspace.whitepoint), axis=-1)
    scatter_settings['c'] = RGB
    axes.scatter(ij[..., 0], ij[..., 1], **scatter_settings)
    # end plot rgb points

    # if filename is not None:  # use opencv or PIL
    canvas = plt.gca().figure.canvas
    canvas.draw()
    color_data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(640,640,-1)
    
    return color_data