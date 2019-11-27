import sys
import warnings
from collections import defaultdict
from itertools import product, combinations

import numpy as np
import pandas as pd
import scipy.stats
from networkx import Graph
from plotly.graph_objs import Scatter, Figure, Layout
from scipy.stats._continuous_distns import _distn_names
from scipy.stats.kde import gaussian_kde

from lyner.click_extras import Pipe

idx = pd.IndexSlice


class Density(Scatter):
    def __init__(self, *args, kde=gaussian_kde, bw_method=None, **kwargs):
        orientation = kwargs.pop('orientation', 'v')
        super(Density, self).__init__(*args, **kwargs)
        x = kwargs.get('x', kwargs.get('y', None))
        y = kwargs.get('y', kwargs.get('x', None))
        x = x[~(np.isnan(x) | np.isinf(x))]
        y = y[~(np.isnan(y) | np.isinf(y))]
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        data = x
        if 'x' in kwargs and 'y' in kwargs:
            data = np.vstack([data, y])
            xpos, ypos = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xpos.ravel(), ypos.ravel()])
        else:
            xpos = np.linspace(xmin, xmax, 100)
            positions = xpos

        kernel = kde(data, bw_method=bw_method)

        Z = np.reshape(kernel(positions).T, xpos.shape)
        if orientation == 'h':
            self.x = Z
            self.y = xpos
            self.fill = 'tozerox'
        else:
            self.x = xpos
            self.y = Z
            self.fill = 'tozeroy'
        self.mode = 'lines'


# mostly taken from by https://gist.github.com/arose13/bc6eb9e6b76e8bd940eefd7a0989ac81
def find_best_fit_distribution(data, distributions=None, criterion='sse', nbins=None):
    if nbins is None:
        nbins = estimate_n_bins(data)
    y, x = np.histogram(data, bins=nbins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    # Distributions to check
    dist_names = set(_distn_names) - {'vonmises', 'digamma', 'dweibull'}
    distributions = [getattr(scipy.stats, dist_name, None) for dist_name in
                     dist_names] if distributions is None else distributions

    # Best holders
    best_distributions = {'aicc': scipy.stats.norm, 'sse': scipy.stats.norm}
    best_params = {'aicc': (0.0, 1.0), 'sse': (0.0, 1.0)}
    best_criterionvalue = {'aicc': np.inf, 'sse': np.inf}

    def aicc(distr, args, loc, scale, data, *_):
        llh = np.sum(distr.logpdf(data, *arg, loc=loc, scale=scale))
        print(f"{args} {loc} {scale}")
        k = len(args) + (1 if loc else 0) + (1 if scale else 0)
        aic = 2 * k - 2 * llh
        aicc = aic + 2 * k * (k + 1) / (n - k - 1)
        return aicc

    def sse(distr, args, loc, scale, data, x, y, *_):
        pdf = distr.pdf(x, *args, loc=loc, scale=scale)
        sse = np.sum(np.power(y - pdf, 2.0))
        return sse

    crit_calculation = {'aicc': aicc, 'sse': sse}

    for distribution in distributions:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                params = distribution.fit(data)

                arg, loc, scale = params[:-2], params[-2], params[-1]

                value = crit_calculation[criterion](distribution, arg, loc, scale, data, x, y)
                if best_criterionvalue[criterion] > value > -np.inf:
                    best_distributions[criterion] = distribution
                    best_params[criterion] = params
                    best_criterionvalue[criterion] = value
        except NotImplementedError:
            pass
    return best_distributions[criterion], best_params[criterion], best_criterionvalue[criterion]


def estimate_n_bins(x):
    """
    Uses the Freedman Diaconis Rule for generating the number of bins required
    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    Bin Size = 2 IQR(x) / (n)^(1/3)
    """
    from scipy.stats import iqr

    x = np.asarray(x)
    hat = 2 * iqr(x, nan_policy='omit') / (np.power(x.shape[0], 1.0 / 3))
    n_bins = int(np.sqrt(x.shape[0])) if hat == 0 else int(np.ceil((np.nanmax(x) - np.nanmin(x)) / hat))
    if not n_bins or np.isnan(n_bins) or n_bins <= 0:
        n_bins = x.shape[0] // 2
    return n_bins


def _mk_networkx_figure(G: Graph, pos, use_weights=True):
    nodelist = list(G.nodes())
    edgelist = list(G.edges())
    node_xy = np.asarray([pos[v] for v in nodelist])
    edge_xy = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
    if use_weights:
        weights = np.asarray([G.get_edge_data(e[0], e[1])['weight'] for e in edgelist])
        weights = (weights - np.min(weights) + 0.05) / (np.max(weights) - np.min(weights))
    else:
        weights = np.zeros_like(edge_xy)
    edge_trace = Scatter(
        x=[],
        y=[],
        text=[],
        line=dict(width=0.0, color='rgba(0,0,0,0)') if use_weights else dict(width=1.0, color='rgba(0,0,0,255)'),
        hoverinfo='text',
        mode='lines')
    shapes = []

    for (((x0, y0), (x1, y1)), w) in zip(edge_xy, weights):
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        if use_weights:
            edge_trace['text'] += (f'{w}',)
            shapes.append(dict(type='line',
                               x0=x0, x1=x1, y0=y0, y1=y1,
                               line=dict(width=w, color=f'rgba(0, 0, 0, {w})'),
                               layer='below'))

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            # showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            # colorscale='YlGnBu',
            # reversescale=True,
            # color=[],
            size=10,
            # colorbar=dict(
            #     thickness=15,
            #     title='Node Connections',
            #     xanchor='left',
            #     titleside='right'
            # ),
            line=dict(width=2)))

    for i, (x, y) in enumerate(node_xy):
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([nodelist[i]])

    return Figure(data=[edge_trace, node_trace],
                  layout=Layout(
                      showlegend=False,
                      hovermode='closest',
                      shapes=shapes,
                      # margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                  ))


class AutoDist(Scatter):
    def __init__(self, criterion='sse', nbins=None, distributions=None, *args, **kwargs):
        super(AutoDist, self).__init__(*args, **kwargs)
        data = kwargs.pop('x', None)
        n = data.shape[0] if hasattr(data, 'shape') else len(data)

        best_distribution, best_param, best_critvalue = find_best_fit_distribution(data, criterion=criterion,
                                                                                   nbins=nbins,
                                                                                   distributions=distributions)
        print(f"Using {best_distribution.name} with params {best_param}, ({criterion} = {best_critvalue})",
              file=sys.stderr)

        args, loc, scale = best_param[:-2], best_param[-2], best_param[-1]
        self.x = np.linspace(np.min(data), np.max(data), np.max([n, nbins * 2]))
        self.y = best_distribution.pdf(self.x, *args, loc=loc, scale=scale) if args else best_distribution.pdf(self.x,
                                                                                                               loc=loc,
                                                                                                               scale=scale)
        self.mode = 'lines'
        self.fill = 'tozeroy'
        self.customdata = (best_distribution, best_param)


def _figure_add_changes(pipe: Pipe, fig):
    from plotly.graph_objs import Histogram, XAxis, Figure, Layout, Bar
    matrix = pipe.matrix
    chngs = pipe.changes
    values = chngs[chngs.columns[0]].values
    values = np.array(values, dtype=np.float)
    minvalue = np.nanmin(values)
    i = 1
    while minvalue <= -np.inf:
        minvalue = np.nanpercentile(values, i)
        i += 1
    values[values == -np.inf] = minvalue
    maxvalue = np.nanmax(values)
    i = 99
    while maxvalue >= np.inf:
        maxvalue = np.nanpercentile(values, i)
        i -= 1
    values[values == np.inf] = maxvalue
    values[values == np.nan] = 0

    values = np.nan_to_num(values)
    ha, hb = (0, 90)
    hues = (hb - ha) * (values - minvalue) / (maxvalue - minvalue) + ha
    sa, sb = (10, 100)
    saturations = (sb - sa) * (values - minvalue) / (np.max(values) - minvalue) + sa
    colors = [f'hsl({hb - v}, {int(s)}%, 60%)' for v, s in zip(hues, saturations)]
    hist = Bar(x=values, y=list(map(str, matrix.genes)),
               xaxis='x3',
               yaxis='y',
               orientation='h',
               opacity=0.66,
               marker=dict(color=colors),
               )
    fig['data'].append(hist)
    fig['layout']['xaxis'] = XAxis(side='left', domain=[0, 0.85])
    fig['layout']['xaxis3'] = XAxis(side='right', domain=[0.86, 1])
    chngs['cluster'] = pipe.matrix.index.get_level_values(0)
    histdata = []
    for g in chngs.groupby(by='cluster'):
        cluster, data = g
        histdata.append(data[data.columns[0]].values[0])

    histdata = np.asarray(histdata, dtype=float)
    histdata = histdata[~(np.isinf(histdata) | np.isnan(histdata))]  # remove nans

    histdata = np.log(histdata)
    histdata = histdata[~(np.isinf(histdata) | np.isnan(histdata))]  # remove -np.inf (log(0))

    nbins = estimate_n_bins(histdata) * 2
    hist = Histogram(x=histdata, histnorm='probability density', nbinsx=nbins)
    dist = AutoDist(x=histdata, nbins=nbins, criterion='sse')
    distr, distr_params = dist.customdata
    dist.customdata = None
    p_low = distr.ppf(0.025, *distr_params)
    p_high = distr.ppf(0.975, *distr_params)
    changes_fig = Figure(data=[hist, dist],
                         layout=Layout(shapes=
                                       [dict(type='line', x0=p_low, x1=p_low, y0=0, y1=1, line=dict(dash='dash')),
                                        dict(type='line', x0=p_high, x1=p_high, y0=0, y1=1,
                                             line=dict(dash='dash'))]))
    return changes_fig


def _figure_get_estimates(pipe):
    from plotly.graph_objs import Histogram, Scatter, XAxis, YAxis, Figure, Layout
    groups = np.unique(pipe.estimate.columns.get_level_values(0))
    param_names = np.unique(pipe.estimate.columns.get_level_values(1))
    from plotly import tools
    # FIXME: prone to fail for len(groups) not in scales (or if 'Set1' not in scales[len(groups)])
    # colors = {group: color for group, color in zip(groups, cl.scales[f'{len(groups)}']['qual']['Set1'])}
    colors = defaultdict(None)
    colors['all'] = None
    if len(param_names) > 2:
        param_tuples = list(product(param_names, param_names))
        n = len(param_tuples)
        nn = int(np.ceil(np.sqrt(n)))
        supplementary_fig = tools.make_subplots(rows=nn, cols=nn,
                                                subplot_titles=[(f'{a} vs {b}' if a != b else f'{a}')
                                                                for a, b in param_tuples],
                                                print_grid=False)
        for i, pt in enumerate(param_tuples):
            a, b = pt
            row, col = divmod(i, nn)

            if a == b:
                traces = [Histogram(x=pipe.estimate.loc[:, idx[group, a]].values.ravel(),
                                    marker=dict(color=colors[group]),
                                    name=group,
                                    legendgroup=group,
                                    showlegend=False)
                          for group in groups]
            else:
                traces = [Scatter(x=pipe.estimate.loc[:, idx[group, a]].values.ravel(),
                                  y=pipe.estimate.loc[:, idx[group, b]].values.ravel(),
                                  mode='markers',
                                  marker=dict(color=colors[group]),
                                  name=group,
                                  legendgroup=group,
                                  showlegend=row == 0 and col == 1)
                          # show only a single legend for all plots (with the same groups)
                          for group in groups]

            for trace in traces:
                supplementary_fig.append_trace(trace, row + 1, col + 1)
                supplementary_fig['layout'][f'xaxis{i + 1}'].update(dict(title=a))
                if a != b:
                    supplementary_fig['layout'][f'yaxis{i + 1}'].update(dict(title=b))
                else:
                    supplementary_fig['layout'][f'yaxis{i + 1}'].update(dict(title='freq'))
                supplementary_fig['layout']['title'] = pipe.distribution.name
        yield (supplementary_fig, "scattermatrix")
    # TODO: proper labels for triangles
    triangles = defaultdict(set)
    for id, row in pipe.estimate.iterrows():
        for group_a, group_b in combinations(groups, 2):
            x_a, y_a = row.loc[group_a, param_names[0]], row.loc[group_a, param_names[1]]
            x_b, y_b = row.loc[group_b, param_names[0]], row.loc[group_b, param_names[1]]
            triangles[id].add((x_a, y_a, group_a))
            triangles[id].add((x_b, y_b, group_b))
    areas = {}
    centroids = {}
    tcolors = {}
    max_tcolor = 0
    max_area = 0
    for (k, tri) in triangles.items():
        tri = list(tri)
        if len(tri) == 3:
            (ax, ay, ga) = tri[0]
            (bx, by, gb) = tri[1]
            (cx, cy, gc) = tri[2]
            area = abs(ax * by + bx * cy + cx * ay - ax * cy - cx * by - bx * ay) / 2
            areas[k] = area
            gx = (ax + bx + cx) / 3.0
            gy = (ay + by + cy) / 3.0
            tcolors[k] = {ga: np.sqrt((gx - ax) ** 2 + (gy - ay) ** 2),
                          gb: np.sqrt((gx - bx) ** 2 + (gy - by) ** 2),
                          gc: np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)}
            tmp = max(max_tcolor, max(tcolors[k].values()))
            max_tcolor = max_tcolor if np.isnan(tmp) or np.isinf(tmp) else tmp
            max_area = max(max_area, area)
        else:
            areas[k] = 0
            tcolors[k] = {"": (0, 0, 0)}
    tcolors2 = {}
    for k, v in tcolors.items():
        tcolors2[k] = (str(255 * v[groups[0]] / max_tcolor),
                       str(255 * v[groups[2]] / max_tcolor),
                       str(255 * v[groups[1]] / max_tcolor))
    shapes = [dict(type='path',
                   path=f'M {" L ".join([str(x) + " " + str(y) for (x, y, _) in tri])} Z',
                   fillcolor=f'rgba({",".join(tcolors2[k])}, '
                   f'{(areas[k] / max_area) ** 0.5 / 4.0})',
                   line=dict(color='rgba(0, 0, 0, 0.1)'))
              for k, tri in triangles.items()]
    fig = Figure(data=[Scatter(x=pipe.estimate.loc[:, idx[group, param_names[0]]].values.ravel(),
                               y=pipe.estimate.loc[:, idx[group, param_names[1]]].values.ravel(),
                               mode='markers',
                               marker=dict(color=colors[group]),
                               name=group,
                               legendgroup=group)
                       for group in groups]
                      + [Density(x=pipe.estimate.loc[:, idx[group, param_names[0]]].values.ravel(),
                                 xaxis='x',
                                 yaxis='y2',
                                 marker=dict(color=colors[group]),
                                 name=group,
                                 legendgroup=group,
                                 showlegend=False,
                                 )
                         for group in groups]
                      + [Density(y=pipe.estimate.loc[:, idx[group, param_names[1]]].values.ravel(),
                                 xaxis='x2',
                                 yaxis='y',
                                 orientation='h',
                                 marker=dict(color=colors[group]),
                                 name=group,
                                 legendgroup=group,
                                 showlegend=False,
                                 )
                         for group in groups],
                 layout=Layout(xaxis=XAxis(domain=[0, 0.9], title=param_names[0], showline=False),
                               xaxis2=XAxis(domain=[0.9, 1], showticklabels=False, showgrid=False,
                                            showline=False,
                                            # autorange='reversed',
                                            ),
                               yaxis=YAxis(domain=[0, 0.9], title=param_names[1], showline=False),
                               yaxis2=YAxis(domain=[0.9, 1], showticklabels=False, showgrid=False,
                                            showline=False,
                                            # autorange='reversed',
                                            ),
                               shapes=shapes,
                               )
                 )
    yield (fig, "estimates")


def _color_by_supplement_button(supplement: pd.DataFrame):
    if supplement is None or supplement.empty:
        return None
    columns = supplement.columns.values.tolist()
    names = supplement.index.values.tolist()
    column_values = {col: list(map(str, supplement[col].fillna("_unknown").values)) for col in columns}
    column_uniques = {col: list(sorted(list(np.unique(column_values[col])))) for col in columns}
    column_colors = {col: {unique: i for i, unique in enumerate(uniques)} for col, uniques in column_uniques.items()}

    def get_color(i, col):
        if column_uniques[col][i] != "_unknown":
            return f"hsv({int(i / len(column_uniques[col]) * 360)}, 0.75, 0.66)"
        else:
            return "hsva(0, 0, 0.75, 0.33)"

    colormap = {col: {i: get_color(i, col) for i in range(len(column_uniques[col]))} for col in columns}
    column_colors = {col: [colormap[col][column_colors[col][v]] for v in column_values[col]] for col in columns}
    return \
        {
            'buttons': [
                {
                    'args': [
                        {
                            'marker': [{'color': column_colors[col], 'size': 12}],
                            'text': [list(map(lambda x: f', {col}: '.join(x), zip(names, column_values[col])))],
                        }],
                    'label': col,
                    'method': 'update',
                }
                for col in columns[::-1] if 2 <= len(column_uniques[col]) <= len(column_values[col])
            ],
            'direction': 'down',
            'showactive': True,
            'xanchor': 'left',
            'yanchor': 'top',
        }


def _colorscale_button():
    return \
        {
            'buttons': [
                {
                    'args': ['colorscale', cs],
                    'label': cs,
                    'method': 'restyle',
                }
                for cs in
                ['Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu',
                 'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland', 'Jet',
                 'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis']
            ],
            'direction': 'down',
            'pad': {'r': 5, 't': 0},
            'showactive': True,
            'xanchor': 'right',
            'yanchor': 'top',
        }
