import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import scipy.cluster.hierarchy as sch


def dendromap_square(data, labels, linkage=lambda x: sch.linkage(x, 'ward')):
    figure = ff.create_dendrogram(data, orientation='bottom', labels=labels, linkagefun=linkage)
    for i in range(len(figure['data'])):
        figure['data'][i]['yaxis'] = 'y2'
        figure['data'][i]['xaxis'] = 'x1'
    dendro_side = ff.create_dendrogram(data, orientation='right', linkagefun=linkage)
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'
        dendro_side['data'][i]['yaxis'] = 'y1'
        figure.add_trace(dendro_side['data'][i])
    # figure['data'].extend(dendro_side['data'])

    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
    dendro_leaves = list(map(int, dendro_leaves))

    # use `data` matrix directly, i.e. do not compute `squareform(pdist(data))`
    Y = linkage(data)
    denD = sch.dendrogram(Y, orientation='right', link_color_func=lambda k: 'black', no_plot=True)
    D = data[denD['leaves'], :][:, denD['leaves']]
    heat_data = D

    # data_dist = pdist(data)
    # heat_data = squareform(data_dist)
    # heat_data = heat_data[dendro_leaves, :]
    # heat_data = heat_data[:, dendro_leaves]

    import numpy as np
    lbls = np.array(labels)[dendro_leaves]
    hovertext = np.empty_like(heat_data, dtype=object)
    for i in range(hovertext.shape[0]):
        for j in range(hovertext.shape[1]):
            hovertext[i, j] = f"{lbls[i]} ↔ {lbls[j]}<br>{heat_data[i, j]}"
    heatmap = [
        go.Heatmap(
            x=lbls,
            y=lbls,
            z=heat_data,
            colorscale='YlGnBu',
            text=hovertext,
            hoverinfo='text'
        )
    ]

    heatmap[0]['x'] = list(range(5, len(lbls) * 10 + 5, 10))
    heatmap[0]['y'] = list(range(5, len(lbls) * 10 + 5, 10))

    # Add Heatmap Data to Figure
    # figure['data'].extend(heatmap)
    figure.add_trace(heatmap[0])

    # Edit Layout
    figure['layout'].update({'width': None, 'height': None})
    figure['layout'].update({'showlegend': False, 'hovermode': 'closest',
                             'autosize': True})

    # Edit xaxis (heatmap x)
    figure['layout']['xaxis'].update({'domain': [.15, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticks': ""})
    # Edit xaxis2 (left hand dendro)
    figure['layout'].update({'xaxis2': {'domain': [0, .15],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'showticklabels': False,
                                        'ticks': ""}})

    # Edit yaxis (heatmap y)
    figure['layout']['yaxis'].update({'domain': [0, .85],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': False,
                                      'ticks': ""})
    # Edit yaxis2 (top side dendro)
    figure['layout'].update({'yaxis2': {'domain': [.85, 1],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'showticklabels': False,
                                        'ticks': ""}})
    figure['layout']['xaxis']['tickvals'] = list(range(5, len(lbls) * 10 + 5, 10))
    figure['layout']['yaxis']['tickvals'] = list(range(5, len(lbls) * 10 + 5, 10))
    return figure


def dendromap(data, linkage='ward', metric='euclidean', heatmap_opts: dict = dict()):
    data = data.T
    distfun = lambda x: sch.distance.pdist(x, metric=metric)
    linkagefun = lambda x: sch.linkage(x, linkage)
    figure = ff.create_dendrogram(data.values,
                                  orientation='bottom',
                                  labels=data.index.values,
                                  linkagefun=linkagefun,
                                  distfun=distfun,
                                  )
    for i in range(len(figure['data'])):
        figure['data'][i]['yaxis'] = 'y2'
        figure['data'][i]['xaxis'] = 'x1'
    dendro_side = ff.create_dendrogram(data.values.T,
                                       orientation='right',
                                       labels=data.columns.values,
                                       linkagefun=linkagefun,
                                       distfun=distfun,
                                       )
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'
        dendro_side['data'][i]['yaxis'] = 'y1'
        figure.add_trace(dendro_side['data'][i])

    Y = linkagefun(data)
    X = linkagefun(data.T)
    denD1 = sch.dendrogram(X, orientation='right', link_color_func=lambda k: 'black', no_plot=True)
    denD2 = sch.dendrogram(Y, orientation='bottom', link_color_func=lambda k: 'black', no_plot=True)
    data: pd.DataFrame = data
    d_colindex = data.columns.values[denD1['leaves']]
    d_rowindex = data.index.values[denD2['leaves']]
    data = data.reindex(index=d_rowindex, columns=d_colindex).T

    dendro_side['layout']['yaxis']['ticktext'] = d_colindex
    figure['layout']['yaxis']['ticktext'] = dendro_side['layout']['yaxis']['ticktext']
    figure['layout']['yaxis']['tickvals'] = dendro_side['layout']['yaxis']['tickvals']
    figure['layout']['yaxis']['side'] = "right"
    figure['layout']['xaxis']['ticktext'] = d_rowindex

    # D = data[denD1['leaves'], :][:, denD1['leaves']]
    # D = D[denD2['leaves'], :][:, denD2['leaves']]
    heat_data = data.values
    import numpy as np
    hovertext = np.empty_like(heat_data, dtype=object)
    for i in range(hovertext.shape[0]):
        for j in range(hovertext.shape[1]):
            hovertext[i, j] = f"{data.index[i]} ↔ {data.columns[j]}<br>{heat_data[i, j]}"
    heatmap = [
        go.Heatmap(
            x=data.index.values,
            y=data.columns.values,
            z=heat_data,
            text=hovertext,
            hoverinfo='text',
            colorbar={'x': 1.1, 'len': 0.825, 'yanchor': 'bottom', 'y': 0},
            **heatmap_opts
        )
    ]

    heatmap[0]['x'] = list(range(5, len(data.columns.values) * 10 + 5, 10))
    heatmap[0]['y'] = list(range(5, len(data.index.values) * 10 + 5, 10))

    # Add Heatmap Data to Figure
    # figure['data'].extend(heatmap)
    figure.add_trace(heatmap[0])

    # Edit Layout
    figure['layout'].update({'width': None, 'height': None})
    figure['layout'].update({'showlegend': False, 'hovermode': 'closest',
                             'autosize': True})

    # Edit xaxis (heatmap x)
    figure['layout']['xaxis'].update({'domain': [.15, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticks': ""})
    # Edit xaxis2 (left hand dendro)
    figure['layout'].update({'xaxis2': {'domain': [0, .15],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'showticklabels': False,
                                        'ticks': ""}})

    # Edit yaxis (heatmap y)
    figure['layout']['yaxis'].update({'domain': [0, .85],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': True,
                                      })
    # Edit yaxis2 (top side dendro)
    figure['layout'].update({'yaxis2': {'domain': [.85, 1],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'showticklabels': False,
                                        'ticks': ""}})
    figure['layout']['xaxis']['tickvals'] = list(range(5, len(data.columns.values) * 10 + 5, 10))
    figure['layout']['yaxis']['tickvals'] = list(range(5, len(data.index.values) * 10 + 5, 10))
    return figure
