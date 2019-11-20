import logging
import os
import sys
from collections import defaultdict
from itertools import product

import click
import numpy as np
import pandas as pd
import plotly
import plotly.io as pio
from plotly.graph_objs import Scatter, Figure, Heatmap, Bar, Layout, Histogram
from plotly.io import write_image
from plotly.offline import plot as oplot
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import fcluster, linkage

from lyner._main import rnax
from lyner.click_extras import ListChoice, INT_RANGE_LIST, pass_pipe, arggist, Pipe, CombinatorialChoice, DICT, \
    plain_command_string
from lyner.dendromap import dendromap_square, dendromap
from lyner.misc import connectivity, _connectivity
from lyner.plotly_extras import _figure_add_changes, _figure_get_estimates, \
    _color_by_supplement_button

LOGGER = logging.getLogger("lyner")
logging.basicConfig(level=logging.NOTSET)

pio.orca.config.use_xvfb = True


@rnax.command()
@click.option("--axis", "-a", type=click.IntRange(0, 1), default=0)
@click.option("--methods", "-m", type=ListChoice(["NMF", "RPCA", "PCA", "ICA", "TSNE"]), default="PCA")
@click.option("--mode", type=click.Choice({"each", "consensus"}), default="each")
@click.option("--num-components", "-c", type=INT_RANGE_LIST, default="2-6")
@click.option("--num-runs", "-r", type=click.INT, default=1)
@pass_pipe
@arggist
def dendro(pipe: Pipe, axis, methods, mode, num_components, num_runs):
    """Build a dendrogram based on the results of chosen decomposition methods."""
    from sklearn.decomposition import NMF, PCA, FastICA as ICA
    from sklearn.manifold import TSNE
    matrix = pipe.matrix
    data = matrix.values
    if axis == 1:
        data = data.T
        labels = matrix.index.values.tolist()
    else:
        labels = matrix.columns.values.tolist()

    def apply(method):
        def apply_nmf():
            cons = np.mat(np.zeros((data.shape[1], data.shape[1])))
            err = float('inf')
            for components in num_components:
                for _ in range(num_runs):
                    nmf = NMF(n_components=components, init='random')
                    X_r = nmf.fit_transform(data.T)
                    if nmf.reconstruction_err_ < err:
                        best_X_r = X_r
                        err = nmf.reconstruction_err_
                    cons += connectivity(data, X_r.T)
            C = 1 - (cons / (num_runs * len(num_components)))
            return C, best_X_r

        def apply_pca():
            cons = np.mat(np.zeros((data.shape[1], data.shape[1])))
            for components in num_components:
                pca = PCA(n_components=components)
                X_r = pca.fit_transform(data.T)
                cons += connectivity(data, X_r.T)
            C = 1 - (cons / len(num_components))
            return C, X_r

        def apply_ica():
            cons = np.mat(np.zeros((data.shape[1], data.shape[1])))
            for components in num_components:
                pca = ICA(n_components=components)
                X_r = pca.fit_transform(data.T)
                cons += connectivity(data, X_r.T)
            C = 1 - (cons / len(num_components))
            return C, X_r

        def apply_rpca():
            cons = np.mat(np.zeros((data.shape[1], data.shape[1])))
            for components in num_components:
                for _ in range(num_runs):
                    rpca = PCA(n_components=components, svd_solver='randomized')
                    X_r = rpca.fit_transform(data.T)
                    cons += connectivity(data, X_r.T)
            C = 1 - (cons / (num_runs * len(num_components)))
            return C, X_r

        def apply_tsne():
            tsne_methods = {c: 'exact' if c >= 4 else 'barnes_hut' for c in num_components}
            n_sne = data.shape[0]
            cons = np.mat(np.zeros((data.shape[1], data.shape[1])))
            for components in num_components:
                for i in range(num_runs):
                    print(i)
                    tsne = TSNE(n_components=components,
                                perplexity=20,
                                method=tsne_methods[components])
                    rndperm = np.random.permutation(list(range(data.shape[1])))
                    X_r = tsne.fit_transform(data[rndperm[:n_sne], :].T)
                    cons += connectivity(data, X_r.T)
            C = 1. - (cons / (num_runs * len(num_components)))
            return C, X_r

        if "NMF" == method:
            return "NMF", apply_nmf()
        if "PCA" == method:
            return "PCA", apply_pca()
        if "ICA" == method:
            return "ICA", apply_ica()
        if "RPCA" == method:
            return "RPCA", apply_rpca()
        if "TSNE" == method:
            return "TSNE", apply_tsne()

    auto_open = True if 'DISPLAY' in os.environ and os.environ['DISPLAY'] else False
    cons = np.mat(np.zeros((data.shape[1], data.shape[1])))
    for m, (C, X_r) in map(apply, methods):
        # make dendrogram + heatmap
        cluster_idx = fcluster(linkage(C, method='ward'), int(sum(num_components) / len(num_components)),
                               criterion='maxclust')
        cons += _connectivity(data, cluster_idx)
        if mode == "each":
            clusters = defaultdict(list)
            for idx, l in zip(cluster_idx, labels):
                clusters[idx].append(l)
            for (idx, c) in clusters.items():
                print(f"{idx}: {c}")
            pipe.clusters = clusters  # FIXME

            colors = list(cluster_idx)
            fig = dendromap_square(C, labels=labels)
            fig['layout']['title'] = f"Dissimilarity consensus ({m})<br>" \
                                     f"(n_runs: {num_runs}, n_components: {num_components})"
            fig['layout']['xaxis'].update(automargin=True)

            oplot(fig, filename=f"/tmp/dendro_{m}_{axis}.html", show_link=False, auto_open=auto_open)
            best_num_components = X_r.shape[0]
            if best_num_components > 6:
                components = 6
                print(f"Plotting only the first 6 components", file=sys.stderr)
            else:
                components = best_num_components

            # plot all components
            rows = components
            cols = components
            titles = [f"{r} - {c}" for r, c in product(range(rows), range(cols))]
            fig = make_subplots(rows, cols,
                                print_grid=False,
                                shared_xaxes=True,
                                shared_yaxes=True,
                                subplot_titles=titles)
            traces = [(r, c,
                       Scatter(x=X_r[:, r],
                               y=X_r[:, c],
                               mode='markers',
                               marker=dict(color=colors), text=labels,
                               name=f"{r} - {c}",
                               showlegend=False,
                               )
                       )
                      for r, c in product(range(rows), range(cols))]
            for r, c, t in traces:
                fig.append_trace(t, r + 1, c + 1)
            fig.update_layout(title=f"{m} Components")
            oplot(fig, filename=f"/tmp/components_{m}_{axis}.html", show_link=False, auto_open=auto_open)

    if mode == "consensus":
        cons = 1 - cons / len(methods)
        cluster_idx = fcluster(linkage(cons, method='ward'), int(sum(num_components) / len(num_components)),
                               criterion='maxclust')
        clusters = defaultdict(list)
        for idx, l in zip(cluster_idx, labels):
            clusters[idx].append(l)
        pipe.clusters = clusters
        print("{")
        for (idx, c) in clusters.items():
            print(f"\t{idx}: {c},\n")
        print("}")
        fig = dendromap_square(cons, labels=labels)
        fig['layout']['title'] = f"Dissimilarity consensus<br>" \
                                 f"(n: {len(methods)}, n_runs: {num_runs}, n_components: {num_components})"
        fig['layout']['xaxis'].update(automargin=True)

        oplot(fig, filename=f"/tmp/dendro_consensus_{axis}.html", show_link=False, auto_open=auto_open)


@rnax.command()
@click.option("--outfile", "-o", type=click.Path(writable=True, file_okay=True, dir_okay=False))
@click.option("--directory", "-d", type=click.Path(writable=True, dir_okay=True, file_okay=False))
@click.option("--with-annotation", is_flag=True)
@click.option("--annotation-split", type=click.FloatRange(0, 1), default=0.4)
@click.option("--colorscale",
              type=click.Choice(['Greys', 'YlGnBu', 'Greens', 'YlOrRed', 'Bluered',
                                 'RdBu', 'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland',
                                 'Jet', 'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis']),
              default='RdBu')
@click.option("--mode", "-m",
              type=CombinatorialChoice(["heatmap", "scatter", "lines", "bar", "dendrogram", "histogram"]),
              default="heatmap")
@click.option("--mode-config", "-c", type=DICT, default={})
@click.option("--auto-open", "-a", is_flag=True, callback=lambda ctx, param, value: not ctx.params['outfile'] or value)
@pass_pipe
def plot(pipe: Pipe, outfile, directory, colorscale, mode: str, mode_config: dict, auto_open: bool,
         with_annotation: bool, annotation_split: float):
    """Visualize current selection in different ways, depending on context."""
    title = plain_command_string(pipe.command)
    title = f"<span style='font-size:0.7em'>{title}</span>"
    if not directory:
        if outfile:
            if os.sep in outfile:
                directory, outfile = os.path.split(outfile)
        else:
            outfile = "lyner"
            directory = "."
    directory = directory.rstrip(os.sep) if directory else "."
    if outfile.endswith('.svg'):
        filetype = 'svg'
        outfile = outfile.rstrip('.svg')
    elif outfile.endswith('.html'):
        filetype = 'html'
        outfile = outfile.rstrip('.html')
    elif outfile.endswith('.png'):
        filetype = 'png'
        outfile = outfile.rstrip('.png')
    elif outfile.endswith('.json'):
        filetype = 'json'
        outfile = outfile.rstrip('.json')
    else:
        filetype = 'html'
    if not os.path.exists(directory):
        os.mkdir(directory)

    matrix = pipe.matrix

    if "dendrogram" in mode:
        mc = mode_config.copy()
        mc['colorscale'] = colorscale
        if 'zmin' not in mc or 'zmax' not in mc:
            scale = np.nanpercentile(pipe.matrix.values, 90) * 2
            mc['zmin'] = -scale
            mc['zmax'] = scale
        linkage = mc.pop('linkage', 'ward')
        metric = mc.pop('metric', 'euclidean')
        fig = dendromap(pipe.matrix, heatmap_opts=mc, linkage=linkage, metric=metric)
        # fig.update_layout(updatemenus=[_colorscale_button()])
        fig.update_layout(title=title)
        fig.update_layout(template="plotly_white")
        if filetype in {'html'}:
            oplot(fig, show_link=False, filename=f'{directory}{os.sep}{outfile}.html', auto_open=auto_open)
        if filetype in {'svg', 'png'}:
            write_image(fig, f'{directory}{os.sep}{outfile}.svg', width=1920, height=1080)
    if "heatmap" in mode:
        # FIXME
        y_labels = [str(v[1]).strip() if not isinstance(v, str) else v for v in matrix.index.values]
        x_labels = [str(v[1]).strip() if not isinstance(v, str) else v for v in matrix.columns.values]
        fig = Figure(data=[Heatmap(z=matrix.values, x=x_labels, y=y_labels)])
        if with_annotation:
            num_annotations = len(pipe.annotation.columns)
            gridfig = plotly.subplots.make_subplots(rows=num_annotations + 1, cols=1,
                                                    shared_xaxes=True,
                                                    shared_yaxes=False,
                                                    print_grid=False)
            split_annotation, split_data = (annotation_split, 1. - annotation_split)
            annotation_bar_size = split_annotation / num_annotations
            annotation_bar_padding = 0.0025
            for i in range(2, num_annotations + 2):
                gridfig["layout"][f"yaxis{i}"].update(
                    domain=[0 + annotation_bar_size * (i - 2) + annotation_bar_padding,
                            annotation_bar_size * (i - 1) - annotation_bar_padding])
            gridfig["layout"]["yaxis1"].update(domain=[split_annotation, split_annotation + split_data])
        else:
            split_annotation, split_data = (0.0, 1.0)
            gridfig = plotly.subplots.make_subplots(rows=1, cols=1,
                                                    shared_xaxes=True,
                                                    shared_yaxes=False,
                                                    print_grid=False)
        heatmap = fig['data'][0]
        heatmap.update(mode_config)
        if 'zmin' not in mode_config or 'zmax' not in mode_config:
            scale = np.nanpercentile(pipe.matrix.values, 90) * 2
            heatmap['zmin'] = -scale
            heatmap['zmax'] = scale
        else:
            if 'zmin' in mode_config:
                heatmap['zmin'] = mode_config['zmin']
            if 'zmax' in mode_config:
                heatmap['zmax'] = mode_config['zmax']

        if colorscale:
            heatmap['colorscale'] = colorscale

        if 'changes' in pipe:
            changes_fig = _figure_add_changes(pipe, fig)
            oplot(changes_fig, filename=f'{directory}/{outfile}_changes_hist.html', show_link=False,
                  auto_open=auto_open)
        if hasattr(pipe, "estimate"):
            figs = _figure_get_estimates(pipe)
            for (f, fname) in figs:
                oplot(f, show_link=False, filename=f'{directory}/{outfile}_{fname}.html', auto_open=auto_open)
        heatmap["colorbar"].update(len=split_data, yanchor="top", y=1)
        gridfig.append_trace(heatmap, 1, 1)

        gridfig.update_layout(title=title)
        gridfig.update_layout(template="plotly_white")

        if with_annotation:
            annotation: pd.DataFrame = pipe.annotation

            a, b = set(annotation.index), set(
                pipe.matrix.columns.levels[1] if pipe.matrix.columns.nlevels > 1 else pipe.matrix.columns)
            annotation.drop(index=a - (a & b), inplace=True)
            j = 0
            for i, anno in enumerate([[l] for l in list(annotation.columns)]):
                print(i, anno)
                ab = _mk_annotation_bar(annotation[anno].transpose())
                if ab:
                    ab["colorbar"].update(len=annotation_bar_size * 1.125,
                                          yanchor="bottom",
                                          y=0 + (i - j) * annotation_bar_size)
                    gridfig.append_trace(ab, 2 + i - j, 1)
                else:
                    # FIXME this compacts annotation bars in case there are some missing/incompatible
                    # get rid of them at the start of this routine instead.
                    j += 1
        gridfig['layout']['xaxis'].update(side="top")
        gridfig['layout']['title'].update(y=0.075)
        if filetype in {'html'}:
            oplot(gridfig, show_link=False, filename=f'{directory}{os.sep}{outfile}.html', auto_open=auto_open)
        if filetype in {'json'}:
            import json
            json.dump(fig, open(f'{directory}{os.sep}{outfile}.json', 'wt'), cls=plotly.utils.PlotlyJSONEncoder)
        if filetype in {'svg', 'png'}:
            pio.orca.ensure_server()
            write_image(gridfig, f'{directory}{os.sep}{outfile}.svg', width=1920, height=1080)
    if "scatter" in mode:
        matrix = np.transpose(matrix)
        dimension = matrix.shape[1]
        rows = cols = dimension
        labels = matrix.index.values.tolist()
        v = matrix.values

        if dimension == 2:
            trace = Scatter(x=v[:, 0], y=v[:, 1], mode='markers', showlegend=False, text=labels, **mode_config,
                            marker=dict(size=12))
            fig = plotly.graph_objs.Figure(data=[trace])
            if hasattr(pipe, "supplement"):
                color_button = _color_by_supplement_button(pipe.supplement)
                if color_button:
                    fig.update_layout(updatemenus=[color_button])
            fig.update_layout(title=title)
            fig.update_layout(template="plotly_white")
            if filetype in {'html'}:
                oplot(fig, show_link=False, filename=f'{directory}{os.sep}{outfile}.html', auto_open=auto_open)
            if filetype in {'svg', 'png'}:
                write_image(fig, f'{directory}{os.sep}{outfile}.svg', width=1920, height=1080)
        else:
            if dimension > 6:
                LOGGER.warning(f"Plotting {dimension}² subplots may take quite some time.")
            titles = [f"{r} - {c}" for r, c in product(range(dimension), range(dimension))]
            fig = make_subplots(rows, cols,
                                print_grid=False,
                                shared_xaxes=True,
                                shared_yaxes=True,
                                subplot_titles=titles)

            traces = [(r, c,
                       Scatter(x=v[:, r],
                               y=v[:, c],
                               mode='markers',
                               text=labels,
                               name=f"{r} - {c}",
                               showlegend=False,
                               **mode_config,
                               )
                       )
                      for r, c in product(range(dimension), range(dimension))]
            for r, c, t in traces:
                fig.append_trace(t, r + 1, c + 1)
            fig.update_layout(title=f"{dimension} Components")
            fig.update_layout(template="plotly_white")
            if hasattr(pipe, "supplement"):
                color_button = _color_by_supplement_button(pipe.supplement)
                if color_button:
                    fig.update_layout(updatemenus=[color_button])
            if filetype in {'html'}:
                oplot(fig, show_link=False, filename=f'{directory}{os.sep}{outfile}.html', auto_open=auto_open)
            if filetype in {'svg', 'png'}:
                write_image(fig, f'{directory}{os.sep}{outfile}.svg', width=1920, height=1080)
    if "lines" in mode or "bar" in mode:
        dimension = matrix.shape[0]
        column_names = matrix.columns.values.tolist()
        row_names = matrix.index.values.tolist()
        v = matrix.values
        diagram_type = Scatter if "lines" in mode else Bar if "bar" in mode else Scatter
        diagram_options = {} if "lines" in mode else {"opacity": 0.75} if "bar" in mode else {}
        layout_options = {}
        traces = [diagram_type(x=column_names, y=v[row, :], name=row_names[row], **diagram_options, **mode_config)
                  for row in range(dimension)]
        fig = Figure(data=traces, layout=Layout(**layout_options))
        fig.update_layout(title=title)
        fig.update_layout(template="plotly_white")
        if filetype in {'html'}:
            oplot(fig, show_link=False, filename=f'{directory}{os.sep}{outfile}.html', auto_open=auto_open)
        if filetype in {'svg', 'png'}:
            write_image(fig, f'{directory}{os.sep}{outfile}.svg', width=1920, height=1080)
    if "histogram" in mode:
        dimension = matrix.shape[0]
        _column_names = matrix.columns.values.tolist()
        row_names = matrix.index.values.tolist()
        v = matrix.values
        traces = [Histogram(x=v[row, :], name=row_names[row], **mode_config) for row in range(dimension)]
        fig = Figure(data=traces, layout=Layout())
        fig.update_layout(title=title)
        fig.update_layout(template="plotly_white")
        if filetype in {'html'}:
            oplot(fig, show_link=False, filename=f'{directory}{os.sep}{outfile}.html', auto_open=auto_open)
        if filetype in {'svg', 'png'}:
            write_image(fig, f'{directory}{os.sep}{outfile}.svg', width=1920, height=1080)


def _mk_annotation_bar(annotation: pd.DataFrame):
    colorscale, mapping = _define_colorscale(annotation)
    categories = set(map(str.strip, map(str, annotation.values[0]))) - {'nan'}
    num_categories = len(categories)
    if num_categories == 1:
        return None
    if not colorscale:
        colorscale = "Greys"
    values = annotation.values.copy()
    if mapping:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                values[i, j] = mapping[str(values[i, j])] if str(values[i, j]) != 'nan' else values[i, j]
    annotation_bar = Heatmap(y=list(annotation.index), x=list(annotation.columns), z=values, colorscale=colorscale,
                             colorbar=dict(tickmode='array',
                                           tickvals=np.array(list(mapping.values())) if mapping else None,
                                           ticktext=np.array(list(mapping.keys())) if mapping else None),
                             )
    return annotation_bar


def _define_colorscale(annotation: pd.DataFrame):
    if annotation.values.dtype != np.float:
        import colorlover as cl
        categories = set(map(str.strip, map(str, annotation.values[0]))) - {'nan'}
        categories = list(sorted(categories))
        num_categories = len(categories)
        if num_categories == 1:
            return None, None
        if num_categories > 12:
            return None, None
        available_palettes = cl.scales[f'{num_categories + (1 if num_categories == 2 else 0)}']['qual']
        key = list(available_palettes.keys())[0]
        palette = available_palettes[key]
        colorscale = []
        step_size = 1 / num_categories
        colors = [palette[int(np.floor(i / 2.))] for i in range(num_categories * 2)]
        for i, c in enumerate(colors):
            colorscale.append([((i + i % 2) / 2) * step_size, c])
        mapping = dict([(a, b / num_categories) for b, a in enumerate(categories)])
        return colorscale, mapping
    else:
        return None, None
