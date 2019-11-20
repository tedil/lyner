# lyner
A chaining toolbox for working with dataframes.
Such a chain (*♫ badaa baa ♪*) usually starts with reading a tabular / matrix file, such as standard `.csv` files:
`lyner read data.csv`.
This results in the data to be interpreted as a pandas DataFrame object and stored in the pipe in a field called `matrix`.\
Each consecutive command can then make use of this `matrix`:\
`lyner read data.csv transform log2` takes `matrix` and applies a log2 transform in-place.

Since `lyner` offers a wide variety of commands, it is sometimes necessary to store auxiliary data in the pipe; one such command being the `cluster_*` family of commands, which will store cluster indices in `cluster_indices_samples` / `cluster_indices_features` depending on whether clustering was done on columns or rows respectively. You might want to store these indices later on, which can be done like this: `[...] cluster [...] select cluster_indices_samples store sample_clusters.txt`.

One of the more frequently used commands is probably `transpose`, which is why `T` is an accepted shorthand alias. This (surprise!) transposes the current selection.

## Installation
- via pip: `pip install lyner`
- via bioconda (recommended): `conda install -c bioconda lyner`

## Examples
### Plotting
- Cluster data into 3 clusters, then create an interactive heatmap:
```
lyner read data.tsv cluster -n 3 T plot -m heatmap
```

- Create an interactive scatterplot for two ICA components, allowing to change point colors according to information from the annotation:
```
lyner read data.tsv supplement annotation.csv T decompose -m ICA -n 2 plot -m scatter
```

- A more complicated chain:
```
lyner read data.csv \
    T filter --suffix _X,_Y T  \     # discard samples ending with _X or _Y (note transpose at start and end)
    filter --prefix U,V \            # discard features starting with U or V
    normalise unit \                 # normalise data to [0, 1]
    cluster -n 4 \                   # cluster features (4 clusters expected)
    T cluster -n 5 T \               # cluster samples (5 clusters expected)
    filter -v 0.05 \                 # keep only 5% most variable features
    read-annotation annotation.csv \ # read annotation data
    plot -m heatmap -c zmin=0,zmax=1 --with-annotation -o foo.html
```

*more examples to follow*
