# Clustering functionality

Mainly copied from code with external client.

This client had two datasets. In the code I replaced the name of the client by
`xxx` and the names of the two datasets by `yyy` and `zzz`.

`clustering_experiments.py` is actually a notebook that can be generated with `jupytext`.
It was the basis for looking at the data from the different datasets. The difference
now is that there will be only one single dataset, so much from there is not actually needed.

`clusterer.py` contains some functionality that was extracted from the notebook, 
to train the kMeans / PCA.

`data_loading.py` and `fingerprints.py` contain utilities used to load the data 
and generate the fingerprints if necessary.

These modules are then used in the scripts in the `scripts` directory.