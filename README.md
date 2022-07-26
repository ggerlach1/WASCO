# WASCO: A Wasserstein-based statistical tool to compare conformational ensembles of intrinsically disordered proteins

Welcome to the beta version of this IDP ensemble comparison tool. The method implemented in this jupyter notebook computes residue-specific distances between a pair of IDP conformational ensembles, together with an overall distance for the entire ensemble. The comparison is simultaneously made at two scales:
* Global scale: distances between the distributions of the relative positions of all residue pairs in both ensembles. For each pair of residues, we compute the (2-Wasserstein) distance between a pair of probability distributions supported on the three-dimensional euclidean space (point clouds).
* Local scale: distances between the (phi, psi) angle distributions of each ensemble, for each residue along the sequence. For each residue, we compute the (2-Wasserstein) distance between a pair of probability distributions supported on the two dimensional flat torus.

Results are returned through a distance matrix, depicting both scales' results: global distances are included in the lower triangle and local distances along the diagonal. Computations include a correction to mitigate the effect of uncertainty (if independent replicas are provided or sampled). The matrix color scales correspond to:

* If no independent replicas are provided/sampled (and thus, uncertainty is ignored): the intra-ensemble distances between each pair of distributions.
* If independent replicas are provided/sampled (and thus, uncertainty is considered): the proportion of intra-ensemble distances that is added to the intra-ensemble distances to reach the encountered inter-ensemble distances. In the legend, $\Delta W$ corresponds to the difference between the inter-ensemble and the intra-ensemble distances, and $W_{\mathrm{ind}}$ indicates the intra-ensemble differences. In other words, this scale represents **how different are the inter-ensemble distances with respect to the intra-ensemble ones** (e.g. the "net" distance that has been added to uncertainty represents the 150% of such uncertainty). This was set as the easiest interpretable scale, using uncertainty as a reference to which compare the inter-ensemble differences.

The entry (i,j) of the matrix coresponds to the distance between the distributions of the relative positions i-j (one distribution per ensemble). It measures how different is the relative position of residue i with respect to j when changing from one ensemble to the other. The entry (i,i) corresponds to the distance between the distributions of the i-th residue's (phi, psi) angles (one distribution per ensemble). It measures how different is the (phi, psi) distribution of i-th amino-acid when changing from one ensemble to the other.

To apply the comparison tool for a given pair of IDP ensembles, the user can directly execute the [comparison_tool](https://github.com/gonzalez-delgado/WASCO/blob/master/comparison_tool.ipynb) file, which contains its specific instructions and guidelines. This file calls all the other notebooks included in the same folder, which can also be used individually if desired. 

Before running the function, be sure to set Python version to 3.8 and to have
installed all of the following libraries: [numpy](https://numpy.org/),
[os](https://docs.python.org/3/library/os.html),
[ipynb](https://pypi.org/project/ipynb/), [tqdm](https://tqdm.github.io/),
[joblib](https://joblib.readthedocs.io/en/latest/),
[functools](https://docs.python.org/3/library/functools.html),
[mdtraj](https://www.mdtraj.org/1.9.8.dev0/index.html),
[h5py](https://docs.h5py.org/en/stable/),
[itertools](https://docs.python.org/3/library/itertools.html),
[pandas](https://pandas.pydata.org/),
[warnings](https://docs.python.org/3/library/warnings.html),
[Bio](https://biopython.org/),
[time](https://docs.python.org/3/library/time.html),
[shutil](https://docs.python.org/3/library/shutil.html),
[seaborn](https://seaborn.pydata.org/), [matplotlib](https://matplotlib.org/),
[mdanalysis](https://www.mdanalysis.org/), [ot](https://pythonot.github.io/),
[scipy](https://scipy.org/) and [faiss](https://faiss.ai/). 

The file `environment.yml` can be used to create a conda environment with the above packages:
```conda env create --file=environment.yml```

The environment does not have ipynb or jupyter (to run the notebooks, you will still need to install those)

