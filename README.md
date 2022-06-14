# The Chemistry Puppetter

Enable diversity in single-step retrosynthesis models. The models were
trained using the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) framework.
PROBLEM: we use an internal version!!!

### Install / Build
#### Create Environment and Install
```bash
conda create -n rxn-class-token python=3.6
conda install -c rdkit rdkit=2020.03.1 # must be installed manually
git clone git@github.ibm.com:ATO/rxn_class_token.git
cd rxn_class_token/
pip install -e .
```
For development
```bash
pip install -e .\[dev]
```
Install reaction fingerprint
```bash
git clone git@github.com:rxn4chemistry/rxnfp.git
cd rxnfp/
pip install -e .
```
Before committing, please run
```bash
yapf -ipr .
mypy .
flake8
```

### Try it out!

### USPTO Datasets generation
To generate the files for training the chemistry puppeteer models
as well as a forward and a classification model for the analysis, 
first download the dataset and preprocess it:
```python
from rxn_class_token.uspto_datasets_loader import USPTOLoader
loader = USPTOLoader('USPTO_50K')
loader.download_dataset()
loader.process_dataset()
```
Results are saved in `path_to_this_repo/data/uspto`.

Then, you can generate the files for training and inference with the command:
```bash
generate-dataset-files --input_csv_file data/uspto/USPTO_50K_processed.csv
                       --output_path your_output_path
                       --rxn-column-name reactions_can
                       --cluster-column-name class 
                       --model-type retro
```
For options on how to use the command run `generate-dataset-files --help`.
The model-type can be either `retro`,`forward`,`classification`

By specifying the `--cluster-colum-name` you can choose how to build your Chemistry Puppeteer model.
The column `class` in USPTO contains the reaction classes. To see how to choose
a different clustering technique, please check the next section.

When the flag `--baseline` is passed together with the `retro` model type, the data
for the baseline retrosynthesis model is generated.

### Construction of the clusterers
In order to apply a clustering technique different from the reaction classification
provided by [Schneider et al](https://doi.org/10.1021/acs.jcim.6b00564), you can use the following script.
As an example, to generate the clusterer for the `10clustersKmeans` model:

First, set the following environment variables (examples are given as comments):
```bash
export FPS_SAVE_PATH=The absolute filepath where to store the computed fingerprints # path_to_this_repo/data/uspto/USPTO_50K_processed_fingerprints.pkl
export FPS_MODEL_PATH=The absolute path to the trained fingerprints model # in the rxnfp repo under `rxnfp/models/transformers/bert_ft`
export DATA_CSV_PATH=The absolute path to the data on which to compute the fingerprints # path_to_this_repo/data/uspto/USPTO_50K_processed.csv
export RXN_SMILES_COLUMN=The column name where the reactions are stored # reactions_can
```
Then, run the script:
```bash
create-clusterer --clusterer_pkl path_to_this_repo/data/uspto/USPTO_50K_processed_10clustersKmeans_clusterer.pkl
                 --pca_components 3 
                 --n_clusters 10
```

You can tune the number of pca components and the number of clusters to
generate your clusterer.

### Prediction of the cluster id
Once the clusterer is generated, you are ready to compute the cluster ids
for the training/validation/test reactions. This will be added as an additional column
to the input csv file.

```bash
cluster-csv --input_csv path_to_this_repo/data/uspto/USPTO_50K_processed.csv
            --output_csv path_to_this_repo/data/uspto/USPTO_50K_processed_10clustersKmeans.csv
            --clusterer_pkl path_to_this_repo/data/uspto/USPTO_50K_processed_10clustersKmeans_clusterer.pkl
```
If you want, alternatively to fingerprints clustering, to generate random grouping of the reaction classes 
you can pass the option `--n_clusters_random` defining
the number of wanted clusters.
Run `cluster-csv --help` for more information.

You can now generate the files with:
```bash
generate-dataset-files --input_csv_file path_to_this_repo/data/uspto/USPTO_50K_processed_10clustersKmeans.csv
                       --output_path your_output_dir 
                       --rxn-column-name reactions_can
                       --cluster-column-name cluster_id 
                       --model-type retro
```

### TRAINING

### PREDICTION

### EVALUATION

### USE THE MODEL

