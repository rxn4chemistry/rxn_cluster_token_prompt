# Enhancing diversity in language based models for single-step retrosynthesis

Enable diversity in single-step retrosynthesis models. The models were
trained using the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) framework.

### Install / Build
#### Create Environment and Install
```bash
# Create environmentt
conda create -n rxn-cluster-token-prompt python=3.6
conda activate rxn-cluster-token-prompt

# Add required initial packages (rxnfp may require Rust to be installed)
conda install -c rdkit rdkit=2020.03.1 # must be installed manually
pip install rxnfp

# Clone the repository
git clone git@github.ibm.com:ATO/rxn_cluster_token_prompt.git
cd rxn_cluster_token_prompt/
pip install -e .
```
For development
```bash
pip install -e .[dev]
```
When developing, before committing please run
```bash
yapf -ipr .
mypy .
flake8
```

### Try it out!
You can easily try out the rxn cluster token prompt model for high diversity retrosynthesis
predictions with 3 lines of code:
```python
from rxn_cluster_token_prompt.model import RXNClusterTokenPrompt
retro_model = RXNClusterTokenPrompt(n_best=1)
retro_model.retro_predict(["CCN(CC)Cc1ccc(-c2nc(C)c(COc3ccc([C@H](CC(=O)N4C(=O)OC[C@@H]4Cc4ccccc4)c4ccon4)cc3)s2)cc1"], reorder_by_forward_likelihood=True, verbose=True)
```

The code above calls the default model (10clusters on USPTO). 
The `n_best` is the number of predictions per token to retain.

To make predictions on a bigger dataset we recommend to use the procedure outlined
below (after the USPTO dataset preparation), as the one above is not implemented for gpus.

### USPTO Datasets generation
To generate the files for training the high diversity models
as well as a forward and a classification model for the analysis, 
first download the dataset and preprocess it:
```python
from rxn_cluster_token_prompt.uspto_datasets_loader import USPTOLoader
loader = USPTOLoader('USPTO_50K')
loader.download_dataset()
loader.process_dataset()
```
Results are saved in `path_to_this_repo/data/uspto`.

Then, you can generate the files for training and inference with the command:
```bash
generate-dataset-files --input_csv_file path_to_this_repo/data/uspto/USPTO_50K_processed.csv
                       --output_path your_output_path
                       --rxn-column-name reactions_can
                       --cluster-column-name class 
                       --model-type retro
```
For options on how to use the command run `generate-dataset-files --help`.
The model-type can be either `retro`,`forward`,`classification`

By specifying the `--cluster-colum-name` you can choose how to build your cluster token prompt model.
The column `class` in USPTO contains the reaction classes. To see how to choose
a different clustering technique, check the next section.

When the flag `--baseline` is passed together with the `retro` model type, the data
for the baseline retrosynthesis model is generated.

### Construction of the clusterers
In order to apply a clustering technique different from the reaction classification
provided by [Schneider et al](https://doi.org/10.1021/acs.jcim.6b00564), you can use the following script.
As an example, to generate the clusterer for the `10clustersKmeans` model:

First, set the following environment variables (examples are given as comments):
```bash
export FPS_SAVE_PATH=The absolute filepath where to store the computed fingerprints # path_to_this_repo/data/uspto/USPTO_50K_processed_fingerprints.pkl
export DATA_CSV_PATH=The absolute path to the data on which to compute the fingerprints # path_to_this_repo/data/uspto/USPTO_50K_processed.csv
export RXN_SMILES_COLUMN=The column name where the reactions are stored # reactions_can
```
Then, run the script:
```bash
create-clusterer \
  --clusterer_pkl path_to_this_repo/data/uspto/USPTO_50K_processed_10clustersKmeans_clusterer.pkl \
  --pca_components 3 \
  --n_clusters 10
```

You can tune the number of pca components and the number of clusters to
generate your clusterer.

### Prediction of the cluster id
Once the clusterer is generated, you are ready to compute the cluster ids
for the training/validation/test reactions. This will be added as an additional column
to the input csv file.

```bash
cluster-csv \
  --input_csv path_to_this_repo/data/uspto/USPTO_50K_processed.csv \
  --output_csv path_to_this_repo/data/uspto/USPTO_50K_processed_10clustersKmeans.csv \
  --clusterer_pkl path_to_this_repo/data/uspto/USPTO_50K_processed_10clustersKmeans_clusterer.pkl
```
If you want, alternatively to fingerprints clustering, to generate random grouping of the reaction classes 
you can pass the option `--n_clusters_random` defining
the number of wanted clusters.
Run `cluster-csv --help` for more information.

You can now generate the files with:
```bash
generate-dataset-files \
  --input_csv_file path_to_this_repo/data/uspto/USPTO_50K_processed_10clustersKmeans.csv \
  --output_path your_output_dir \
  --rxn-column-name reactions_can \
  --cluster-column-name cluster_id \
  --model-type retro
```
The files will be saved under `your_output_dir/random5`, where 5 is the random seed used to 
generate the splits. You can change the seed with the `--seed` option.

### Training
To train the models you can costumize the script `bin/training.sh` and run it on a system with 
one gpu. The USPTO models were trained up to 130000 steps (roughly 24 hours).

### Prediction

Once your models are trained you can run the predictions with the custumizable script `bin/translate.sh` on a system with 
one gpu.

### Evaluation

To evaluate your models you can customize the script `bin/compute_metrics.sh`. The output is a json file called metrics.json
where the values of accuracy, round-trip accuracy, class-diversity and coverage are reported.
