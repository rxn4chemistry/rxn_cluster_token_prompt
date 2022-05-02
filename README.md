# rxn_class_token

Repository to prepare and analyze data for the class token 

## Install / Build
### Create Environment and Install
```bash
conda create -n rxn-class-token python=3.6
conda install -c rdkit rdkit=2020.03.1 # must be installed manually
pip install -e .
```
For development
```bash
pip install -e .\[dev]
```
Install reaction fingerprint
```bash
git clone git@github.com:rxn4chemistry/rxnfp.git
cd rxnfp
pip install -e .
```
Before committing, please run
```bash
yapf -ipr .
mypy .
flake8
```
## Datasets generation
The data for pistachio is stored in the ibm box folder. 
Click [here](https://ibm.box.com/s/228otc58sl19evweosamxgyjf66cv025) to access the files
### Generate one or more random splits
The random splits can be generated with the following script.
```bash
generate_multiple_splits input_csv_file output_path --seed 10 --seed 42 --split_ratio 0.1
```
The files are saved in folders like `random10` and `random42` under `output_path`.
### Baseline model
The baseline model does not use any class token. To generate it call the following script:
```bash
generate_dataset input_file_train_csv output_path_baseline --output-type train --no-class-token
generate_dataset input_file_test_csv output_path_baseline --output-type test --no-class-token
generate_dataset input_file_valid_csv output_path_baseline --output-type valid --no-class-token
```
### 12tokens model
The data for the 12tokens model can be generated with the following:
```bash
generate_dataset input_file_train_csv output_path_12tokens --output-type train
generate_dataset input_file_test_csv output_path_12tokens --output-type test
generate_dataset input_file_valid_csv output_path_12tokens --output-type valid
```
### group1 model
The data for the group1 model can be generated with the following:
```bash
generate_dataset input_file_train_csv output_path_group1 --output-type train --map-file path_to_this_repo/maps/group1.json

generate_dataset input_file_test_csv output_path_group1 --output-type test --map-file path_to_this_repo/maps/group1.json

generate_dataset input_file_valid_csv output_path_group1 --output-type valid --map-file path_to_this_repo/maps/group1.json

```
For costum mapping of the reaction classes, a json similar to the `group1.json` can be provided

### Preparing a new dataset for predictions
If predictions need to be launched on a new dataset, to prepare the dataset the following script can be used.
For the 12tokens model
```bash
generate_prediction_dataset input_file_precursors_txt --precursors
generate_prediction_dataset input_file_product_txt --product
```
For the group1 model
```bash
generate_prediction_dataset input_file_precursors_txt --precursors --map-file path_to_this_repo/maps/group1.json
generate_prediction_dataset input_file_product_txt --product --map-file path_to_this_repo/maps/group1.json
```

### Generating the class tokens from kmeans clustering
Set the following environment variables
```bash
export FPS_SAVE_PATH=The path where to store the computed fingerprints
export FPS_MODEL_PATH=The path to the trained fingerprints model
export DATA_CSV_PATH=The path to the data on which to compute the fingerprints
```