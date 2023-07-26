# Predict the CDR conformation for CDR loops in an AbDb structure 

## Requirements
```
- python == 3.9
- numpy==1.23.5
- pandas==2.0.3
- scipy==1.1.0
- scikit-learn==1.3.0
- PyYAML==6.0.1 
- joblib==1.3.1
- biopython==1.81
```

## Dependencies 
- `ABDB `: a snapshot of AbDb database, the version used in the publication is `20220926`
- `classifier`: CDR conformation classifiers output by this study, each classifier is a scikit-learn AP AffinityPropagation object, packed as `joblib` file, for details refer to scikit-learn documentation at [here](https://scikit-learn.org/stable/model_persistence.html)
- `LRC_AP_cluster.json`: a JSON file containing information about LRC groups, Canonical clusters, and AP clusters
  
Place these folders and files inside `./dirs` or create soft links inside `./dirs` point to them, for example 
```
$ cd ./dirs
$ ln -s /path/to/ABDB ./ABDB
$ ln -s /path/to/classifier ./classifier
```

## Usage
Create a python 3.9 environment and install dependencies
```bash 
# create an environment named cdrclass
$ conda create -n cdrclass python=3.9
$ conda activate cdrclass

# install dependencies
$ cd /path/to/CDRConformationClassification
$ pip install .  
```

Run classification on a single AbDb structure
```bash
$ python classify_general_abdb_entry.py \
    --cdr all \
    --outdir ./results \
    --config ./config/classify_general_abdb_entry-runtime.yaml \
    1a2y_0P
```
This outputs a JSON file in `./results` directory, the file name is `1a2y_0P.json`, it should look like the following: 
```JSON
[
    {
        "H1": {
            "closest_lrc": "H1-10-allT",
            "closest_AP_cluster_label": 46,
            "closest_AP_cluster_exemplar_id": "6azk_0",
            "closest_AP_cluster_size": 93,
            "closest_can_cluster_index": 1,
            "merged_AP_cluster_label": null,
            "merged_AP_cluster_exemplar_id": null,
            "merged_AP_cluster_size": null,
            "merged_can_cluster_index": null,
            "merge_with_closest_exemplar_torsional": true,
            "merge_with_any_exemplar_cartesian": null,
            "merged": true
        }
    },
]
```
Here, only the H1 CDR loop is classified, the classification results are stored in a list of dictionaries, each dictionary contains the classification results for a CDR loop.

- `closest_lrc`: the closest LRC group in torsional space 
- `closest_AP_cluster_label`: the closest AP cluster label, this is a unique integer assigned to each AP cluster within a LRC group
- `closest_AP_cluster_exemplar_id`: the closest AP cluster exemplar ID, this is the ID of the structure that is used as the exemplar for the AP cluster
- `closest_AP_cluster_size`: the size of the closest AP cluster, this is the number of structures in the AP cluster
- `closest_can_cluster_index`: the closest canonical cluster index, this is the index of the canonical cluster that the closest AP cluster belongs to, and this is a unique integer assigned to each canonical cluster within a LRC group
- `merged_AP_cluster_label`: the merged AP cluster label, this is a unique integer assigned to each AP cluster within a LRC group
- `merged_AP_cluster_exemplar_id`: the merged AP cluster exemplar ID, this is the ID of the structure that is used as the exemplar for the AP cluster
- `merged_AP_cluster_size`: the size of the merged AP cluster, this is the number of structures in the AP cluster
- `merged_can_cluster_index`: the merged canonical cluster index, this is the index of the canonical cluster that the merged AP cluster belongs to, and this is a unique integer assigned to each canonical cluster within a LRC group
- `merge_with_closest_exemplar_torsional`: True if the query CDR conformation is merged with the closest AP cluster measured in torsional space
- `merge_with_any_exemplar_cartesian`: 
  - `True` if the query CDR conformation is merged with an AP cluster measured in Cartesian space
  - `False` otherwise  
  - `null` if searching in Cartesian space was not carried out, i.e. the query CDR conformation is merged with the closest AP cluster measured in torsional space
- "merged": 
  - `True` if the query CDR conformation is merged with an AP cluster`, 
  - `False` otherwise

Note, if a query CDR conformation is not merged with any AP cluster, those fields starts with `merged_` will be `null`, for example:
```JSON
{
        "H1": {
            "closest_lrc": "H1-10-allT",
            "closest_AP_cluster_label": 7,
            "closest_AP_cluster_exemplar_id": "1hkl_0",
            "closest_AP_cluster_size": 1,
            "closest_can_cluster_index": 5,
            "merged_AP_cluster_label": null,
            "merged_AP_cluster_exemplar_id": null,
            "merged_AP_cluster_size": null,
            "merged_can_cluster_index": null,
            "merge_with_closest_exemplar_torsional": false,
            "merge_with_any_exemplar_cartesian": false,
            "merged": false
        }
}
```
