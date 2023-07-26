Place folders required by `classify_general_abdb_entry.py` inside this directory or simply create soft links to them. 

- `ABDB `: a snapshot of AbDb database, the version used in the publication is `20220926`
- `classifier`: CDR conformation classifiers output by this study, each classifier is a scikit-learn AP AffinityPropagation object, packed as `joblib` file, for details refer to scikit-learn documentation at [here](https://scikit-learn.org/stable/model_persistence.html)
- `LRC_AP_cluster.json`: a JSON file containing information about LRC groups, Canonical clusters, and AP clusters 