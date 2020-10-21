# Multi-Label Classification Models for the Prediction of Cross-Coupling Reaction Conditions
Code repository for [Multi-Label Classification Models for the Prediction of Cross-Coupling Reaction Conditions](https://chemrxiv.org/articles/preprint/Multi-Label_Classification_Models_for_the_Prediction_of_Cross-Coupling_Reaction_Conditions/13087769?file=25049036), Michael R. Maser*, Alexander Y. Cui*, Serim Ryou*, Travis J. DeLano, Yisong Yue, Sarah E. Reisman ChemRxiv. 

Including:

# Graph Neural Networks for the Prediction of Substrate-Specific Organic Reaction Conditions
Chainer implementation of [Graph Neural Networks for the Prediction of Substrate-Specific Organic Reaction Conditions](https://arxiv.org/abs/2007.04275), Serim Ryou*, Michael R. Maser*, Alexander Y. Cui*, Travis J. DeLano, Yisong Yue, Sarah E. Reisman, ICML 2020 Graph Representation Learning and Beyond (GRL+) Workshop. arXiv:2007.04275

## Requirements for GNN modeling

1. Install [chainer-chemistry](https://github.com/chainer/chainer-chemistry)
2. Download the dataset by following the [instruction](https://github.com/slryou41/reaction-gcnn/blob/master/data/data_processing_example.ipynb) from Reaxys®
3. This code supports the label dictionary of suzuki, CN coupling, Negishi and PKR datasets.
2. Training command (with gpu)
```python
python train.py -m <METHOD> -e <NUM_EPOCHS> -o <OUTPUT_DIR> -g 0 --data-name <One from suzuki, CN, Negishi or PKR>
```

Example:
```python
python train.py -m relgcn -e 50 -o relgcn_output -g 0 --data-name suzuki
```

3. Testing command (with gpu)
```python
python predict.py -m <METHOD> -i <DIR_WITH_MODEL> -g 0 --load-modelname <FILEPATH_TO_MODEL> --data-name <One from suzuki, CN, Negishi or PKR>
```

Example:
```python
python predict.py -m relgcn -i relgcn_output -g 0 --load-modelname relgcn_output/model_epoch-1 --data-name suzuki
```

4. Modify the path to the result directory in ``convert_to_evaluation_format.ipynb`` and generate json files. 
