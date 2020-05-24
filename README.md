# Requirements

This code supports suzuki, CN coupling and negishi datasets. 
1. Install chainer-chemistry
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