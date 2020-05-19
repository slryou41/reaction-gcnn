# Requirements

This code supports suzuki dataset only. 
1. Install chainer-chemistry
2. Training command (with gpu)
''python train_suzuki_classification.py -m <METHOD> -e <NUM_EPOCHS> -o <OUTPUT_DIR> -g 0''
3. Testing command (with gpu)
''python predict_suzuki_classification.py -m <METHOD> -i <DIR_WITH_MODEL> -g 0 --load-modelname <FILEPATH_TO_MODEL>''
    