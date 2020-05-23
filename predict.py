#!/usr/bin/env python

from __future__ import print_function

import json
import numpy
import os
import pickle
import pandas

import sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..', '..'))
if rootDir not in sys.path: # add parent dir to paths
    sys.path.append(rootDir)
    print(rootDir)

from argparse import ArgumentParser
from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator

from chainer import cuda, serializers
from chainer import Variable
from chainer import functions as F

from chainer_chemistry.models.prediction import Regressor
from models.classifier import Classifier
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict

from dataset.suzuki_csv_file_parser import SuzukiCSVFileParser as CSVFileParser

from chainer_chemistry.datasets import NumpyTupleDataset

from sklearn.preprocessing import StandardScaler  # NOQA
from train import GraphConvPredictor, set_up_predictor  # NOQA

from chainer_chemistry.utils import save_json
import chainer

class ScaledGraphConvPredictor(GraphConvPredictor):
    def __init__(self, *args, **kwargs):
        """Initializes the (scaled) graph convolution predictor. This uses
        a standard scaler to rescale the predicted labels.
        """
        super(ScaledGraphConvPredictor, self).__init__(*args, **kwargs)

    def __call__(self, atoms1, adjs1, atoms2, adjs2, atoms3, adjs3, conds):
        h = super(ScaledGraphConvPredictor, self).__call__(atoms1, adjs1, atoms2, adjs2, atoms3, adjs3, conds)
        
        scaler_available = hasattr(self, 'scaler')
        numpy_data = isinstance(h.data, numpy.ndarray)
        if scaler_available:
            h = self.scaler.inverse_transform(cuda.to_cpu(h.data))
            if not numpy_data:
                h = cuda.to_gpu(h)
        return h #Variable(h)


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn',
                   'relgat']
    scale_list = ['standardize', 'none']

    # Set up the argument parser.
    parser = ArgumentParser(description='Regression on own dataset')
#     parser.add_argument('--datafile', '-d', type=str,
#                         # default='oxidation_test.csv',
#                         default='data/suzuki_type_test_v2.csv',
#                         # default='CN_coupling_test.csv',
#                         help='csv file containing the dataset')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='nfp')
#     parser.add_argument('--label', '-l', nargs='+',
#                         # default=['Yield', 'Temperature', 'Reagent', 'Catalyst'],
#                         default=['Yield', 'M', 'L', 'B', 'S', 'A', 'id'],
#                         help='target label for regression')
    parser.add_argument('--scale', type=str, choices=scale_list,
                        help='label scaling method', default='standardize')
    parser.add_argument('--conv-layers', '-c', type=int, default=4,
                        help='number of convolution layers')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='id of gpu to use; negative value means running'
                        'the code on cpu')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to save the computed model to')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--unit-num', '-u', type=int, default=128,
                        help='number of units in one layer of the model')
    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')
    parser.add_argument('--in-dir', '-i', type=str, default='result',
                        help='directory containing the saved model')
    parser.add_argument('--model-filename', type=str, default='classifier.pkl',
                        help='saved model filename')
    parser.add_argument('--load-modelname', type=str,
                        help='load model filename')
    parser.add_argument('--data-name', type=str, default='suzuki',
                        help='dataset name')
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()
    device = args.gpu
    method = args.method
    
    if args.data_name == 'suzuki':
        datafile = 'data/suzuki_type_test_v2.csv'
        class_num = 119
        class_dict = {'M': 28, 'L': 23, 'B': 35, 'S': 10, 'A': 17}
        dataset_filename = 'test_data.npz'
        labels = ['Yield', 'M', 'L', 'B', 'S', 'A', 'id']
    elif args.data_name == 'CN':
        datafile = 'data/CN_coupling_test.csv'
        class_num = 206
        class_dict = {'M': 44, 'L': 47, 'B': 13, 'S': 22, 'A': 74}
        dataset_filename = 'test_CN_data.npz'
        labels = ['Yield', 'M', 'L', 'B', 'S', 'A', 'id']
    elif args.data_name == 'Neigishi':
        datafile = 'data/Neigishi_test.csv'
        class_num = 106
        class_dict = {'M': 32, 'L': 20, 'T': 8, 'S': 10, 'A': 30}
        dataset_filename = 'test_Neigishi_data.npz'
        labels = ['Yield', 'M', 'L', 'T', 'S', 'A', 'id']
    elif args.data_name == 'PKR':
        datafile = 'data/PKR_test.csv'
        class_num = 83
        class_dict = {'M': 18, 'L': 6, 'T': 7, 'S': 15, 'A': 11, 'G': 1, 'O': 13, 'P': 4, 'other': 1}
        dataset_filename = 'PKR_data.npz'
        labels = ['Yield', 'M', 'L', 'T', 'S', 'A', 'G', 'O', 'P', 'other', 'id']
    else:
        raise ValueError('Unexpected dataset name')
    
    cache_dir = os.path.join('input', '{}_all'.format(method))
    
    # Dataset preparation.
    def postprocess_label(label_list):
        return numpy.asarray(label_list, dtype=numpy.float32)

    print('Preprocessing dataset...')
    
    # Load the cached dataset.
    dataset_cache_path = os.path.join(cache_dir, dataset_filename)
    
    dataset = None
    if os.path.exists(dataset_cache_path):
        print('Loading cached dataset from {}.'.format(dataset_cache_path))
        dataset = NumpyTupleDataset.load(dataset_cache_path)
    if dataset is None:
        preprocessor = preprocess_method_dict[args.method]()
        parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label,
                              labels=labels, smiles_col=['Reactant1', 'Reactant2', 'Product'],
                              label_dicts=class_dict)
        dataset = parser.parse(datafile)['dataset']
        
        # Cache the laded dataset.
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        NumpyTupleDataset.save(dataset_cache_path, dataset)
    
    labels = dataset.get_datasets()[-2]
    ids = dataset.get_datasets()[-1][:,1].reshape(-1,1)
    yields = dataset.get_datasets()[-1][:,0].reshape(-1,1).astype('float32') # [:,0] added
    dataset = NumpyTupleDataset(*(dataset.get_datasets()[:-2] + (yields, labels,)))

    # Load the standard scaler parameters, if necessary.
    scaler = None    
    test = dataset

    print('Predicting...')
    # Set up the regressor.
    model_path = os.path.join(args.in_dir, args.model_filename)
    
    if os.path.exists(model_path):
        classifier = Classifier.load_pickle(model_path, device=args.gpu)
    else:
        predictor = set_up_predictor(args.method, args.unit_num,
                                 args.conv_layers, class_num)
        classifier = Classifier(predictor, lossfun=F.sigmoid_cross_entropy,
                            metrics_fun=F.binary_accuracy, device=args.gpu)
    
    if args.load_modelname:
        serializers.load_npz(args.load_modelname, classifier)
    scaled_predictor = ScaledGraphConvPredictor(graph_conv=classifier.predictor.graph_conv, mlp=classifier.predictor.mlp)
    classifier.predictor = scaled_predictor
    
    # This callback function extracts only the inputs and discards the labels.
    def extract_inputs(batch, device=None):
        return concat_mols(batch, device=device)[:-1]
    
    # Predict the output labels.
    # Prediction function rewrite!!!
    y_pred = classifier.predict(
        test, converter=extract_inputs)
    y_pred_max = numpy.argmax(y_pred, axis=1)
    y_pred_max = y_pred_max.reshape(-1, 1)
    # y_pred_idx = y_pred.argsort(axis=1) # ascending
    
    # Extract the ground-truth labels.
    t = concat_mols(test, device=-1)[-1]  # device 11/14 memory issue
    original_t = cuda.to_cpu(t)
    t_idx = original_t.squeeze(1)
    t_idx = t_idx.argsort(axis=1)
    # gt_indx = numpy.where(original_t == 1)
    
    # Construct dataframe.
    df_dict = {}
    for i, l in enumerate(labels[:1]):
        df_dict.update({'y_pred_{}'.format(l): y_pred_max[:,-1].tolist(), # [:,-1]
                        't_{}'.format(l): t_idx[:, -1].tolist(), })
    df = pandas.DataFrame(df_dict)

    # Show a prediction/ground truth table with 5 random examples.
    print(df.sample(5))

    n_eval = 10
    
    for target_label in range(y_pred_max.shape[1]):
        label_name = labels[:1][0][target_label]
        print('label_name = {}, y_pred = {}, t = {}'
              .format(label_name, y_pred_max[:n_eval, target_label],
                      t_idx[:n_eval, -1]))
        
    # Perform the prediction.
    print('Evaluating...')
    test_iterator = SerialIterator(test, 16, repeat=False, shuffle=False)
    eval_result = Evaluator(test_iterator, classifier, converter=concat_mols,
                            device=args.gpu)()
    print('Evaluation result: ', eval_result)

    with open(os.path.join(args.in_dir, 'eval_result.json'), 'w') as f:
        json.dump(eval_result, f)
        
    res_dic = {}
    for i in range(len(y_pred)):
        res_dic[i] = str(ids[i])
    json.dump(res_dic, open(os.path.join(args.in_dir, "test_ids.json"), "w"))
    
    pickle.dump(y_pred, open(os.path.join(args.in_dir, "pred.pkl"), "wb"))
    pickle.dump(original_t, open(os.path.join(args.in_dir, "gt.pkl"), "wb"))


if __name__ == '__main__':
    main()
