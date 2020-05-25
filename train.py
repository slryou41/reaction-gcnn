#!/usr/bin/env python

from __future__ import print_function

import chainer
import numpy
import os
import pickle

import sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..', '..'))
if rootDir not in sys.path: # add parent dir to paths
    sys.path.append(rootDir)
    print(rootDir)

from argparse import ArgumentParser
from chainer.datasets import split_dataset_random
from chainer import cuda, serializers
from chainer import functions as F
from chainer import optimizers
from chainer import training
from chainer import Variable
from chainer.iterators import SerialIterator
from chainer.training import extensions as E
from sklearn.preprocessing import StandardScaler
from chainer import iterators

import chainer_chemistry
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.models import (
    MLP, GGNN, MPNN, SchNet, WeaveNet, RSGCN, RelGAT)  # for future use..

from models import RelGCN, Classifier, NFP

from dataset.suzuki_csv_file_parser import SuzukiCSVFileParser as CSVFileParser


class GraphConvPredictor(chainer.Chain):
    def __init__(self, graph_conv, mlp=None):
        """Initializes the graph convolution predictor.

        Args:
            graph_conv: The graph convolution network required to obtain
                        molecule feature representation.
            mlp: Multi layer perceptron; used as the final fully connected
                 layer. Set it to `None` if no operation is necessary
                 after the `graph_conv` calculation.
        """

        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            if isinstance(mlp, chainer.Link):
                self.mlp = mlp
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp

    def __call__(self, atoms1, adjs1, atoms2, adjs2, atoms3, adjs3, conds):
        
        h1 = self.graph_conv(atoms1, adjs1)
        h2 = self.graph_conv(atoms2, adjs2)
        h3 = self.graph_conv(atoms3, adjs3)
        h = F.concat((h1, h2, h3), axis=1)
        
        if self.mlp:
            h = self.mlp(h)
        return h
    
    
class MeanAbsError(object):
    def __init__(self, scaler=None):
        """Initializes the (scaled) mean absolute error metric object.

        Args:
            scaler: Standard label scaler.
        """
        self.scaler = scaler

    def __call__(self, x0, x1):
        if isinstance(x0, Variable):
            x0 = cuda.to_cpu(x0.data)
        if isinstance(x1, Variable):
            x1 = cuda.to_cpu(x1.data)
        if self.scaler is not None:
            scaled_x0 = self.scaler.inverse_transform(cuda.to_cpu(x0))
            scaled_x1 = self.scaler.inverse_transform(cuda.to_cpu(x1))
            diff = scaled_x0 - scaled_x1
        else:
            diff = cuda.to_cpu(x0) - cuda.to_cpu(x1)
        return numpy.mean(numpy.absolute(diff), axis=0)[0]


class RootMeanSqrError(object):
    def __init__(self, scaler=None):
        """Initializes the (scaled) root mean square error metric object.

        Args:
            scaler: Standard label scaler.
        """
        self.scaler = scaler

    def __call__(self, x0, x1):
        if isinstance(x0, Variable):
            x0 = cuda.to_cpu(x0.data)
        if isinstance(x1, Variable):
            x1 = cuda.to_cpu(x1.data)
        if self.scaler is not None:
            scaled_x0 = self.scaler.inverse_transform(cuda.to_cpu(x0))
            scaled_x1 = self.scaler.inverse_transform(cuda.to_cpu(x1))
            diff = scaled_x0 - scaled_x1
        else:
            diff = cuda.to_cpu(x0) - cuda.to_cpu(x1)
        return numpy.sqrt(numpy.mean(numpy.power(diff, 2), axis=0)[0])


class ScaledAbsError(object):
    def __init__(self, scaler=None):
        """Initializes the (scaled) absolute error object.

        Args:
            scaler: Standard label scaler.
        """
        self.scaler = scaler

    def __call__(self, x0, x1):
        if isinstance(x0, Variable):
            x0 = cuda.to_cpu(x0.data)
        if isinstance(x1, Variable):
            x1 = cuda.to_cpu(x1.data)
        if self.scaler is not None:
            scaled_x0 = self.scaler.inverse_transform(cuda.to_cpu(x0))
            scaled_x1 = self.scaler.inverse_transform(cuda.to_cpu(x1))
            diff = scaled_x0 - scaled_x1
        else:
            diff = cuda.to_cpu(x0) - cuda.to_cpu(x1)
        return numpy.mean(numpy.absolute(diff), axis=0)[0]


def set_up_predictor(method, n_unit, conv_layers, class_num):
    """Sets up the graph convolution network  predictor.

    Args:
        method: Method name. Currently, the supported ones are `nfp`, `ggnn`,
                `schnet`, `weavenet` and `rsgcn`.
        n_unit: Number of hidden units.
        conv_layers: Number of convolutional layers for the graph convolution
                     network.
        class_num: Number of output classes.

    Returns:
        An instance of the selected predictor.
    """

    predictor = None
    mlp = MLP(out_dim=class_num, hidden_dim=n_unit)
    # if method == 'weavenet':
    #     mlp = MLP(out_dim=class_num, hidden_dim=n_unit) #100+61)
    
    if method == 'nfp':
        print('Training an NFP predictor...')
        if chainer_chemistry.__version__ == '0.7.0':
          nfp = NFP(out_dim=n_unit, hidden_channels=n_unit, n_update_layers=conv_layers)
        else:
          nfp = NFP(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
        predictor = GraphConvPredictor(nfp, mlp)
    elif method == 'ggnn':
        print('Training a GGNN predictor...')
        if chainer_chemistry.__version__ == '0.7.0':
          ggnn = GGNN(out_dim=n_unit, hidden_channels=n_unit, n_update_layers=conv_layers)
        else:
          ggnn = GGNN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
        predictor = GraphConvPredictor(ggnn, mlp)
    elif method == 'mpnn':
        print('Training a MPNN predictor...')
        if chainer_chemistry.__version__ == '0.7.0':
          mpnn = MPNN(out_dim=n_unit, hidden_channels=n_unit, n_update_layers=conv_layers)
        else:
          mpnn = MPNN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
        predictor = GraphConvPredictor(mpnn, mlp)
    elif method == 'schnet':
        print('Training an SchNet predictor...')
        if chainer_chemistry.__version__ == '0.7.0':
          schnet = SchNet(out_dim=class_num, hidden_channels=n_unit,
                          n_update_layers=conv_layers)
        else:
          schnet = SchNet(out_dim=class_num, hidden_dim=n_unit,
                          n_layers=conv_layers)

        predictor = GraphConvPredictor(schnet, None)
    elif method == 'weavenet':
        print('Training a WeaveNet predictor...')
        n_atom = 20
        n_sub_layer = 1
        weave_channels = [50] * conv_layers

        weavenet = WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
                            n_sub_layer=n_sub_layer) #, n_atom=n_atom)
        predictor = GraphConvPredictor(weavenet, mlp)
    elif method == 'rsgcn':
        print('Training an RSGCN predictor...')
        if chainer_chemistry.__version__ == '0.7.0':
          rsgcn = RSGCN(out_dim=n_unit, hidden_channels=n_unit, n_update_layers=conv_layers)
        else:
          rsgcn = RSGCN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
        predictor = GraphConvPredictor(rsgcn, mlp)
    elif method == 'relgcn':
        print('Training an RelGCN predictor...')
        num_edge_type = 4

        relgcn = RelGCN(out_channels=n_unit, num_edge_type=num_edge_type,
                        scale_adj=True, readout=True)
        predictor = GraphConvPredictor(relgcn, mlp)
    elif method == 'relgat':
        print('Training an RelGAT predictor...')
        if chainer_chemistry.__version__ == '0.7.0':
          relgat = RelGAT(out_dim=n_unit, hidden_channels=n_unit,
                          n_update_layers=conv_layers)
        else:
          relgat = RelGAT(out_dim=n_unit, hidden_dim=n_unit,
                          n_layers=conv_layers)
        predictor = GraphConvPredictor(relgat, mlp)
    else:
        raise ValueError('[ERROR] Invalid method: {}'.format(method))
    return predictor


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'mpnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn',
                   'relgat']
    scale_list = ['standardize', 'none']

    # Set up the argument parser.
    parser = ArgumentParser(description='Regression on own dataset')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='nfp')
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
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--unit-num', '-u', type=int, default=128,  #16
                        help='number of units in one layer of the model')
    parser.add_argument('--seed', '-s', type=int, default=777,
                        help='random seed value')
    parser.add_argument('--train-data-ratio', '-r', type=float, default=0.9, #0.7,
                        help='ratio of training data w.r.t the dataset')
    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')
    parser.add_argument('--model-filename', type=str, default='classifier.pkl',
                        help='saved model filename')
    parser.add_argument('--data-name', type=str, default='suzuki',
                        help='dataset name')
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()
    method = args.method
    
    if args.data_name == 'suzuki':
        datafile = 'data/suzuki_type_train_v2.csv'
        class_num = 119
        class_dict = {'M': 28, 'L': 23, 'B': 35, 'S': 10, 'A': 17}
        dataset_filename = 'data.npz'
        labels = ['Yield', 'M', 'L', 'B', 'S', 'A', 'id']
    elif args.data_name == 'CN':
        datafile = 'data/CN_coupling_train.csv'
        class_num = 206
        class_dict = {'M': 44, 'L': 47, 'B': 13, 'S': 22, 'A': 74}
        dataset_filename = 'CN_data.npz'
        labels = ['Yield', 'M', 'L', 'B', 'S', 'A', 'id']
    elif args.data_name == 'Negishi':
        datafile = 'data/Negishi_train.csv'
        class_num = 106
        class_dict = {'M': 32, 'L': 20, 'T': 8, 'S': 10, 'A': 30}
        dataset_filename = 'Negishi_data.npz'
        labels = ['Yield', 'M', 'L', 'T', 'S', 'A', 'id']
    elif args.data_name == 'PKR':
        datafile = 'data/PKR_train.csv'
        class_num = 83
        class_dict = {'M': 18, 'L': 6, 'T': 7, 'S': 15, 'A': 11, 'G': 1, 'O': 13, 'P': 4, 'other': 1}
        dataset_filename = 'PKR_data.npz'
        labels = ['Yield', 'M', 'L', 'T', 'S', 'A', 'G', 'O', 'P', 'other', 'id']
    else:
        raise ValueError('Unexpected dataset name')
    
    cache_dir = os.path.join('input', '{}_all'.format(method))
    
    # Dataset preparation. Postprocessing is required for the regression task.
    def postprocess_label(label_list):
        return numpy.asarray(label_list, dtype=numpy.float32)
    
    # Load the cached dataset.
    dataset_cache_path = os.path.join(cache_dir, dataset_filename)

    dataset = None
    if os.path.exists(dataset_cache_path):
        print('Loading cached dataset from {}.'.format(dataset_cache_path))
        dataset = NumpyTupleDataset.load(dataset_cache_path)
    if dataset is None:
        print('Preprocessing dataset...')
        preprocessor = preprocess_method_dict[args.method]()
        parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label,
                               labels=labels, smiles_col=['Reactant1', 'Reactant2', 'Product'],
                               label_dicts=class_dict)

        # Load the entire dataset.
        dataset = parser.parse(datafile)['dataset']

        # Cache the laded dataset.
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        NumpyTupleDataset.save(dataset_cache_path, dataset)
    
    # Scale the label values, if necessary.
    if args.scale == 'standardize':
        scaler = StandardScaler()
        labels = dataset.get_datasets()[-2]
        # yields = dataset.get_datasets()[-1]
        yields = dataset.get_datasets()[-1][:,0].reshape(-1,1).astype('float32')
        
        # Filter index here
        # range_exp = (0.0 <= yields) & (yields <= 1.0)
        range_exp = numpy.argsort(yields[:,0])  # ascending
        start_len = 0
        end_len = len(yields) #int(len(yields) / 4)
        range_exp = range_exp[start_len:end_len]        
        
        range_dataset = (dataset.get_datasets()[0][range_exp],
                         dataset.get_datasets()[1][range_exp],
                         dataset.get_datasets()[2][range_exp],
                         dataset.get_datasets()[3][range_exp],
                         dataset.get_datasets()[4][range_exp],
                         dataset.get_datasets()[5][range_exp])
        yields = yields[range_exp]
        labels = labels[range_exp]
        
        dataset = NumpyTupleDataset(*(range_dataset + (yields, labels,)))
        
    else:
        scaler = None

    # Split the dataset into training and validation.
    train_data_size = int(len(dataset) * args.train_data_ratio)
    train, valid = split_dataset_random(dataset, train_data_size, args.seed)

    # Set up the predictor.
    predictor = set_up_predictor(args.method, args.unit_num,
                                 args.conv_layers, class_num)

    # Set up the iterator.
    train_iter = iterators.SerialIterator(train, args.batchsize)
    valid_iter = iterators.SerialIterator(valid, args.batchsize, repeat=False,
                                          shuffle=False)
    
    # Set up the regressor.
    device = args.gpu
    
    classifier = Classifier(predictor, lossfun=F.sigmoid_cross_entropy,
                            metrics_fun=F.binary_accuracy, device=args.gpu)
    
    # Set up the optimizer.
    optimizer = optimizers.Adam()
    optimizer.setup(classifier)
    
    # Set up the updater.
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       converter=concat_mols)

    # Set up the trainer.
    print('Training...')
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(E.Evaluator(valid_iter, classifier, device=device,
                               converter=concat_mols))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.snapshot_object(classifier, filename='model_epoch-{.updater.epoch}'))  # save every epoch
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport([
        'epoch', 'main/loss', 'main/accuracy', 'validation/main/loss',
        'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(E.ProgressBar())
    trainer.run()

    # Save the regressor's parameters.
    model_path = os.path.join(args.out, args.model_filename)
    print('Saving the trained model to {}...'.format(model_path))
    classifier.save_pickle(model_path, protocol=args.protocol)

    # Save the standard scaler's parameters.
    if scaler is not None:
        with open(os.path.join(args.out, 'scaler.pkl'), mode='wb') as f:
            pickle.dump(scaler, f, protocol=args.protocol)


if __name__ == '__main__':
    main()
