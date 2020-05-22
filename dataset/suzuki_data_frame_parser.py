from logging import getLogger

import numpy
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import rdFMCS, AllChem, Draw, rdchem
from copy import copy

from chainer_chemistry.dataset.parsers.base_parser import BaseFileParser
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset

import traceback
import random



def get_local_submolecule(reactant, product, show_molecules=False, local_dist=3):

    all_product_atoms = {i for i in range(product.GetNumAtoms())}
    # print(product.GetNumAtoms())
    common_atoms = set(product.GetSubstructMatch(reactant))
    missing_indices = all_product_atoms.difference(common_atoms)
    original_missing_atoms = copy(missing_indices)
    # print(missing_indices)
    close_atom_indices = missing_indices
    for i in range(local_dist):
        new_close_atom_indices = set()
        for close_atom_index in close_atom_indices:
            close_atom = product.GetAtomWithIdx(close_atom_index)
            for neighbor_atom in close_atom.GetNeighbors():
                new_close_atom_indices.add(neighbor_atom.GetIdx())
        close_atom_indices = close_atom_indices.union(new_close_atom_indices)
    # print(close_atom_indices)
    """
    far_atom_indices = all_product_atoms.difference(close_atom_indices)
    # print(far_atom_indices)
    editable_product = rdchem.EditableMol(product)
    for index in sorted(far_atom_indices, reverse=True):
        editable_product.RemoveAtom(index)
    close_product = editable_product.GetMol()
    
    for atom in close_product.GetAtoms():
        atom.SetIsAromatic(False)
    
    editable_product = rdchem.EditableMol(product)
    for index in sorted(far_atom_indices.union(original_missing_atoms),
                        reverse=True):
        editable_product.RemoveAtom(index)
    close_reactant = editable_product.GetMol()

    for atom in close_reactant.GetAtoms():
        atom.SetIsAromatic(False)
    
    if show_molecules:
        display(Draw.MolToImage(reactant))
        display(Draw.MolToImage(product))
        display(Draw.MolToImage(close_product))
        display(Draw.MolToImage(close_reactant))

    return close_reactant, close_product
    """
    return close_atom_indices, close_atom_indices  # Mask variable for product 


def randomSmiles(m1):
#     # m = Chem.MolFromSmiles(smiles)
#     ans = list(range(m.GetNumAtoms()))
#     numpy.random.shuffle(ans)
#     nm = Chem.RenumberAtoms(m,ans)
#     return Chem.MolToSmiles(nm, canonical=True, isomericSmiles=True) #, canonical=self.canonical, isomericSmiles=self.isomericSmiles)
    
    m1.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0,m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i,v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(m1)


class SuzukiDataFrameParser(BaseFileParser):
    """data frame parser

    This FileParser parses pandas dataframe.
    It should contain column which contain SMILES as input, and
    label column which is the target to predict.

    Args:
        preprocessor (BasePreprocessor): preprocessor instance
        labels (str or list or None): labels column
        smiles_col (str): smiles column
        postprocess_label (Callable): post processing function if necessary
        postprocess_fn (Callable): post processing function if necessary
        logger:
    """

    def __init__(self, preprocessor,
                 labels=None,
                 smiles_col=['Reactant', 'Product'],
                 postprocess_label=None, postprocess_fn=None,
                 logger=None, label_dicts=None):
        super(SuzukiDataFrameParser, self).__init__(preprocessor)
        if isinstance(labels, str):
            labels = [labels, ]
        self.labels = labels  # type: list
        self.smiles_col = smiles_col
        self.postprocess_label = postprocess_label
        self.postprocess_fn = postprocess_fn
        self.logger = logger or getLogger(__name__)
        self.label_dicts = label_dicts

    def parse(self, df, return_smiles=False, target_index=None,
              return_is_successful=False):
        """parse DataFrame using `preprocessor`

        Label is extracted from `labels` columns and input features are
        extracted from smiles information in `smiles` column.

        Args:
            df (pandas.DataFrame): dataframe to be parsed.
            return_smiles (bool): If set to `True`, smiles list is returned in
                the key 'smiles', it is a list of SMILES from which input
                features are successfully made.
                If set to `False`, `None` is returned in the key 'smiles'.
            target_index (list or None): target index list to partially extract
                dataset. If None (default), all examples are parsed.
            return_is_successful (bool): If set to `True`, boolean list is
                returned in the key 'is_successful'. It represents
                preprocessing has succeeded or not for each SMILES.
                If set to False, `None` is returned in the key 'is_success'.

        Returns (dict): dictionary that contains Dataset, 1-d numpy array with
            dtype=object(string) which is a vector of smiles for each example
            or None.

        """
        logger = self.logger
        pp = self.preprocessor
        smiles_list = [[] for i in range(len(self.smiles_col))]
        is_successful_list = []

        # counter = 0
        if isinstance(pp, MolPreprocessor):
            if target_index is not None:
                df = df.iloc[target_index]
            # import pdb; pdb.set_trace()
            features = None
            smiles_indices = [df.columns.get_loc(c) for c in self.smiles_col]
            # smiles_index1 = df.columns.get_loc(self.smiles_col[0])
            # smiles_index2 = df.columns.get_loc(self.smiles_col[1])
            if self.labels is None:
                labels_index = []  # dummy list
            else:
                labels_index = [df.columns.get_loc(c) for c in self.labels]

            total_count = df.shape[0]
            fail_count = 0
            success_count = 0
            for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
                smiles = [row[i] for i in smiles_indices]
                labels = [row[i] for i in labels_index]
                                
                # Reagent vector for each category
                #  'M', 'L', 'B', 'S', 'A'
                for reagent_i in range(1, 6):
                    if len(labels[reagent_i]) == 2:
                        labels[reagent_i] = []
                    else:
                        labels[reagent_i] = [int(x) for x in labels[reagent_i][1:-1].split(',')]
                    
                # num_dicts = {'M': 28, 'L': 23, 'B': 35, 'S': 10, 'A': 17} # A has nan -> 18
                # num_dicts = {'M': 44, 'L': 47, 'B': 13, 'S': 22, 'A': 74}  # For C-N coupling
                
                boundaries = []
                total_labels = 0
                for lb, lb_num in self.label_dicts.items():
                    total_labels += lb_num
                                                        
                
                # For now, predict single vector
                rea_cat = numpy.zeros(total_labels+len(self.label_dicts)+1, dtype='float32')  # 113 -> 119 (nan + 5 null)
                # rea_cat = numpy.zeros(206, dtype='float32')  # 113 -> 119 (nan + 5 null)
                for ii in range(1, 1+len(self.label_dicts)):
                    rea_cat[labels[ii]] = 1.
                    
                # if the sample does not have reagent in each category, map to null class
                for ii in range(1, 1+len(self.label_dicts)):
                    if len(labels[ii]) == 0:  # no reagent in this category
                        rea_cat[total_labels + ii] = 1.
                        # rea_cat[201 + ii-1] = 1.
                    
                labels = [labels[0], labels[-1], rea_cat]
                
                try:
                    mols = [Chem.MolFromSmiles(s) for s in smiles]
                    # mol1 = Chem.MolFromSmiles(smiles1)
                    # mol2 = Chem.MolFromSmiles(smiles2)
                    if None in mols:
                        fail_count += 1
                        if return_is_successful:
                            is_successful_list.append(False)
                        continue
                        
                    # Note that smiles expression is not unique.
                    # we obtain canonical smiles
                    canonical_smiles = []
                    final_mols = []
                    input_features = []
                    for mol in mols:
                        _canonical_smiles, _mol = pp.prepare_smiles_and_mol(mol) # NON CANONICAL SMILES!!!! 11/21
                        _input_features = pp.get_input_features(mol)  # _mol -> mol (non_canonical smiles)
                        
                        canonical_smiles.append(_canonical_smiles)
                        final_mols.append(mol) # _mol -> mol
                        input_features.append(_input_features)

                    # Extract label
                    if self.postprocess_label is not None:
                        labels[:1] = self.postprocess_label(labels[:1])  # Only yield 

                    if return_smiles:
                        for ii in range(len(canonical_smiles)):
                            smiles_list[ii].append(canonical_smiles[ii])
                        
                except MolFeatureExtractionError as e:
                    # This is expected error that extracting feature failed,
                    # skip this molecule.
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                except Exception as e:
                    logger.warning('parse(), type: {}, {}'
                                   .format(type(e).__name__, e.args))
                    logger.info(traceback.format_exc())
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                
                # Initialize features: list of list
                if features is None:
                    num_features = 0
                    for i in range(len(input_features)):
                        if isinstance(input_features[i], tuple):
                            num_features += len(input_features[i])
                        else:
                            num_features += 1
                            
                    if self.labels is not None:
                        # Catalysts, reagents: additional feature
                        num_features += 2  #1
                    features = [[] for _ in range(num_features)]
                
                for i in range(len(input_features)):
                    if isinstance(input_features[i], tuple):
                        for ii in range(len(input_features[i])):
                            features[i*len(input_features[0])+ii].append(input_features[i][ii])
                    else:
                        features[i].append(input_features[i])
                
                if self.labels is not None:
                    # NOTE THAT LABEL ORDER SHOULD BE YIELD, TEMP, [CATALYSTS, REAGENTS]
                    features[len(features) - 1].append(labels[:2]) # 1->2 when id
                    features[len(features) - 2].append(labels[2:])  # features: smile, smile, [rea,cat], [yield, temp]
                success_count += 1
                if return_is_successful:
                    is_successful_list.append(True)
                
            ret = []

            for feature in features:
                try:
                    feat_array = numpy.asarray(feature)
                except ValueError:
                    # Temporal work around.
                    # See,
                    # https://stackoverflow.com/questions/26885508/why-do-i-get-error-trying-to-cast-np-arraysome-list-valueerror-could-not-broa
                    feat_array = numpy.empty(len(feature), dtype=numpy.ndarray)
                    feat_array[:] = feature[:]
                ret.append(feat_array)
            result = tuple(ret)
            logger.info('Preprocess finished. FAIL {}, SUCCESS {}, TOTAL {}'
                        .format(fail_count, success_count, total_count))
        else:
            raise NotImplementedError

        smileses1 = numpy.array(smiles_list1) if return_smiles else None
        smileses2 = numpy.array(smiles_list2) if return_smiles else None
        
        if return_is_successful:
            is_successful = numpy.array(is_successful_list)
        else:
            is_successful = None

        if isinstance(result, tuple):
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(*result)
            dataset = NumpyTupleDataset(*result)
        else:
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(result)
            dataset = NumpyTupleDataset(result)
        return {"dataset": dataset,
                "smiles": [smileses1, smileses2],
                "is_successful": is_successful}

    def extract_total_num(self, df):
        """Extracts total number of data which can be parsed

        We can use this method to determine the value fed to `target_index`
        option of `parse` method. For example, if we want to extract input
        feature from 10% of whole dataset, we need to know how many samples
        are in a file. The returned value of this method may not to be same as
        the final dataset size.

        Args:
            df (pandas.DataFrame): dataframe to be parsed.

        Returns (int): total number of dataset can be parsed.

        """
        return len(df)
