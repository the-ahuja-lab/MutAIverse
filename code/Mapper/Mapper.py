import io
from importlib import resources
import pkg_resources
import zipfile
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from matchms.importing import load_from_msp, load_from_mzml
from matchms.filtering import default_filters, normalize_intensities,select_by_mz
from matchms import calculate_scores
from matchms.similarity import CosineGreedy
import csv
import time
from rdkit.Chem import rdFMCS
from matplotlib import colors
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
from rdkit import DataStructs
from PIL import Image
from io import BytesIO
from IPython.display import display
import numpy as np

from typing import Optional
from collections import defaultdict
import re
from scipy.sparse import csr_matrix, save_npz,load_npz
import gensim
from matchms.Spectrum import Spectrum
from matchms.typing import SpectrumType
import pickle
from typing import List
from typing import Union
from gensim.models import Word2Vec
from gensim.models.basemodel import BaseTopicModel
import hnswlib
import sqlite3
from itertools import chain

def fast_high(row, ref_id_column='E0_ref_id', smile_column='E0_ref_smile'):
    def get_mol(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return mol

    def find_matches_one(mol, submol):
        match_dict = {}
        mols = [mol, submol]
        res = rdFMCS.FindMCS(mols)
        mcsp = Chem.MolFromSmarts(res.smartsString)
        matches = mol.GetSubstructMatches(mcsp)
        return matches

    adduct_mol = get_mol(row[smile_column])

    # Define base da strings
    da_base_A = 'OCC1OC(CC1O)n1cnc2c1nc(N)nc2N'
    da_base_T = 'C1C(C(OC1N2C=NC3=C(N=C(N=C32)N)N)CO)O'
    da_base_G = 'NC1=NC2=C(N=CN2[C@H]2C[C@H](O)[C@@H](CO)O2)C(=O)N1'
    da_base_C = 'C1C(C(OC1N2C=CC(=NC2=O)N)CO)O'

    # Check conditions based on ref_id_column prefix
    if row[ref_id_column].startswith("gen_A"):
        da = da_base_A
    elif row[ref_id_column].startswith("gen_T"):
        da = da_base_T
    elif row[ref_id_column].startswith("gen_G"):
        da = da_base_G
    elif row[ref_id_column].startswith("gen_C"):
        da = da_base_C
    else :
        da =adduct_mol
    da_mol = get_mol(da)
    matches = find_matches_one(adduct_mol, da_mol)
    return adduct_mol

#normalize spectra 
def spectra_preprocess(s):
    s = default_filters(s)
    s = normalize_intensities(s)
    #s = select_by_mz(s,220,640)
    return s

def match_spectra(query_spectra, reference_spectra):
    batch_size = 10
    rows = []

    for i in tqdm(range(0, len(query_spectra), batch_size)):
        utm_batch = query_spectra[i:i + batch_size]
        scores = calculate_scores(references=reference_spectra,
                                  queries=utm_batch,
                                  similarity_function=CosineGreedy(),
                                  is_symmetric=False)

        for j, query_spectrum in enumerate(utm_batch):
            best_matches = scores.scores_by_query(query_spectrum,True) #'CosineGreedy_score',)
            for (reference, score) in best_matches[:1]: #len(reference_spectra)]:
                query_num = f"{i + j}"
                query_id=query_spectra[i+j].metadata['title']
                #query_smile=query_spectra[i+j].metadata['smiles/inchi']
                #query_collision_energy=query_spectra[i+j].metadata['comment']
                reference_id = reference.metadata['id']
                #seed = reference.metadata['seed']
                smiles_inchi = reference.metadata['smiles/inchi']
                Collision_Energy = reference.metadata['comment']
                score_value = f"{score[0]:.4f}"
                num_matching_peaks = score[1]
                rows.append([query_num,query_id, reference_id, smiles_inchi, Collision_Energy, score_value, num_matching_peaks])
                # if score[0] > 0.9:
                #     print(f"Found hit: {query_num}, {library}, Reference ID={reference_id}, Smiles/InChI={smiles_inchi}, {Collision_Energy}, Score={score_value}, Number of Matching Peaks={num_matching_peaks}")

    df = pd.DataFrame(rows, columns=["Query","q_id", "Reference Scan ID", "ref_Smiles/Inchi", "ref_Collision_Energy", "Score", "Number of Matching Peaks"])
    return df
    


#filter MS spectra based on metadata
def filter_spectra(input_list, metadata_key, metadata_value):
    new_list = []
    for element in input_list:
        if element.metadata.get(metadata_key) == metadata_value:
            new_list.append(element)
    return new_list
    
# MS/MS library selection
def map(library,mzml_file_path,MS_level,plot):
    # Get a list of all files in the specified folder
    mzml_file= os.path.basename(mzml_file_path)
    if library=='mutaiverse':
        lib_path='mutaiverse.msp'
        output_file = os.path.join(f'{mzml_file}_MutAIverse_results.csv')
    elif library=='bonafide_adducts':
        lib_path='exp_279.msp'
        output_file = os.path.join(f'{mzml_file}_bonafide_adducts_results.csv')
    elif library=='suspected_adducts':
        lib_path='suspected_adducts_303.msp'
        output_file = os.path.join(f'{mzml_file}_suspected_adducts_results.csv')
    else:
        print('Choose valid MS/MS library')

    src_path=pkg_resources.get_distribution('MutAIverse').location
    ref=list(load_from_msp(src_path+'/MutAIverse/'+lib_path))
    refs=[spectra_preprocess(s) for s in ref]
    lib=refs

    lib_energy0=filter_spectra(lib,'comment','Energy0')
    lib_energy1=filter_spectra(lib,'comment','Energy1')
    lib_energy2=filter_spectra(lib,'comment','Energy2')
    
    start_time = time.time()
    
    # Process each mzML file (replace 'spectrum_processing_function' with your actual processing function)
    ut_s=list(load_from_mzml(mzml_file_path,MS_level))
    ut = [spectra_preprocess(s) for s in ut_s]
    print('****calculating cosine similarity****')
    df_energy0 = match_spectra(ut, lib_energy0)
    df_energy1 = match_spectra(ut, lib_energy1)
    df_energy2 = match_spectra(ut, lib_energy2)
    
    
    df_e0 = df_energy0
    df_e0.rename(columns={'Reference Scan ID': 'E0_ref_id'}, inplace=True)
    df_e0.rename(columns={'ref_Smiles/Inchi': 'E0_ref_smile'}, inplace=True)
    df_e0.rename(columns={'Score':'Score_E0'}, inplace=True)


    df_e1=df_energy1
    df_e1.rename(columns={'Reference Scan ID': 'E1_ref_id'}, inplace=True)
    df_e1.rename(columns={'ref_Smiles/Inchi': 'E1_ref_smile'}, inplace=True)
    df_e1.rename(columns={'Score':'Score_E1'}, inplace=True)



    df_e2=df_energy2
    df_e2.rename(columns={'Reference Scan ID': 'E2_ref_id'}, inplace=True)
    df_e2.rename(columns={'ref_Smiles/Inchi': 'E2_ref_smile'}, inplace=True)
    df_e2.rename(columns={'Score':'Score_E2'}, inplace=True)

    df_comb=pd.concat([df_e0['Query'],df_e0['q_id'],df_e0['Score_E0'], df_e0['E0_ref_smile'],df_e0['E0_ref_id'],df_e0['Number of Matching Peaks'],df_e1['Score_E1'], df_e1['E1_ref_smile'],df_e1['E1_ref_id'],df_e1['Number of Matching Peaks'],df_e2['Score_E2'], df_e2['E2_ref_smile'],df_e2['E2_ref_id'],df_e2['Number of Matching Peaks']],axis=1)
    
    # Save results for each mzML file
    df_comb.to_csv(output_file)
    
    if plot==True :
        print('****saving plots****')
        df=pd.read_csv(output_file)
        df['Score_E0'] = df['Score_E0'].astype(float)
        df['Score_E1'] = df['Score_E1'].astype(float)
        df['Score_E2'] = df['Score_E2'].astype(float)
    # Plot histograms for each mzML file
        scores_to_plot = ['Score_E0', 'Score_E1', 'Score_E2']
        num_bins = 20
        colors = ['skyblue', 'lightgreen', 'coral']

        for i, score_column in enumerate(scores_to_plot):
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=score_column, bins=num_bins, kde=True, color=colors[i])
            plt.xlabel('cosine score',weight='bold')
            plt.ylabel("Frequency",weight='bold')
            plt.title(f"{score_column} -Distribution of cosine similarity: {mzml_file} vs DNA Adducts)")
            plt.grid(True)
            plt.xlim(0,1)
            plt.savefig(os.path.join(f'{mzml_file}_{score_column}_histogram.png'))
            plt.close()
    
        
    else:
        PandasTools.RenderImagesInAllDataFrames(images=True)
        df_comb['STRUCTURE_EO']=df_comb.apply(fast_high, axis=1,ref_id_column='E0_ref_id', smile_column='E0_ref_smile')
        df_comb['STRUCTURE_E1']= df_comb.apply(fast_high, axis=1,ref_id_column='E1_ref_id', smile_column='E1_ref_smile')
        df_comb['STRUCTURE_E2']= df_comb.apply(fast_high, axis=1,ref_id_column='E2_ref_id', smile_column='E2_ref_smile')
        return df_comb


def slow_high(row, ref_id_column='COMPID', smile_column='SMILES'):
    def get_mol(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return mol

    def find_matches_one(mol, submol):
        match_dict = {}
        mols = [mol, submol]
        res = rdFMCS.FindMCS(mols)
        mcsp = Chem.MolFromSmarts(res.smartsString)
        matches = mol.GetSubstructMatches(mcsp)
        return matches

    adduct_mol = get_mol(row[smile_column])

    # Define base da strings
    da_base_A =  'OCC1OC(CC1O)n1cnc2c1nc(N)nc2N'
    da_base_T = 'C1C(C(OC1N2C=NC3=C(N=C(N=C32)N)N)CO)O'
    da_base_G = 'NC1=NC2=C(N=CN2[C@H]2C[C@H](O)[C@@H](CO)O2)C(=O)N1'
    da_base_C = 'C1C(C(OC1N2C=CC(=NC2=O)N)CO)O'

    # Check conditions based on ref_id_column prefix
    if row[ref_id_column].startswith("gen_A"):
        da = da_base_A
    elif row[ref_id_column].startswith("gen_T"):
        da = da_base_T
    elif row[ref_id_column].startswith("gen_G"):
        da = da_base_G
    elif row[ref_id_column].startswith("gen_C"):
        da = da_base_C

    da_mol = get_mol(da)
    matches = find_matches_one(adduct_mol, da_mol)
    return adduct_mol

def load_database(db_file):
    # db_file = 'data/IN_SILICO_LIBRARY1.db'
    cursor = sqlite3.connect(db_file)
    content = cursor.execute('SELECT * from IN_SILICO_LIBRARY')
    data = [row for row in content]
    return data

class Spikes:
    """
    Stores arrays of intensities and M/z values, with some checks on their internal consistency.
    """
    def __init__(self, mz=None, intensities=None):
        assert isinstance(mz, np.ndarray), "Input argument 'mz' should be a np.array."
        assert isinstance(intensities, np.ndarray), "Input argument 'intensities' should be a np.array."
        assert mz.shape == intensities.shape, "Input arguments 'mz' and 'intensities' should be the same shape."
        assert mz.dtype == "float", "Input argument 'mz' should be an array of type float."
        assert intensities.dtype == "float", "Input argument 'intensities' should be an array of type float."

        self._mz = mz
        self._intensities = intensities

        assert self._is_sorted(), "mz values are out of order."

    def __eq__(self, other):
        return \
            self.mz.shape == other.mz.shape and \
            np.allclose(self.mz, other.mz) and \
            self.intensities.shape == other.intensities.shape and \
            np.allclose(self.intensities, other.intensities)

    def __len__(self):
        return self._mz.size

    def __getitem__(self, item):
        return [self.mz, self.intensities][item]

    def _is_sorted(self):
        return np.all(self.mz[:-1] <= self.mz[1:])

    def clone(self):
        return Spikes(self.mz, self.intensities)

    @property
    def mz(self):
        """getter method for mz private variable"""
        return self._mz.copy()

    @property
    def intensities(self):
        """getter method for intensities private variable"""
        return self._intensities.copy()

    @property
    def to_np(self):
        """getter method to return stacked np array of both peak mz and
        intensities"""
        return np.vstack((self.mz, self.intensities)).T


class Spectrum:

    def __init__(self, mz: np.array, intensities: np.array, metadata: Optional[dict] = None):
        """
        Parameters
        ----------
        mz
            Array of m/z for the peaks
        intensities
            Array of intensities for the peaks
        metadata
            Dictionary with for example the scan number of precursor m/z.
        """
        self.peaks = Spikes(mz=mz, intensities=intensities)
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

    def __eq__(self, other):
        return \
            self.peaks == other.peaks and \
            self.__metadata_eq(other.metadata)

    def __metadata_eq(self, other_metadata):
        if self.metadata.keys() != other_metadata.keys():
            return False
        for i, value in enumerate(list(self.metadata.values())):
            if isinstance(value, np.ndarray):
                if not np.all(value == list(other_metadata.values())[i]):
                    return False
            elif value != list(other_metadata.values())[i]:
                return False
        return True

    def clone(self):
        """Return a deepcopy of the spectrum instance."""
        clone = Spectrum(mz=self.peaks.mz,
                         intensities=self.peaks.intensities,
                         metadata=self.metadata)
        clone.losses = self.losses
        return clone

    def get(self, key: str, default=None):

        return self._metadata.copy().get(key, default)

    def set(self, key: str, value):

        self._metadata[key] = value
        return self

    @property
    def metadata(self):
        return self._metadata.copy()

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def peaks(self) -> Spikes:
        return self._peaks.clone()

    @peaks.setter
    def peaks(self, value: Spikes):
        self._peaks = value
    
class Document:
    """
    Use this as parent class to build your own document class. An example used for
    mass spectra is SpectrumDocument."""
    def __init__(self, obj):
        """
        Parameters
        ----------
        obj:
            Input object of desired class.
        """
        self._obj = obj
        self._index = 0
        self._make_words()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.words)

    def __next__(self):
        """gensim.models.Word2Vec() wants its corpus elements to be iterable"""
        if self._index < len(self.words):
            word = self.words[self._index]
            self._index += 1
            return word
        self._index = 0
        raise StopIteration

    def __str__(self):
        return self.words.__str__()

    def _make_words(self):
        print("You should override this method in your own subclass.")
        self.words = []
        return self

class SpectrumDocument(Document):
    """Create documents from spectra.

    Every peak (and loss) positions (m/z value) will be converted into a string "word".
    The entire list of all peak words forms a spectrum document. 

    """
    def __init__(self, spectrum, n_decimals: int = 1):
        """

        Parameters
        ----------
        spectrum: SpectrumType
            Input spectrum.
        n_decimals
            Peak positions are converted to strings with n_decimal decimals.
            The default is 2, which would convert a peak at 100.387 into the
            word "peak@100.39".
        """
        self.n_decimals = n_decimals
        self.weights = None
        super().__init__(obj=spectrum)
        self._add_weights()

    def _make_words(self):
        """Create word from peaks ."""
        format_string = "{}@{:." + "{}".format(self.n_decimals) + "f}"
        peak_words = [format_string.format("peak", mz) for mz in self._obj.peaks.mz]

        self.words = peak_words
        return self
    
    def _add_weights(self):
        """Add peaks (and loss) intensities as weights."""
        assert self._obj.peaks.intensities.max() <= 1, "peak intensities not normalized"

        peak_intensities = self._obj.peaks.intensities.tolist()
        #peak_mz = self._obj.peaks.mz.tolist()
        #w = np.arange(len(peak_mz )) + 1

        self.weights = peak_intensities
        return self
     
    def get(self, key: str, default=None):
        """Retrieve value from Spectrum metadata dict.

        """
        assert not hasattr(self, key), "Key cannot be attribute of SpectrumDocument class"
        return self._obj.get(key, default)

    @property
    def metadata(self):
        """Return metadata of original spectrum."""
        return self._obj.metadata
    @property
    def peaks(self) -> Spikes:
        """Return peaks of original spectrum."""
        return self._obj.peaks



def calc_vector(model: BaseTopicModel, document: Document,
                intensity_weighting_power: Union[float, int] = 0,
                allowed_missing_percentage: Union[float, int] = 0) -> np.ndarray:
    """
    model
        Pretrained word2vec model to convert words into vectors.
    document
        Document containing document.words and document.weights.
    intensity_weighting_power
        Specify to what power weights should be raised. The default is 0, which
        means that no weighing will be done.
    allowed_missing_percentage:
        Set the maximum allowed percentage of the document that may be missing
        from the input model. This is measured as percentage of the weighted, missing
        words compared to all word vectors of the document. Default is 0, which
        means no missing words are allowed.

    Returns
    -------
    vector
        Vector representing the input document in latent space. Will return None
        if the missing percentage of the document in the model is > allowed_missing_percentage.
    """
    assert max(document.weights) <= 1.0, "Weights are not normalized to unity as expected."
    assert 0 <= allowed_missing_percentage <= 100.0, "allowed_missing_percentage must be within [0,100]"

    def _check_model_coverage():
        """Return True if model covers enough of the document words."""
        if len(idx_not_in_model) > 0:
            weights_missing = np.array([document.weights[i] for i in idx_not_in_model])
            weights_missing_raised = np.power(weights_missing, intensity_weighting_power)
            missing_percentage = 100 * weights_missing_raised.sum() / (weights_raised.sum()
                                                                       + weights_missing_raised.sum())
            print("Found {} word(s) missing in the model.".format(len(idx_not_in_model)),
                  "Weighted missing percentage not covered by the given model is {:.2f}%.".format(missing_percentage))

            message = ("Missing percentage is larger than set maximum.",
                       "Consider retraining the used model or increasing the allowed percentage.")
            assert missing_percentage <= allowed_missing_percentage, message

    idx_not_in_model = [i for i, x in enumerate(document.words) if x not in model.wv.key_to_index]

    words_in_model = [x for i, x in enumerate(document.words) if i not in idx_not_in_model]
    weights_in_model = np.asarray([x for i, x in enumerate(document.weights)
                                      if i not in idx_not_in_model]).reshape(len(words_in_model), 1)

    word_vectors = model.wv[words_in_model]
    weights_raised = np.power(weights_in_model, intensity_weighting_power)

    _check_model_coverage()

    weights_raised_tiled = np.tile(weights_raised, (1, model.wv.vector_size))
    vector = np.sum(word_vectors * weights_raised_tiled, 0)
    return vector


class  spec_to_wordvector():
    """
    Adapted from: https://github.com/iomega/spec2vec
    Calculate spec2vec similarity scores between a reference and a query.

    Using a trained model, spectrum documents will be converted into spectrum
    vectors. The spec2vec similarity is then the cosine similarity score between
    two spectrum vectors.

   """
    
    def __init__(self, model: Word2Vec, intensity_weighting_power: Union[float, int] = 0,
                 allowed_missing_percentage: Union[float, int] = 0, progress_bar: bool = False):
        """

        Parameters
        ----------
        model:
            Expected input is a gensim word2vec model that has been trained on
            the desired set of spectrum documents.
        intensity_weighting_power:
            Spectrum vectors are a weighted sum of the word vectors. The given
            word intensities will be raised to the given power.
            The default is 0, which means that no weighing will be done.
        allowed_missing_percentage:
            Set the maximum allowed percentage of the document that may be missing
            from the input model. This is measured as percentage of the weighted, missing
            words compared to all word vectors of the document. Default is 0, which
            means no missing words are allowed.
        progress_bar:
            Set to True to monitor the embedding creating with a progress bar.
            Default is False.
        """
        self.model = model
        self.n_decimals = self._get_word_decimals(self.model)
        self.intensity_weighting_power = intensity_weighting_power
        self.allowed_missing_percentage = allowed_missing_percentage
        self.vector_size = model.wv.vector_size
        self.disable_progress_bar = not progress_bar

    @staticmethod
    def _get_word_decimals(model):
        """Read the decimal rounding that was used to train the model"""
        word_regex = r"[a-z]{4}@[0-9]{1,5}."
        example_word = next(iter(model.wv.key_to_index))

        return len(re.split(word_regex, example_word)[-1])

    def _calculate_embedding(self, spectrum_in: Union[SpectrumDocument, Spectrum]):
        """Generate Spec2Vec embedding vectors from input spectrum (or SpectrumDocument)"""
        assert spectrum_in.n_decimals == self.n_decimals, \
                "Decimal rounding of input data does not agree with model vocabulary."

        return calc_vector(self.model,
                           spectrum_in,
                           self.intensity_weighting_power,
                           self.allowed_missing_percentage)

def generate_word_vectors_from_sample(mzml_file_path,level):
    """
    Generate word vectors from spectra in an mzML file using a pre-trained Word2Vec model.

    Parameters:
    - mzml_file_path (str): Path to the mzML file containing mass spectrometry data.
    - model_file_path (str): Path to the pre-trained Word2Vec model file.

    Returns:
    - word_vec (csr_matrix): Sparse matrix containing the word vectors generated from spectra.
    """

    # Load spectra from mzML file
    spectra = list(load_from_mzml(mzml_file_path, level))

    # Preprocess spectra
    spectra = [default_filters(s) for s in spectra]
    spectra = [normalize_intensities(s) for s in spectra]

    # Load pre-trained Word2Vec model
    src_path=pkg_resources.get_distribution('MutAIverse').location
    model = gensim.models.Word2Vec.load(src_path+'/MutAIverse/references_word2vec.model')

    # Initialize spec_to_wordvector object
    spectovec = spec_to_wordvector(model=model, intensity_weighting_power=0.5)

    word2vectors = []
    for spectrum in tqdm(spectra):
        spectrum_doc = SpectrumDocument(spectrum, n_decimals=0)
        vector = spectovec._calculate_embedding(spectrum_doc)
        word2vectors.append(vector)

    # Convert word vectors to sparse matrix
    word_vec = csr_matrix(np.array(word2vectors))

    return word_vec


def fast_map(mzml_file_path,level=2, k=1, ef_query=300, Energy=0):
    """
    Search the HNSW index with given query data and return results as a DataFrame.

    Parameters:
    - index_file (str): Path to the saved HNSW index file.
    - query_data (numpy.ndarray): Query data in dense matrix format.
    - k (int): Number of nearest neighbors to search for.
    - ef_query (int): Parameter controlling the number of elements to visit during a query.
    - db_file (str): Path to the SQLite database file.
    - table_name (str): Name of the table containing chemical data.
    - plot_save_path (str): Path to save the plot.

    Returns:
    - pandas.DataFrame: DataFrame containing search results with columns ['Query_Index', 'Nearest_Neighbor_Index', 'Similarity', 'SMILES', 'COMPID'].
    - matplotlib.pyplot.figure: The plot object.
    """
    
    query_data=generate_word_vectors_from_sample(mzml_file_path,level)
    xq= query_data.todense().astype('float32')
    xq_len = np.linalg.norm(xq, axis=1, keepdims=True)
    xq = xq/xq_len
    # Load the index
    p = hnswlib.Index(space='cosine', dim=query_data.shape[1])
    with resources.open_binary('MutAIverse','mtverse_E'+str(Energy)+'.bin') as fp:
        bin_obj = fp.read()
    

    src_path=pkg_resources.get_distribution('MutAIverse').location
    p.load_index(src_path+'/MutAIverse/mtverse_E'+str(Energy)+'.bin', max_elements=query_data.shape[0])

    # Set query parameter
    p.set_ef(ef_query)

    # Perform the search
    I, D = p.knn_query(xq, k)

    # Convert distances to similarities
    similarities = 1 - D

    # Fetch additional information from the database
    with resources.open_binary('MutAIverse','mtverse_E'+str(Energy)+'.db') as fp:
        db_obj = fp.read()
    src_path=pkg_resources.get_distribution('MutAIverse').location
    gradedb = sqlite3.connect(src_path+'/MutAIverse/mtverse_E'+str(Energy)+'.db')
    cursor = gradedb.cursor()
    db_file = 'mtverse_E'+str(Energy)+'.db'
    cursor.execute(f"SELECT COMPID, SMILES, MZS,INTENSITYS from {os.path.splitext(os.path.basename(db_file))[0]}")
    all_smiles = []
    compid = []
    for row in tqdm(cursor):
        all_smiles.append(row[1])
        compid.append(row[0])
    matching_index = list(chain.from_iterable(I))
    matching_smiles = [all_smiles[i] for i in matching_index]
    matching_id = [compid[i] for i in matching_index]

    # Create DataFrame
    result_df = pd.DataFrame({
        'Query_Index': np.arange((query_data.shape[0])),
        'Nearest_Neighbor_Index': I.flatten(),
        'Similarity': similarities.flatten(),
        'SMILES': matching_smiles,
        'COMPID': matching_id
    })
    PandasTools.RenderImagesInAllDataFrames(images=True)
    result_df['STRUCTURE']= result_df.apply(slow_high, axis=1,ref_id_column='COMPID', smile_column='SMILES')
    # Plot density plot for similarities
    plt.figure(figsize=(8, 6))
    sns.histplot(result_df['Similarity'], kde=True, stat='density')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Density Plot of Similarities')
    plt.show()

    # Save the plot if save path is provided
    

    return result_df


def unzip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)



def load_library():
        src_path=pkg_resources.get_distribution('MutAIverse').location
        os.system('wget -q https://zenodo.org/records/13867395/files/lib.zip')
        unzip_file('lib.zip',src_path+'/MutAIverse/')
        os.remove('lib.zip')
        print('Library setup success!')
