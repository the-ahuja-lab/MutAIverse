# MutAIverse
Facilitating the identification of DNA adducts from untargeted metabolomics mass spectrometry data along with predictive capabilities to determine potential source genotoxins responsible for the novel identified or pre-existing adduct formation.


The single strong dependency for this resource is **[RDKit](https://www.rdkit.org/)**, which can be installed in a local [Conda](https://conda.io/) environment.

**Other dependencies**
1. Match ms
2. Hnswlib
3. Gensim
4. pandas
5. numpy
6. matplotlib
7. tqdm


## Adduct Mapper module
MutAIverse provides two approaches for mapping query MS spectra against *in silico* MS MS spectral library of Experimentally validated adducts or Synthetic DNA adducts of MutAIverse.


### Brute force Approach 
Cosine Similarity-based mapping 

```Python
from MutAIverse import Mapper
Mapper.map('bonafide_adducts',sample_file_path='/path-to-mzML-file',MS_level=1,plot=True)

```

Additional arguments 

    Parameters:
    - library (str): bonafide_adducts/MutAIversee
    - sample_file_path (str): Path to the mzML file containing mass spectrometry data.
    - ms level (int): 1 (MS spectrum) or 2 (MS/MS spectrum)
    - plot (bool; default True): for visualizations
    return
    - Result CSV file with suffix _MutAIversee_results.csv or _bonafide_adducts_results.csv



### Quick Search Approach 
Approximate Nearest Neighbour-based mapping, which executes through 2 steps
1. Generation of spectral embeddings from query MS spectra
2. Mapping using the HNSW index of the spectral embeddings

```python
from MutAIverse import Mapper
Mapper.fast_map(mzml_file_path)

```

Additional arguments 

    Parameters:
    - mzml_file_path (str): Path to the mzML file containing mass spectrometry data.
    - level (int; default 2): 1 (MS spectrum) or 2 (MS/MS spectrum)
    - k (int; default 1): Number of nearest neighbors to search for.
    - ef_query (int; default 300): Parameter controlling the number of elements to visit during a query.
    - Energy (int; default 0): 
    
    Returns:
    - pandas.DataFrame: DataFrame containing search results with columns ['Query_Index', 'Nearest_Neighbor_Index', 'Cosine Similarity', 'SMILES', 'COMPID', 'Structures'].
    - visualizations(density plot and histograms)


## Adduct Linker module
MutAIverse is also capable of re-tracing a DNA adduct to its possible source Genotoxin.


### Fragment-based linking 
biotransformation backtracking based on abnormalities spliced from the base nucleotides

```python
from MutAIverse import Linker
query_smiles = 'OC[C@H]1O[C@H](CC1O)n1c[n+](c2c1nc(N)[nH]c2=O)C1OC2C(C1O)c1c(O2)cc(c2c1oc(=O)c1c2CCC1=O)OC' 
Linker.backtrace(Adduct = query_smiles)

```

Additional arguments 

    Parameters:
    - Adduct (str): Path to the mzML file containing mass spectrometry data.
    - knn (int; default 20): Number of nearest neighbors to narrow down the search space. 
    - tophit (int; default 5): Minimum number of Genotoxins to be linked.
    
    Returns:
    - pandas.DataFrame: DataFrame containing search results with columns ['Query', 'Fragment', 'Metabolites', 'N-Transformation', 'Genotoxin', 'Probability'].
    - visualizations(Traced smiles 2D structures in rows)


This module also has a sub-function dedicated only to visualize backtrace() output with a user-supplied probability threshold.
```python
import pandas as pd
from MutAIverse import Linker 
df = pd.read_csv('output DataFrame of Linker.backtrace() function')
plot_trace(file=df)
```

Additional arguments 

    Parameters:
    - file (pandas.DataFrame): output DataFrame of Linker.backtrace() function
    - cutoff (int; default 80): Minimum probability threshold 
    Returns:
    - visualizations(Traced smiles 2D structures in rows)
  

