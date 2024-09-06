from rdkit import Chem as Chem
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
import pandas as pd
import joblib
from rdkit.Chem import rdFMCS
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
import io
from importlib import resources

def _remove_atoms_by_nr(emol: Chem.rdchem.EditableMol, list_atomnrs):
        list_atomnrs_sorted = sorted(list_atomnrs, reverse=True)
        list_frags = []
        for atomnr in list_atomnrs_sorted:
            emol.RemoveAtom(atomnr)
        return(emol.GetMol())


def remove_substr_matches_return_frag(mol_smiles:str,
                                      patt:str,
                                      pattAsSMARTS:bool=False,
                                     ):
    mol = Chem.MolFromSmiles(mol_smiles)
    if (pattAsSMARTS):
        patt_mol = Chem.MolFromSmarts(patt)
    else:
        patt_mol = Chem.MolFromSmiles(patt)
    matches = mol.GetSubstructMatches(patt_mol) #tuple of tuple of atom nrs ((1,2,3), (4,5,6))
    matches = [list(m) for m in matches] #convert to list of lists
    list_frags = []
    for m in matches:
        emol = Chem.EditableMol(mol)
        frag = _remove_atoms_by_nr(emol, m)
        list_frags.append(frag)
    print(f"{len(list_frags)} substructure match(es) found.")
    if len(list_frags) == 0:
        print("No matches found, returning input molecule")
        return []
    return list_frags  

def calculate_tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None
    return DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol1, 2), AllChem.GetMorganFingerprint(mol2, 2))

def atom_count(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return molecule.GetNumAtoms()

def extract_x(query=''):
    mol1 = Chem.MolFromSmiles(query)
    if mol1 is None:
        return None
    atgcu={'C1=NC2=NC=NC(=C2N1)N' : 'A',
    'CC1=CNC(=O)NC1=O' : 'T',
    'C1=NC2=C(N1)C(=O)NC(=N2)N' : 'G',
    'C1=C(NC(=O)N=C1)N' : 'C',
    'C1=CNC(=O)NC1=O' : 'U'}
    similarity=0
    adtX=''
    cor=''
    for i in atgcu.keys():
        if atom_count(i) > atom_count(query):
            continue
        if calculate_tanimoto_similarity(query,i) > similarity:
            similarity= calculate_tanimoto_similarity(query,i)
            cor=atgcu[i]
            adtX=remove_substr_matches_return_frag(query,i)
    frags=[]
    if len(adtX)==0:
        raise TypeError("Warning: The query adduct is missing nucleotide bases (dA, dT, dC, dG)")
    else:
        for i in adtX:
            sm=Chem.MolToSmiles(i)
            if '.' in sm:
                for j in sm.split('.'):
                    if calculate_tanimoto_similarity(j,'OCC1OCCC1O')==1:
                        continue
                    else:
                        frags.append(j)
            else:
                if calculate_tanimoto_similarity(sm,'OCC1OCCC1O')==1:
                    continue
                else:
                    frags.append(sm)
    return frags,cor

def similarity_search(dset,external_fp,k):
    with resources.open_binary('MutAIverse',dset+'_123.pkl') as fp:
        obj = fp.read()
    dataset_mols = joblib.load(io.BytesIO(obj))
    with resources.open_binary('MutAIverse',dset+'_mol_ids.pkl') as fp:
        obj = fp.read()
    iDs=joblib.load(io.BytesIO(obj))
    for i in range(len(iDs)):
        dataset_mols[i].SetProp("_Name", iDs[i])
    with resources.open_binary('MutAIverse',dset+'_123fp.pkl') as fp:
        obj = fp.read()
    dataset_fps = joblib.load(io.BytesIO(obj))
    similarities = BulkTanimotoSimilarity(external_fp, dataset_fps)
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    nearest_neighbors_indices = sorted_indices[:k]
    nearest_neighbors_similarities = [similarities[i] for i in nearest_neighbors_indices]
    nearest_neighbor_mols = [dataset_mols[i] for i in nearest_neighbors_indices]
    with resources.open_binary('MutAIverse',dset+'_ids-sm_it123.pkl') as fp:
        obj = fp.read()
    idIndex=joblib.load(io.BytesIO(obj))
    KNN_hits=pd.DataFrame()
    idss=[]
    simils=[]
    sms=[]
    for mol, similarity in zip(nearest_neighbor_mols, nearest_neighbors_similarities):
        idss.append(mol.GetProp("_Name"))
        simils.append(similarity)
        sms.append(idIndex[mol.GetProp("_Name")])
    KNN_hits['ID']=idss
    KNN_hits['SMILES']=sms
    KNN_hits['Similarity']=simils
    return KNN_hits

def KNN_Hits(external_mol,k):
    external_fp = AllChem.GetMorganFingerprintAsBitVect(external_mol, radius=3, nBits=2048)
    P1_Hits=similarity_search('P1',external_fp,k)
    P2_Hits=similarity_search('P2',external_fp,k)
    return P1_Hits,P2_Hits
    
def MCS_Hits(df,query_mol,k):
    dataset_mols=[]
    sms=df['SMILES'].tolist()
    iDs=df['ID'].tolist()
    for i in range(len(sms)):
        mol=Chem.MolFromSmiles(sms[i])
        mol.SetProp("_Name", iDs[i])
        dataset_mols.append(mol)
    mcs_values = []
    for mol in dataset_mols:
        mcs_result = rdFMCS.FindMCS([query_mol, mol])
        mcs_values.append((mol.GetProp("_Name"), mcs_result.numAtoms))
    sorted_mcs_values = sorted(mcs_values, key=lambda x: x[1], reverse=True)
    mcs_values_only=[]
    for j in range(len(sorted_mcs_values)):
        mcs_values_only.append(sorted_mcs_values[j][1])
    resIDs=[]
    if mcs_values_only.count(sorted_mcs_values[0][1]) > k:
        # Print all SMILES with the highest MCS value
        #print("All SMILES with the highest MCS value:")
        for smiles, mcs_value in sorted_mcs_values:
            if mcs_value == sorted_mcs_values[0][1]:
                resIDs.append(smiles)
            else:
                break
        #print(mcs_values_only.count(sorted_mcs_values[0][1]),sorted_mcs_values[0][1])
        #print(resIDs)
        return resIDs
    else:
        # Print the top 5 SMILES
        #print("Top "+str(k)+" SMILES with highest MCS value to the query:")
        for SMILES, mcs_value in sorted_mcs_values[:k]:
            resIDs.append(SMILES)
        #print(sorted_mcs_values[0][1])
        #print(resIDs)
        return resIDs
def compare_smiles(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return False
    mol1_canonical = Chem.MolToSmiles(mol1)
    mol2_canonical = Chem.MolToSmiles(mol2)
    return mol1_canonical == mol2_canonical

def uni_smiles(my_list):
    unique_list = []
    for i in range(len(my_list)):
        is_unique = True
        for j in range(i + 1, len(my_list)):
            if compare_smiles(my_list[i], my_list[j]):
                is_unique = False
                break
        if is_unique:
            unique_list.append(my_list[i])
    return unique_list
    
def flatten_list(nested_list):
    flat_list = []
    for i in nested_list:
        if isinstance(i, list):
            flat_list.extend(flatten_list(i))
        else:
            flat_list.append(i)
    return flat_list
    
def IT1s(ID,phs):
    with resources.open_binary('MutAIverse',phs+'_indexTrack_it123.pkl') as fp:
        obj = fp.read()
    addt=joblib.load(io.BytesIO(obj))
    sms=flatten_list(addt[ID])
    Gsms=[]
    for i in range(len(sms)):
        if sms[i] == 1:
            if sms[i+1] not in Gsms:
                Gsms.append(sms[i+1])
    if len(Gsms)==1:
        return Gsms
    else:
        return uni_smiles(Gsms)
    
def IT2s(ID,phs):
    with resources.open_binary('MutAIverse',phs+'_indexTrack_it123.pkl') as fp:
        obj = fp.read()
    addt=joblib.load(io.BytesIO(obj))
    sms=flatten_list(addt[ID])
    Gsm1=[]
    Gsm2=[]
    for i in range(len(sms)):
        if sms[i] == 1:
            if sms[i+1] not in Gsm1:
                Gsm1.append(sms[i+1])
        if sms[i] == 2:
            if sms[i+2] not in Gsm2:
                Gsm2.append(sms[i+2])
    if len(Gsm1)>0:
        if len(Gsm1)==1:
            return Gsm1,'r1'
        else:
            return uni_smiles(Gsm1),'r1'
    else:
        if len(Gsm2)==1:
            return Gsm2,'r2'
        else:
            return uni_smiles(Gsm2),'r2'
            
def IT3s(ID,phs):
    with resources.open_binary('MutAIverse',phs+'_indexTrack_it123.pkl') as fp:
        obj = fp.read()
    addt=joblib.load(io.BytesIO(obj))
    sms=flatten_list(addt[ID])
    r1s=[]
    r2s=[]
    for i in sms:
        if str(i).startswith('IT2'):
            l,r=IT2s(i,phs)
            if r=='r1':
                r1s.extend(l)
            else:
                r2s.extend(l)
    if len(r1s) > 0:
        return uni_smiles(r1s),'r1'
    else:
        return uni_smiles(r2s),'r2'

def MTVers_parser(id_list,phs):
    with resources.open_binary('MutAIverse',phs+'_ids-sm_it123.pkl') as fp:
        obj = fp.read()
    id_sm=joblib.load(io.BytesIO(obj))
    src_m=[]
    src_gn=[]
    trns=[]
    for ids in id_list:
        if ids.startswith('GT'):
            trns.append(0)
            src_m.append('NA')
            src_gn.append(id_sm[ids])
        if ids.startswith('IT1'):
            gn=IT1s(ids,phs)
            for i in gn:
                trns.append(1)
                src_m.append(id_sm[ids])
                src_gn.append(i)
        if ids.startswith('IT2'):
            gn,cnt=IT2s(ids,phs)
            for i in gn:
                src_m.append(id_sm[ids])
                src_gn.append(i)
                if cnt == 'r1':
                    trns.append(1)
                else:
                    trns.append(2)
        if ids.startswith('IT3'):
            gn,cnt=IT3s(ids,phs)
            for i in gn:
                src_m.append(id_sm[ids])
                src_gn.append(i)
                if cnt == 'r1':
                    trns.append(2)
                else:
                    trns.append(3)
    GenTox=pd.DataFrame()
    GenTox['Metabolites']=src_m
    GenTox['N-Transformation']=trns
    GenTox['Genotoxin']=src_gn
    return GenTox
    
def add_SM_clm(df,Adon_mol):
    SMs=[]
    CalSM=[]
    for sm in df['Metabolites'].tolist():
        if sm in CalSM:
            continue
        else:
            CalSM.append(sm)
        if sm == 'NA':
            GNsms=df[df['Metabolites']==sm]
            GSMs=[]
            penal=[]
            for gsm in GNsms['Genotoxin'].tolist():
                smmol=Chem.MolFromSmiles(gsm)
                mols=[Adon_mol,smmol]
                res=rdFMCS.FindMCS(mols)
                GSMs.append(res.numAtoms/Adon_mol.GetNumAtoms())
                penal.append((smmol.GetNumAtoms()-res.numAtoms)/smmol.GetNumAtoms())
            GNsms['x']=GSMs
            GNsms['Probability'] = (GNsms['x'] - penal)*100
            SMs.append(GNsms)
        else:
            subdf=df[df['Metabolites']==sm]
            smmol=Chem.MolFromSmiles(sm)
            mols=[Adon_mol,smmol]
            res=rdFMCS.FindMCS(mols)
            x=res.numAtoms/Adon_mol.GetNumAtoms()
            subdf['x']=[x]*len(subdf)
            subdf['Probability'] = (subdf['x'] - ((smmol.GetNumAtoms()-res.numAtoms)/smmol.GetNumAtoms()))*100
            SMs.append(subdf)
    dfr=pd.concat(SMs)
    #dfr['Probability'] = dfr['x'] * (100/dfr['Transformations'])
    return dfr.sort_values(by='Probability',ascending=False).drop('x', axis=1)

def plot_trace(file,df='',cutoff=80):
    if len(df)==0:
        df=pd.read_csv(file)
    subdf=df[df['Probability']>cutoff]
    smiles_list=[]
    title=[]
    for i in range(len(subdf)):
        smiles_list.append(subdf['Query'].tolist()[i])
        title.append('Adduct')
        smiles_list.append(subdf['Fragment'].tolist()[i])
        title.append('Fragment')
        if subdf['Metabolites'].tolist()[i] == 'NA':
            smiles_list.append('')
        else:
            smiles_list.append(subdf['Metabolites'].tolist()[i])

        title.append('Metabolite '+str(i+1))
        title.append('Genotoxin '+str(i+1))
        smiles_list.append(subdf['Genotoxin'].tolist()[i])
    compounds_per_row = 4
    num_rows = len(subdf)
    num_compounds = len(smiles_list)
    num_cols = compounds_per_row
    num_full_rows = num_compounds // compounds_per_row
    if num_compounds % compounds_per_row != 0:
        num_full_rows += 1
    fig, axes = plt.subplots(num_rows, compounds_per_row, figsize=(4 * compounds_per_row, 4 * num_rows))
    for i, ax in enumerate(axes.flat):
        if i < num_compounds:
            try:
                mol = Chem.MolFromSmiles(smiles_list[i])
            except:
                mol = Chem.MolFromSmiles('')
            img = Draw.MolToImage(mol, size=(200, 200))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title[i])
    for j in range(num_compounds, num_rows * compounds_per_row):
        axes.flat[j].axis('off')
    plt.tight_layout()
    plt.show()

def backtrace(Adduct='',knn=20,tophits=5,plot=False,cutoff=80):
    if len(Adduct)==0:
        raise ValueError('No input provided!')
    else:
        Adon=extract_x(Adduct)
    Adon_mol = Chem.MolFromSmiles(Adon[0][0])
    if Adon_mol.GetNumAtoms() < 3 :
        raise ValueError('Warning: Query adduct has insufficient atoms for tracing!')
    p1,p2=KNN_Hits(Adon_mol,knn)
    p1_hits=MCS_Hits(p1,Adon_mol,tophits)
    p2_hits=MCS_Hits(p2,Adon_mol,tophits)
    p1_gentox=MTVers_parser(p1_hits,'P1')
    p2_gentox=MTVers_parser(p2_hits,'P2')
    rdf=pd.concat([p1_gentox,p2_gentox],axis=0)
    outd=add_SM_clm(rdf,Adon_mol)
    res=pd.DataFrame()
    #res.insert(loc=0, column='Fragment', value=[Adon[0][0]]*len(outd))
    #res.insert(loc=0, column='Query', value=[Adduct]*len(outd))
    res['Query']=[Adduct]*len(outd)
    res['Fragment']=[Adon[0][0]]*len(outd)
    for cm in ['Metabolites','N-Transformation','Genotoxin','Probability']:
        res[cm]=outd[cm].tolist()
    if plot:
        plot_trace('x',res,cutoff)
    return res.reset_index(drop=True)
