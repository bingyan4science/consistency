import pandas as pd
import sys
from rdkit import Chem
from py2opsin import py2opsin
import warnings
import contextlib
warnings.filterwarnings("ignore")
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

#iupac_canon = pd.read_csv("/scratch/by2192/implicit_chain_of_thought/data/llasmol_reaction/iupac_predictions_canon.csv")

epochs = []
smiles_acc_list = []
iupac_acc_list = []
smiles_canon_acc_list = []
iupac_canon_acc_list = []
smiles_consist_list = []
iupac_consist_list= []
false_smiles_consist_list = []
false_smiles_common_list = []
false_iupac_consist_list = []
false_iupac_common_list = []
smiles_common_true_list = []
iupac_common_true_list = []

#import ipdb; ipdb.set_trace()
for epoch in range(55):
    #print (f'Epoch {epoch}')
    epochs.append(epoch) 
    input_file_smiles = f'train_models/combined_gpt2/smiles_e60_onebatch_bsz16_lr5e-5/valid_results_epoch{epoch}_splitsmiles.csv'
    df_in_smiles = pd.read_csv(input_file_smiles)
    df_in_smiles_false = df_in_smiles[df_in_smiles['Predicted']!=df_in_smiles['Target']]
    df_in_smiles_true = df_in_smiles[df_in_smiles['Predicted']==df_in_smiles['Target']]
    input_file_smiles2 = f'train_models/combined_gpt2/iupac_e60_onebatch_bsz16_lr5e-5/valid_results_epoch{epoch}_splitsmiles.csv'
    df_in_smiles2 = pd.read_csv(input_file_smiles2)
    df_in_smiles2_false = df_in_smiles2[df_in_smiles2['Predicted']!=df_in_smiles2['Target']]
    df_in_smiles2_true = df_in_smiles2[df_in_smiles2['Predicted']==df_in_smiles2['Target']]
    df_in_smiles_common = pd.merge(df_in_smiles_false, df_in_smiles2_false, how='inner', on=['Target'])
    consist_smiles = df_in_smiles_common[df_in_smiles_common['Predicted_x']==df_in_smiles_common['Predicted_y']].shape[0]
    df_in_smiles_common_true = pd.merge(df_in_smiles_true, df_in_smiles2_true, how='inner', on=['Target'])
    smiles_common_true = df_in_smiles_common_true.shape[0]
    smiles_common_true_list.append(smiles_common_true)
    acc_smiles = df_in_smiles[df_in_smiles['Predicted']==df_in_smiles2['Predicted']].shape[0] / len(df_in_smiles)
    #print (f'acc_smiles: {acc_smiles}')
    #print(f'common false smiles prediction #: {len(df_in_smiles_common)} consist false: {consist_smiles}')


    input_file_iupac = f'train_models/combined_gpt2/iupac_e60_onebatch_bsz16_lr5e-5/valid_results_epoch{epoch}_splitiupac.csv'
    df_in_iupac = pd.read_csv(input_file_iupac)
    df_in_iupac_false = df_in_iupac[df_in_iupac['Predicted']!=df_in_iupac['Target']]
    df_in_iupac_true = df_in_iupac[df_in_iupac['Predicted']==df_in_iupac['Target']]
    input_file_iupac2 = f'train_models/combined_gpt2/smiles_e60_onebatch_bsz16_lr5e-5/valid_results_epoch{epoch}_splitiupac.csv'
    df_in_iupac2 = pd.read_csv(input_file_iupac2)
    df_in_iupac2_false = df_in_iupac2[df_in_iupac2['Predicted']!=df_in_iupac2['Target']]
    df_in_iupac2_true = df_in_iupac2[df_in_iupac2['Predicted']==df_in_iupac2['Target']]
    df_in_iupac_common = pd.merge(df_in_iupac_false, df_in_iupac2_false, how='inner', on=['Target'])
    consist_iupac = df_in_iupac_common[df_in_iupac_common['Predicted_x']==df_in_iupac_common['Predicted_y']].shape[0]
    df_in_iupac_common_true = pd.merge(df_in_iupac_true, df_in_iupac2_true, how='inner', on=['Target'])
    iupac_common_true = df_in_iupac_common_true.shape[0]
    iupac_common_true_list.append(iupac_common_true)
    acc_iupac = df_in_iupac[df_in_iupac['Predicted']==df_in_iupac2['Predicted']].shape[0] / len(df_in_iupac)
    #print (f'acc_iupac: {acc_iupac}')
    #print(f'common false iupac prediction #: {len(df_in_iupac_common)} consist false: {consist_iupac}')

    smiles_consist_list.append(acc_smiles)
    iupac_consist_list.append(acc_iupac)
    false_smiles_consist_list.append(consist_smiles)
    false_smiles_common_list.append(len(df_in_smiles_common))
    false_iupac_consist_list.append(consist_iupac)
    false_iupac_common_list.append(len(df_in_iupac_common))

    df_out_smiles = pd.DataFrame(columns = ['target', 'prediction', 'target_canon', 'pred_canon'])
    df_out_iupac = pd.DataFrame(columns = ['target', 'prediction', 'target_canon', 'pred_canon'])
    
    def canon(mol):
        if mol == '':
            return ''
        try:
            with nostdout():
                mol_canon = Chem.CanonSmiles(mol)
        except:
            mol_canon = mol
        return mol_canon
    
    def iupac2smi(mol):
        try:
            mol_smiles = py2opsin(mol)
        except:
            mol_smiles = ''
            mol_canon = ''
        else:
            mol_canon = canon(mol_smiles)
        return mol_canon
    
    df_out_smiles['target'] = df_in_smiles['Target'].apply(lambda x: x.strip())
    df_out_smiles['prediction'] = df_in_smiles['Predicted'].apply(lambda x: x.strip())
    #df_out_smiles['target_canon'] = df_out_smiles['target'].apply(canon)
    #df_out_smiles['pred_canon'] = df_out_smiles['prediction'].apply(canon)
    
    df_acc_smiles = df_out_smiles[df_out_smiles['target']==df_out_smiles['prediction']]
    acc_smiles = df_acc_smiles[df_acc_smiles['target']!=''].shape[0]
    #df_acc_smiles_canon = df_out_smiles[df_out_smiles['target_canon']==df_out_smiles['pred_canon']]
    #acc_smiles_canon = df_acc_smiles_canon[df_acc_smiles_canon['target_canon'] != ''].shape[0]
    
    df_out_iupac['target'] = df_in_iupac['Target'].apply(lambda x: x.strip())
    df_out_iupac['prediction'] = df_in_iupac['Predicted'].apply(lambda x: x.strip())

    #l = df_out_iupac['prediction']
    #mol_smiles = py2opsin(l)
    #df_out_iupac['pred_canon'] = mol_smiles
    #df_out_iupac['pred_canon'] = df_out_iupac['pred_canon'].apply(canon)

    #l = df_out_iupac['target']
    #mol_smiles = py2opsin(l)
    #df_out_iupac['target_canon'] = mol_smiles
    #df_out_iupac['target_canon'] = df_out_iupac['target_canon'].apply(canon)
    
    df_acc_iupac = df_out_iupac[df_out_iupac['target']==df_out_iupac['prediction']]
    acc_iupac = df_acc_iupac[df_acc_iupac['target']!=''].shape[0]
    #df_acc_iupac_canon = df_out_iupac[df_out_iupac['target_canon']==df_out_iupac['pred_canon']]
    #acc_iupac_canon = df_acc_iupac_canon[df_acc_iupac_canon['target_canon'] != ''].shape[0]

    #import pdb; pdb.set_trace()
    
    #consist = df_out_smiles[df_out_smiles['pred_canon']==df_out_iupac['pred_canon']].shape[0]
    total = len(df_out_smiles)
    true_common = 0
    for i in range(len(df_out_smiles)):
        if df_out_smiles.loc[i, 'prediction'] == df_out_smiles.loc[i, 'target']:
            if df_out_iupac.loc[i, 'prediction'] == df_out_iupac.loc[i, 'target']:
                true_common += 1
    #print(true_common)

    false_common = 0
    for i in range(len(df_out_smiles)):
        if df_out_smiles.loc[i, 'prediction'] != df_out_smiles.loc[i, 'target']:
            if df_out_iupac.loc[i, 'prediction'] != df_out_iupac.loc[i, 'target']:
                false_common += 1
                
    #print(false_common)
    
    #df_out_smiles.to_csv(input_file_smiles[:-4] + "_canon.csv", index=False)
    #df_out_iupac.to_csv(input_file_iupac[:-4] + "_canon.csv", index=False)
    
    #print(f"accuracy_smiles is {acc_smiles/total} accuracy_iupac is {acc_iupac/total}")
    #print(f"accuracy_smiles_canon is {acc_smiles_canon/total} accuracy_iupac_canon is {acc_iupac_canon/total}")
    #print(f"consistent instances # is {consist} consistency is {consist/total}")

    smiles_acc_list.append(acc_smiles/total)
    iupac_acc_list.append(acc_iupac/total)
    #smiles_canon_acc_list.append(acc_smiles_canon/total)
    #iupac_canon_acc_list.append(acc_iupac_canon/total)

#import ipdb; ipdb.set_trace()
#df_summary = pd.DataFrame()
#df_summary['epoch'] = epochs
#df_summary['false_smiles_consist'] = false_smiles_consist_list
#df_summary['false_smiles_common'] = false_smiles_common_list
#df_summary['false_iupac_consist'] = false_iupac_consist_list
#df_summary['false_iupac_common'] = false_iupac_common_list
#df_summary['smiles_consist'] = smiles_consist_list
#df_summary['iupac_consist'] = iupac_consist_list
#df_summary['smiles_acc'] = smiles_acc_list
#df_summary['iupac_acc'] = iupac_acc_list
#df_summary['smiles_canon_acc'] = smiles_canon_acc_list
#df_summary['iupac_canon_acc'] = iupac_canon_acc_list
#df_summary.to_csv("/scratch/by2192/implicit_chain_of_thought/consist_summary.csv", index=False)

for item in smiles_common_true_list:
    print(item)
print("iupac common true")
for item in iupac_common_true_list:
    print(item)
