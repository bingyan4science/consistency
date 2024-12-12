import pandas as pd
import sys
import warnings
import contextlib
warnings.filterwarnings("ignore")
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

smiles_acc_list = []
iupac_acc_list = []
smiles_smiles_true_list = []
smiles_iupac_true_list = []
iupac_smiles_true_list = []
iupac_iupac_true_list = []
smiles_consist_list = []
iupac_consist_list= []
false_smiles_consist_list = []
false_smiles_common_list = []
false_iupac_consist_list = []
false_iupac_common_list = []
smiles_common_true_list = []
iupac_common_true_list = []

for epoch in range(1):
    input_file_smiles = f'train_models_kl/smiles/valid_results_splitsmiles.csv'
    df_in_smiles = pd.read_csv(input_file_smiles)
    df_in_smiles_false = df_in_smiles[df_in_smiles['Predicted_0']!=df_in_smiles['Target']]
    df_in_smiles_true = df_in_smiles[df_in_smiles['Predicted_0']==df_in_smiles['Target']]
    smiles_smiles_true_list.append(df_in_smiles_true.shape[0])
    input_file_smiles2 = f'train_models_kl/iupac/valid_results_splitsmiles.csv'
    df_in_smiles2 = pd.read_csv(input_file_smiles2)
    df_in_smiles2_false = df_in_smiles2[df_in_smiles2['Predicted_0']!=df_in_smiles2['Target']]
    df_in_smiles2_true = df_in_smiles2[df_in_smiles2['Predicted_0']==df_in_smiles2['Target']]
    iupac_smiles_true_list.append(df_in_smiles2_true.shape[0])
    df_in_smiles_common = pd.merge(df_in_smiles_false, df_in_smiles2_false, how='inner', on=['Target'])
    consist_smiles = df_in_smiles_common[df_in_smiles_common['Predicted_0_x']==df_in_smiles_common['Predicted_0_y']].shape[0]
    df_in_smiles_common_true = pd.merge(df_in_smiles_true, df_in_smiles2_true, how='inner', on=['Target'])
    smiles_common_true = df_in_smiles_common_true.shape[0]
    smiles_common_true_list.append(smiles_common_true)
    acc_smiles = df_in_smiles[df_in_smiles['Predicted_0']==df_in_smiles2['Predicted_0']].shape[0] / len(df_in_smiles)
    input_file_iupac = f'train_models_kl/smiles/valid_results_splitiupac.csv'
    df_in_iupac = pd.read_csv(input_file_iupac)
    df_in_iupac_false = df_in_iupac[df_in_iupac['Predicted_0']!=df_in_iupac['Target']]
    df_in_iupac_true = df_in_iupac[df_in_iupac['Predicted_0']==df_in_iupac['Target']]
    smiles_iupac_true_list.append(df_in_iupac_true.shape[0])
    input_file_iupac2 = f'train_models_kl/iupac/valid_results_splitiupac.csv'
    df_in_iupac2 = pd.read_csv(input_file_iupac2)
    df_in_iupac2_false = df_in_iupac2[df_in_iupac2['Predicted_0']!=df_in_iupac2['Target']]
    df_in_iupac2_true = df_in_iupac2[df_in_iupac2['Predicted_0']==df_in_iupac2['Target']]
    iupac_iupac_true_list.append(df_in_iupac2_true.shape[0])
    df_in_iupac_common = pd.merge(df_in_iupac_false, df_in_iupac2_false, how='inner', on=['Target'])
    consist_iupac = df_in_iupac_common[df_in_iupac_common['Predicted_0_x']==df_in_iupac_common['Predicted_0_y']].shape[0]
    df_in_iupac_common_true = pd.merge(df_in_iupac_true, df_in_iupac2_true, how='inner', on=['Target'])
    iupac_common_true = df_in_iupac_common_true.shape[0]
    iupac_common_true_list.append(iupac_common_true)
    acc_iupac = df_in_iupac[df_in_iupac['Predicted_0']==df_in_iupac2['Predicted_0']].shape[0] / len(df_in_iupac)
    
    smiles_consist_list.append(acc_smiles)
    iupac_consist_list.append(acc_iupac)
    false_smiles_consist_list.append(consist_smiles)
    false_smiles_common_list.append(len(df_in_smiles_common))
    false_iupac_consist_list.append(consist_iupac)
    false_iupac_common_list.append(len(df_in_iupac_common))

    df_out_smiles = pd.DataFrame(columns = ['target', 'prediction'])
    df_out_iupac = pd.DataFrame(columns = ['target', 'prediction'])
    
    df_out_smiles['target'] = df_in_smiles['Target'].apply(lambda x: x.strip())
    df_out_smiles['prediction'] = df_in_smiles['Predicted_0'].apply(lambda x: str(x).strip())
    
    df_acc_smiles = df_out_smiles[df_out_smiles['target']==df_out_smiles['prediction']]
    acc_smiles = df_acc_smiles[df_acc_smiles['target']!=''].shape[0]
   
    df_out_iupac['target'] = df_in_iupac['Target'].apply(lambda x: x.strip())
    df_out_iupac['prediction'] = df_in_iupac['Predicted_0'].apply(lambda x: str(x).strip())
    
    df_acc_iupac = df_out_iupac[df_out_iupac['target']==df_out_iupac['prediction']]
    acc_iupac = df_acc_iupac[df_acc_iupac['target']!=''].shape[0]
    
    total = len(df_out_smiles)
    true_common = 0
    for i in range(len(df_out_smiles)):
        if df_out_smiles.loc[i, 'prediction'] == df_out_smiles.loc[i, 'target']:
            if df_out_iupac.loc[i, 'prediction'] == df_out_iupac.loc[i, 'target']:
                true_common += 1

    false_common = 0
    for i in range(len(df_out_smiles)):
        if df_out_smiles.loc[i, 'prediction'] != df_out_smiles.loc[i, 'target']:
            if df_out_iupac.loc[i, 'prediction'] != df_out_iupac.loc[i, 'target']:
                false_common += 1
                

    smiles_acc_list.append(acc_smiles/total)
    iupac_acc_list.append(acc_iupac/total)
    
print('consistency smiles:')
for item in smiles_consist_list:
    print(item)

print('consistency iupac:')
for item in iupac_consist_list:
    print(item)

print('false but consist smiles:')
for item in false_smiles_consist_list:
    print(item)
print('common false smiles:')
for item in false_smiles_common_list:
    print(item)

print('false but consist iupac:')
for item in false_iupac_consist_list:
    print(item)
print('common false iupac:')
for item in false_iupac_common_list:
    print(item)

print('common true smiles:')
for item in smiles_common_true_list:
    print(item)

print('common true iupac:')
for item in iupac_common_true_list:
    print(item)

print('true smiles input smiles output')
for item in smiles_smiles_true_list:
    print(item)

print('true smiles input iupac output')
for item in smiles_iupac_true_list:
    print(item)

print('true iupac input smiles output')
for item in iupac_smiles_true_list:
    print(item)

print('true iupac input iupac output')
for item in iupac_iupac_true_list:
    print(item)
