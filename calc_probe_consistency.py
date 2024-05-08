import pandas as pd
import pubchempy
from chemicalconverters import NamesConverter
from rdkit import Chem

def get_canon_smiles(smiles):
    try:
        smiles_canon = Chem.CanonSmiles(smiles)
    except:
        print(f"Error: {smiles} cannot be canonized.")
        smiles_canon = ""
    return smiles_canon


def translate(smiles):
    #import ipdb; ipdb.set_trace()
    smiles_canon = get_canon_smiles(smiles)
    #translation_train_set = train_set.filter(lambda example: example["task"] == 'name_conversion-i2s' or example["task"] == 'name_conversion-s2i')
    
    try:
        compounds = pubchempy.get_compounds(smiles_canon, namespace='smiles')
    except:
        iupac = None
    else:
        if len(compounds) >= 1:
            iupac = compounds[0].iupac_name
        else:
            iupac = None
    
    if iupac == None:
        try:
            #import ipdb; ipdb.set_trace()
            iupac_try, validation = converter.smiles_to_iupac(smiles, validate=True)
        except:
            iupac = None
        else:
            if validation:
                iupac = iupac_try[0]
            else:
                iupac = iupac_try[0]
                print(f"validation is false. the predicted iupac is {iupac} for smiles {smiles_canon} and {smiles}")
    return iupac


df_summary = pd.DataFrame(columns = ['epochs', 'consist_fg', 'inconsist_fg', 'consist_fg_consist', 'consist_fg_inconsist'])

epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35, 40]

df_summary['epochs'] = epochs
consist_fg = []
inconsist_fg = []
consist_fg_consist = []
consist_fg_inconsist = []

for epoch in epochs:
    #import ipdb; ipdb.set_trace()
    print(f'epoch: {epoch}')
    fin_fg_smi = f"train_models/combined_gpt2/smiles_e60_onebatch_bsz16_lr5e-5/fg_e{epoch}/probe_test_layer5_results_epoch0.csv"
    df_fg_smi = pd.read_csv(fin_fg_smi)
    fin_fg_iup = f"train_models/combined_gpt2/iupac_e60_onebatch_bsz16_lr5e-5/fg_e{epoch}/probe_test_layer5_results_epoch0.csv"
    df_fg_iup = pd.read_csv(fin_fg_iup)
    fin_pp_smi_smi = f"train_models/combined_gpt2/smiles_e60_onebatch_bsz16_lr5e-5/valid_results_epoch{epoch}_splitsmiles.csv"
    df_pp_smi_smi = pd.read_csv(fin_pp_smi_smi)
    fin_pp_iup_smi = f"train_models/combined_gpt2/iupac_e60_onebatch_bsz16_lr5e-5/valid_results_epoch{epoch}_splitsmiles.csv"
    df_pp_iup_smi = pd.read_csv(fin_pp_iup_smi)
    fin_pp_smi_iup = f"train_models/combined_gpt2/smiles_e60_onebatch_bsz16_lr5e-5/valid_results_epoch{epoch}_splitiupac.csv"
    df_pp_smi_iup = pd.read_csv(fin_pp_smi_iup)
    fin_pp_iup_iup = f"train_models/combined_gpt2/iupac_e60_onebatch_bsz16_lr5e-5/valid_results_epoch{epoch}_splitiupac.csv"
    df_pp_iup_iup = pd.read_csv(fin_pp_iup_iup)

    df_out = pd.DataFrame(columns = ['target_smi_product', 'pred_smi_smi', 'pred_iup_smi', \
                                    'target_iup_product', 'pred_smi_iup', 'pred_iup_iup', \
                                    'pred_smi_smi_iup', 'pred_iup_smi_iup', \
                                    'target_fg', 'pred_fg_smi', 'pred_fg_iup'])
    
    df_out['target_fg'] = df_fg_smi['Target']
    df_out['pred_fg_smi'] = df_fg_smi['Predicted']
    df_out['pred_fg_iup'] = df_fg_iup['Predicted']
    df_out['target_smi_product'] = df_pp_smi_smi['Target']
    df_out['pred_smi_smi'] = df_pp_smi_smi['Predicted']
    df_out['pred_iup_smi'] = df_pp_iup_smi['Predicted']
    df_out['target_iup_product'] = df_pp_smi_iup['Target']
    df_out['pred_smi_iup'] = df_pp_smi_iup['Predicted']
    df_out['pred_iup_iup'] = df_pp_iup_iup['Predicted']
    
    df_out['pred_smi_smi_iup'] = df_out['pred_smi_smi'].apply(translate)
    df_out['pred_iup_smi_iup'] = df_out['pred_iup_smi'].apply(translate)
    fout = f"fg_pp_{epoch}_merged.csv"
    df_out.to_csv(fout, index=False)

    df_out['pred_smi_smi_iup'] = df_out['pred_smi_smi_iup'].apply(lambda x: str(x).replace(',', ''))
    df_out['pred_iup_smi_iup'] = df_out['pred_iup_smi_iup'].apply(lambda x: str(x).replace(',', ''))
    
    consist_fg_item = 0
    inconsist_fg_item = 0
    consist_fg_consist_item = 0
    consist_fg_inconsist_item = 0

    for index, row in df_out.iterrows():
        if row['pred_fg_smi']==row['pred_fg_iup']:
            consist_fg_item += 1
            if row['pred_smi_smi'] == row['pred_iup_smi']:
                consist_fg_consist_item += 0.25
            if row['pred_smi_smi_iup'] == row['pred_iup_iup']:
                consist_fg_consist_item += 0.25
            if row['pred_smi_iup'] == row['pred_iup_iup']:
                consist_fg_consist_item += 0.25
            if row['pred_smi_iup'] == row['pred_iup_smi_iup']:
                consist_fg_consist_item += 0.25
        else:
            inconsist_fg_item += 1
            if row['pred_smi_smi'] == row['pred_iup_smi']:
                consist_fg_inconsist_item += 0.25
            if row['pred_smi_smi_iup'] == row['pred_iup_iup']:
                consist_fg_inconsist_item += 0.25
            if row['pred_smi_iup'] == row['pred_iup_iup']:
                consist_fg_inconsist_item += 0.25
            if row['pred_smi_iup'] == row['pred_iup_smi_iup']:
                consist_fg_inconsist_item += 0.25
    
    consist_fg.append(consist_fg_item)
    inconsist_fg.append(inconsist_fg_item)
    consist_fg_consist.append(consist_fg_consist_item)
    consist_fg_inconsist.append(consist_fg_inconsist_item)

df_summary['consist_fg'] = consist_fg
df_summary['inconsist_fg'] = inconsist_fg
df_summary['consist_fg_consist'] = consist_fg_consist
df_summary['consist_fg_inconsist'] = consist_fg_inconsist

df_summary.to_csv("fg_pp_consistency_summary_partial_score.csv", index=False)
