import os

# Richard hall 2017
# IFG main code
# Guillaume Godin 2017
# refine output function
# astex_ifg: identify functional groups a la Ertl, J. Cheminform (2017) 9:36
from rdkit import Chem
from collections import namedtuple

def merge(mol, marked, aset):
    bset = set()
    for idx in aset:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            jdx = nbr.GetIdx()
            if jdx in marked:
                marked.remove(jdx)
                bset.add(jdx)
    if not bset:
        return
    merge(mol, marked, bset)
    aset.update(bset)

# atoms connected by non-aromatic double or triple bond to any heteroatom
# c=O should not match (see fig1, box 15).  I think using A instead of * should sort that out?
PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
# atoms in non aromatic carbon-carbon double or triple bonds
PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('C=,#C')
# acetal carbons, i.e. sp3 carbons connected to tow or more oxygens, nitrogens or sulfurs; these O, N or S atoms must have only single bonds
PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
# all atoms in oxirane, aziridine and thiirane rings
PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')

PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC)

def identify_functional_groups(mol):
    marked = set()
#mark all heteroatoms in a molecule, including halogens
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6,1): # would we ever have hydrogen?
            marked.add(atom.GetIdx())

#mark the four specific types of carbon atom
    for patt in PATT_TUPLE:
        for path in mol.GetSubstructMatches(patt):
            for atomindex in path:
                marked.add(atomindex)

#merge all connected marked atoms to a single FG
    groups = []
    while marked:
        grp = set([marked.pop()])
        merge(mol, marked, grp)
        groups.append(grp)

#extract also connected unmarked carbon atoms
    ifg = namedtuple('IFG', ['atomIds', 'atoms', 'type'])
    ifgs = []
    for g in groups:
        uca = set()
        for atomidx in g:
            for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
                if n.GetAtomicNum() == 6:
                    uca.add(n.GetIdx())
        ifgs.append(ifg(atomIds=tuple(list(g)), atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True), type=Chem.MolFragmentToSmiles(mol, g.union(uca),canonical=True)))
    return ifgs
from dataclasses import dataclass
import os
import copy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def extract_answer(text):
    split_pattern = '####'
    if split_pattern not in text:
        return text.strip().replace(',', '')
    else:
        _, ans = text.strip().split('####', 1)
        ans = '####' + ans
        ans = ans.strip().replace(',', '')
        return ans

def extract_cot(text):
    split_pattern = '####'
    if split_pattern not in text:
        return None
    else:
        cot, _ = text.strip().split('####', 1)
        cot = cot.strip()
        return cot

class CoTDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        print (f'Creating features from dataset file at {file_path}')
        #import ipdb; ipdb.set_trace()
        #bos_tok = tokenizer.bos_token
        eos_tok = tokenizer.eos_token

        with open(file_path, encoding="utf-8") as f:
            #lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
            #                                                                 and len(line.split('||')) ==2 )]
            lines = [line.split('||') for line in f.readlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]

        smiles_file_path = file_path.replace('llasmol_reaction_iupac_combined_80k_noinstruction', 'llasmol_reaction_smiles_combined_80k_noinstruction')
        with open(smiles_file_path, encoding="utf-8") as f:
            lines2 = [line.split('||') for line in f.readlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]

        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)
        src_lines_smiles, _ = list(zip(*lines2))
        src_lines_smiles = list(src_lines_smiles)

        #edited_sents_cot = []
        #edited_sents_only = []
        edited_sents_all = []
        #edited_sents_nocot = []
        #import pdb; pdb.set_trace()
        itos = []
        stoi = {}
        with open('function_groups.txt') as fin:
            for line in fin:
                itos.append(line.strip())
                stoi[line.strip()] = len(stoi)


        self.probe_labels_all = [] 
        for src, tgt, src_smiles in zip(src_lines, tgt_lines, src_lines_smiles):
            src = src.strip().split('.')
            src_smiles = src_smiles.strip().split('.')
            probe_labels = torch.zeros(len(src)-1, len(stoi)).long()
            #import pdb; pdb.set_trace()
            for ia, smiles in enumerate(src_smiles[:-1]):
                m = Chem.MolFromSmiles(smiles)                                                                                                                                                      
                fgs = identify_functional_groups(m)                                                                                                                                                 
                ts = [fg.type for fg in fgs]
                for t in ts:
                    if t in stoi:
                        probe_labels[ia, stoi[t]] = 1
            self.probe_labels_all.append(probe_labels)


            src = f' {eos_tok} '.join(src)
            ans = extract_answer(tgt)
            cot = extract_cot(tgt)
            if src is None:
                src = ''
            if ans is None:
                ans = ''
            if cot is None:
                cot = ''
            #sent = ' {} {} '.format(src, bos_tok) + cot + ' {}'.format(eos_tok)
            #edited_sents_cot.append(sent)
            #sent = ' {} {} '.format(src, bos_tok)
            #edited_sents_only.append(sent)

            sent = ' {} {} '.format(src, eos_tok) + cot + ' {} '.format(eos_tok) + ans + ' {}'.format(eos_tok)
            edited_sents_all.append(sent)
            #sent = ' {} {} '.format(src, bos_tok) + ans + ' {}'.format(eos_tok)
            #edited_sents_nocot.append(sent)
        #import ipdb; ipdb.set_trace()
        #batch_encoding_cot = tokenizer(edited_sents_cot, add_special_tokens=True, truncation=True, max_length=max_length)
        #batch_encoding_only = tokenizer(edited_sents_only, add_special_tokens=True, truncation=True, max_length=max_length)
        batch_encoding_all = tokenizer(edited_sents_all, add_special_tokens=True, truncation=True, max_length=max_length)
        #batch_encoding_nocot = tokenizer(edited_sents_nocot, add_special_tokens=True, truncation=True, max_length=max_length)
        #self.examples_cot = batch_encoding_cot["input_ids"]
        #self.examples_only = batch_encoding_only["input_ids"]
        self.examples_all = batch_encoding_all["input_ids"]
        #self.examples_nocot = batch_encoding_nocot["input_ids"]

        #self.labels_cot = copy.deepcopy(self.examples_cot)
        self.labels_all = copy.deepcopy(self.examples_all)
        #self.labels_cot_shift = copy.deepcopy(self.examples_cot)
        #self.labels_nocot = copy.deepcopy(self.examples_nocot)

        self.src_sent_cot = []
        self.tgt_sent_cot = []

        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        separator = tokenizer.eos_token_id #tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        #import pdb; pdb.set_trace()
        for i, elem in enumerate(self.labels_all):
            try:
                sep_idx = [i for i, n in enumerate(elem) if n == separator][-3] + 1
                #sep_idx = elem.index(separator) + 1
                #assert sep_idx == elem.index(separator) + 1
                #sep_idx = elem[::-1].index(separator) + 1
            except:
                #import ipdb; ipdb.set_trace()
                sep_idx = len(self.labels_all[i])
                self.labels_all[i][sep_idx-1] = separator
            #self.src_sent_cot.append(self.examples_cot[i][:sep_idx-1])
            #self.tgt_sent_cot.append(self.examples_cot[i][sep_idx-1:])
            #self.labels_cot[i][:sep_idx] = [-100] * sep_idx
            assert self.labels_all[i][sep_idx-1] == separator
            self.labels_all[i][:sep_idx] = [-100] * sep_idx
            #self.labels_cot_shift[i][:sep_idx-1] = [-100] * (sep_idx-1)
            temp_src_len += sep_idx-1
            temp_tgt_len += len(elem) - (sep_idx-1)
            temp_count += 1
        #import ipdb; ipdb.set_trace()
        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len/temp_tgt_len)

        #self.src_sent_nocot = []
        #self.tgt_sent_nocot = []
        #temp_src_len = 0
        #temp_tgt_len = 0
        #temp_count = 0
        #separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        #for i, elem in enumerate(self.labels_nocot):
        #    sep_idx = elem.index(separator) + 1
        #    self.src_sent_nocot.append(self.examples_nocot[i][:sep_idx-1])
        #    self.tgt_sent_nocot.append(self.examples_nocot[i][sep_idx-1:])
        #    self.labels_nocot[i][:sep_idx] = [-100] * sep_idx
        #    temp_src_len += sep_idx-1
        #    temp_tgt_len += len(elem) - (sep_idx-1)
        #    temp_count += 1

        #print('tgt_avg: ', temp_tgt_len / temp_count)
        #print('src_avg: ', temp_src_len / temp_count)
        #print('ratios: ', temp_src_len/temp_tgt_len)


        #import ipdb; ipdb.set_trace()


        #print(edited_sents_all[0])
        #print(self.labels_cot[0])
        #print(self.labels_nocot[0])
        #print(self.examples_nocot[0])
        #print(edited_sents_nocot[0])
        #print(self.src_sent_nocot[0])
        #print(self.tgt_sent_nocot[0])

    def __len__(self):
        #return len(self.examples_cot)
        return len(self.examples_all)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (#torch.tensor(self.examples_cot[i], dtype=torch.long),
                #torch.tensor(self.examples_nocot[i], dtype=torch.long),
                #torch.tensor(self.labels_cot[i], dtype=torch.long),
                #torch.tensor(self.labels_cot_shift[i], dtype=torch.long),
                #torch.tensor(self.labels_nocot[i], dtype=torch.long),
                #torch.tensor(self.src_sent_cot[i], dtype=torch.long),
                #torch.tensor(self.src_sent_nocot[i], dtype=torch.long),
                #torch.tensor(self.tgt_sent_cot[i], dtype=torch.long),
                #torch.tensor(self.tgt_sent_nocot[i], dtype=torch.long),
                #torch.tensor(self.examples_only[i], dtype=torch.long),
                torch.tensor(self.examples_all[i], dtype=torch.long),
                torch.tensor(self.labels_all[i], dtype=torch.long),
                self.probe_labels_all[i],
                )
@dataclass
class CoTDataCollator:
    """
    VAEData collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids_all, labels_all, probe_labels_all = zip(*examples)
        max_len_probe_labels = max([probe_labels.shape[0] for probe_labels in probe_labels_all])
        #import pdb; pdb.set_trace()
        #input_ids_cot, input_ids_nocot, labels_cot, labels_cot_shift, labels_nocot, src_cot, src_nocot, tgt_cot, tgt_nocot, input_ids_only, input_ids_all, labels_all = zip(*examples)
        probe_labels_all_tensor = torch.zeros(len(probe_labels_all), max_len_probe_labels,  probe_labels_all[0].shape[-1]).long().fill_(-1)
        for ip, probe_labels in enumerate(probe_labels_all):
            probe_labels_all_tensor[ip, :probe_labels.shape[0], :] = probe_labels
        #input_ids_cot = self._tensorize_batch(input_ids_cot)
        #input_ids_cot[input_ids_cot.lt(0)] = self.tokenizer.eos_token_id
        #input_ids_only = self._tensorize_batch(input_ids_only)
        #input_ids_only[input_ids_only.lt(0)] = self.tokenizer.eos_token_id
        input_ids_all = self._tensorize_batch(input_ids_all)
        input_ids_all[input_ids_all.lt(0)] = self.tokenizer.eos_token_id
        #input_ids_nocot = self._tensorize_batch(input_ids_nocot)
        #input_ids_nocot[input_ids_nocot.lt(0)] = self.tokenizer.eos_token_id
        #labels_cot = self._tensorize_batch(labels_cot)
        labels_all = self._tensorize_batch(labels_all)
        #labels_cot_shift = self._tensorize_batch(labels_cot_shift)
        #labels_nocot = self._tensorize_batch(labels_nocot)
        #return {"input_ids_cot": input_ids_cot, "input_ids_nocot": input_ids_nocot, "labels_cot": labels_cot, "labels_cot_shift": labels_cot_shift, "labels_nocot": labels_nocot, 'input_ids_only': input_ids_only, 'input_ids_all': input_ids_all, 'labels_all': labels_all}
        return {'input_ids_all': input_ids_all, 'labels_all': labels_all, 'probe_labels_all': probe_labels_all_tensor}

    def _tensorize_batch(self, examples):
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=-100)
