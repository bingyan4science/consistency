from dataclasses import dataclass
import os
import copy
import torch
import random
import numpy as np
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

class Dataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length, base_model, shuffle=False):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        print (f'Creating features from dataset file at {file_path}')
        eos_tok = tokenizer.eos_token

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.readlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        if shuffle:
            random.shuffle(lines)
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        edited_sents_all = []
        edited_src_all = []
        edited_tgt_all = []
        #import ipdb; ipdb.set_trace()
        for src, tgt in zip(src_lines, tgt_lines):
            src = src.strip().split('.')
            src = f' {eos_tok} '.join(src)
            ans = extract_answer(tgt)
            if src is None:
                src = ''
            if ans is None:
                ans = ''
            if base_model == "Salesforce/codet5-small":
                sent_src = ' {} {} '.format(src, eos_tok)
                sent_tgt = ' {} {} '.format(ans, eos_tok)
                edited_src_all.append(sent_src)
                edited_tgt_all.append(sent_tgt)
            else:
                sent = ' {} {} '.format(src, eos_tok) + ' {} '.format(eos_tok) + ans + ' {}'.format(eos_tok)
                edited_sents_all.append(sent)
        
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        separator = tokenizer.eos_token_id
        lens = []
        tgt_lens = []
        if base_model == "Salesforce/codet5-small":
            src_encoding_all = tokenizer(
                edited_src_all, 
                add_special_tokens=True, 
                truncation=True, 
                max_length=max_length
                )
            tgt_encoding_all = tokenizer(
                edited_tgt_all, 
                add_special_tokens=True, 
                truncation=True, 
                max_length=max_length
                )
            self.examples_all = src_encoding_all["input_ids"]
            #labels = tgt_encoding_all["input_ids"]
            #labels = [[l if l != tokenizer.pad_token_id else -100 for l in label] for label in labels]
            self.labels_all = tgt_encoding_all["input_ids"]

            for i, (src_elem, tgt_elem) in enumerate(zip(self.examples_all, self.labels_all)):
                #src_sep_idx = [i for i, n in enumerate(src_elem) if n == separator][-1]
                #tgt_sep_idx = [i for i, n in enumerate(tgt_elem) if n == separator][-1]
                temp_src_len += len(src_elem)
                temp_tgt_len += len(tgt_elem)
                temp_count += 1
                lens.append(len(src_elem)+len(tgt_elem))
                tgt_lens.append(len(tgt_elem))
            
        else:
            batch_encoding_all = tokenizer(edited_sents_all, add_special_tokens=True, truncation=True, max_length=max_length)
            self.examples_all = batch_encoding_all["input_ids"]
            self.labels_all = copy.deepcopy(self.examples_all)

            for i, elem in enumerate(self.labels_all):
                try:
                    sep_idx = [i for i, n in enumerate(elem) if n == separator][-3] + 1
                except:
                    sep_idx = len(self.labels_all[i])
                    self.labels_all[i][sep_idx-1] = separator
                assert self.labels_all[i][sep_idx-1] == separator
                self.labels_all[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx-1
                temp_tgt_len += len(elem) - (sep_idx-1)
                temp_count += 1
                lens.append(len(elem))
                tgt_lens.append(len(elem) - (sep_idx-1))
        
        lens = np.array(lens)
        tgt_lens = np.array(tgt_lens)
        for p in [20, 40, 60, 80, 90, 95, 96, 97, 98, 99]:
            print (f'p: {p}, {np.percentile(lens, p)}')
        for p in [20, 40, 60, 80, 90, 95, 96, 97, 98, 99]:
            print (f'tgt p: {p}, {np.percentile(tgt_lens, p)}')
        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len/temp_tgt_len)


    def __len__(self):
        return len(self.examples_all)

    def __getitem__(self, i):
        return (torch.tensor(self.examples_all[i], dtype=torch.long),
                torch.tensor(self.labels_all[i], dtype=torch.long),
                )
@dataclass
class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        #import ipdb; ipdb.set_trace()
        input_ids_all, labels_all = zip(*examples)
        input_ids_all = self._tensorize_batch(input_ids_all)
        input_ids_all[input_ids_all.lt(0)] = self.tokenizer.eos_token_id
        labels_all = self._tensorize_batch(labels_all)
        return {'input_ids_all': input_ids_all, 'labels_all': labels_all}

    def _tensorize_batch(self, examples):
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=-100)
