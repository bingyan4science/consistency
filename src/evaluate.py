import torch
from torch.utils.data import DataLoader
import argparse
import os
import tqdm
from models.model import Model
from data import Dataset, DataCollator, extract_answer
import logging
import pandas as pd
import re

import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
torch.manual_seed(1234)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, model, max_new_tokens, epoch, save_model, split):
    model.eval()
    total_instances = 0
    total_correct = 0
    tgt_all = []
    
    num_return_sequences = 32
    predicted_all = [[] for _ in range(num_return_sequences)]
    columns = ['Target']
    for j in range(num_return_sequences):
        columns.append(f'Predicted_{j}')
    df = pd.DataFrame(columns = columns)
    batch_id = -1
    for batch in tqdm.tqdm(dataloader):
        batch_id += 1
        input_ids_all = batch['input_ids_all'].to(device)
        input_ids = input_ids_all
        batch_size = input_ids.shape[0]
        total_instances += batch_size * num_return_sequences

        # Generate
        beam_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences
        )
        # Evaluate
        if model.config.base_model == "Salesforce/codet5-small":
            labels_all = batch['labels_all'].to(device)
            for i, (input_ids_all_i, labels_all_i, beam_output_i) in enumerate(zip(input_ids_all, labels_all, beam_output)):
                end_idx = len(input_ids_all_i)-1
                while input_ids_all_i[end_idx] == tokenizer.eos_token_id:
                    end_idx -= 1
                input_ids_all_i = input_ids_all_i[:end_idx+1]
                end_idx = len(labels_all_i)-1
                while labels_all_i[end_idx] == -100:
                    end_idx -= 1
                labels_all_i = labels_all_i[:end_idx+1]
                tgt_text = tokenizer.decode(labels_all_i, skip_special_tokens=True)
                ans = extract_answer(tgt_text)

                assert num_return_sequences == beam_output_i.shape[0]
                for j in range(num_return_sequences):
                    sampled_label_i = beam_output_i[j]
                    end_idx = len(sampled_label_i)-1
                    while sampled_label_i[end_idx] == -100:
                        end_idx -= 1
                    sampled_label_i = sampled_label_i[:end_idx+1]
                    pred_text = tokenizer.decode(sampled_label_i, skip_special_tokens=True)
                    pred_ans = extract_answer(pred_text)
                    if ans == pred_ans:
                        total_correct += 1
                    if j == 0:
                        tgt_all.append(ans.strip().split("####")[1].strip())
                    try:
                        predicted_all[j].append(re.split(r'#*#', pred_ans)[1].strip())
                    except:
                        if "#" not in pred_ans:
                            predicted_all[j].append(pred_ans.strip())
                        else:
                            import ipdb; ipdb.set_trace()
                
                if i == 0:
                    print (f'Input: {tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
                    print (f'Target: {tgt_text}')
                    print (f'Predicted: {pred_text}')
                    print ('')
   
        else:
            for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_output)):
                end_idx = len(input_ids_all_i)-1
                while input_ids_all_i[end_idx] == tokenizer.eos_token_id:
                    end_idx -= 1
                input_ids_all_i = input_ids_all_i[:end_idx+2]
                sep_position = [i for i, n in enumerate(input_ids_all_i.view(-1)) if n == tokenizer.eos_token_id][-3]
                tgt = input_ids_all_i[sep_position+1:]
                tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
                ans = extract_answer(tgt_text)

                assert num_return_sequences == beam_output_i.shape[0]
                for j in range(num_return_sequences):
                    pred_text = tokenizer.decode(beam_output_i[j][sep_position+1:], skip_special_tokens=True)
                    pred_ans = extract_answer(pred_text)
                    if ans == pred_ans:
                        total_correct += 1

                    if j == 0:
                        tgt_all.append(ans.strip().split("####")[1].strip())
                    try:
                        predicted_all[j].append(re.split(r'#*#', pred_ans)[1].strip())
                    except:
                        if "#" not in pred_ans:
                            predicted_all[j].append(pred_ans.strip())
                        else:
                            import ipdb; ipdb.set_trace()
            
                if i == 0:
                    print (f'Input: {tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
                    print (f'Target: {tgt_text}')
                    print (f'Predicted: {pred_text}')
                    print ('')
    
    accuracy = total_correct / total_instances
    df['Target'] = tgt_all
    for j in range(num_return_sequences):
        df[f'Predicted_{j}'] = predicted_all[j]
    dirname = f'{save_model}/sample32'
    os.makedirs(dirname, exist_ok=True)
    df.to_csv(f'{dirname}/valid_results_epoch{epoch}_split{split}.csv', index=False)
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--base_model', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    print (args)

    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    first = True
    for epoch in range(args.epochs):
        flag = True
        for split in ['smiles', 'iupac']:
            out_file = f'{args.save_model}/sample32_trans/valid_results_epoch{epoch}_split{split}.csv'
            if not os.path.exists(out_file):
                flag = False
        if flag:
            continue
        print (f'Generating: {epoch}')
        model = Model.from_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))
        model = model.to(device).to(ptdtype)
        base_model = model.config.base_model

        # Load data
        if first:
            first = False
            tokenizer = model.tokenizer
            collate_fn = DataCollator(tokenizer)
            smiles_dataset = Dataset(tokenizer, os.path.join(args.data_folder, 'test_smiles.txt'), 1024, base_model)
            smiles_dataloader = DataLoader(smiles_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
            iupac_dataset = Dataset(tokenizer, os.path.join(args.data_folder, 'test_iupac.txt'), 1024, base_model)
            iupac_dataloader = DataLoader(iupac_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
        for split, dataloader in [('smiles', smiles_dataloader), ('iupac', iupac_dataloader)]:
            accuracy = evaluate(dataloader, tokenizer, ctx, model, args.max_new_tokens, epoch, args.save_model, split)
            print (f'Epoch: {epoch}, split: {split}, accuracy: {accuracy}')

if __name__ == "__main__":
    main()
