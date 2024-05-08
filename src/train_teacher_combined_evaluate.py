import math
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import argparse
import os
import sys
import tqdm
from models.teacher import Teacher
from models.configuration_teacher import TeacherConfig
from combineddata import CoTDataset, CoTDataCollator, extract_answer
import logging
import pandas as pd

from utils import get_sep_position
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
def evaluate(dataloader, tokenizer, ctx, teacher, max_new_tokens, epoch, save_model, split):
    teacher.eval()
    total_instances = 0
    total_tokens = 0
    total_correct = 0
    total_correct_tokens = 0
    total_loss = 0
    tgt_all = []
    predicted_all = []
    df = pd.DataFrame(columns = ['Target', 'Predicted'])
    batch_id = -1
    for batch in tqdm.tqdm(dataloader):
        batch_id += 1
        input_ids_all = batch['input_ids_all'].to(device)
        labels = batch['labels_all'].to(device)
        # Remove answer part
        #sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        #input_ids = input_ids_all[:, :sep_positions.max().item()+1]
        input_ids = input_ids_all
        batch_size = input_ids.shape[0]
        total_instances += batch_size

        # Generate
        beam_output = teacher.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
        # Evaluate
        for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_output)):
            end_idx = len(input_ids_all_i)-1
            while input_ids_all_i[end_idx] == tokenizer.eos_token_id:
                end_idx -= 1
            input_ids_all_i = input_ids_all_i[:end_idx+2]
            #sep_position = sep_positions[i].item()
            sep_position = [i for i, n in enumerate(input_ids_all_i.view(-1)) if n == tokenizer.eos_token_id][-3]
            tgt = input_ids_all_i[sep_position+1:]
            tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
            ans = extract_answer(tgt_text)
            pred_text = tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
            pred_ans = extract_answer(pred_text)
            if ans == pred_ans:
                total_correct += 1

            #import pdb; pdb.set_trace()
            tgt_all.append(ans.strip().split("####")[1].strip())
            predicted_all.append(pred_ans.split("####")[1].strip())
        
            if i == 0:
                print (f'Input: {tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
                print (f'Target: {tgt_text}')
                print (f'Predicted: {pred_text}')
                print ('')
    accuracy = total_correct / total_instances
    #import pdb; pdb.set_trace()
    df['Target'] = tgt_all
    df['Predicted'] = predicted_all
    df.to_csv(f'{save_model}/valid_results_epoch{epoch}_split{split}.csv', index=False)
    return accuracy, 0, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--base_model', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    args.batch_size = 1
    args.batch_size = 16

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
            out_file = f'{args.save_model}/valid_results_epoch{epoch}_split{split}.csv'
            if not os.path.exists(out_file):
                flag = False
        if flag:
            continue
        print (f'Generating: {epoch}')
        teacher = Teacher.from_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))
        teacher = teacher.to(device).to(ptdtype)

        # Load data
        if first:
            first = False
            tokenizer = teacher.tokenizer
            collate_fn = CoTDataCollator(tokenizer)
            smiles_dataset = CoTDataset(tokenizer, os.path.join(args.data_folder, 'test_smiles.txt'), 1024)
            smiles_dataloader = DataLoader(smiles_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
            iupac_dataset = CoTDataset(tokenizer, os.path.join(args.data_folder, 'test_iupac.txt'), 1024)
            iupac_dataloader = DataLoader(iupac_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
        for split, dataloader in [('smiles', smiles_dataloader), ('iupac', iupac_dataloader)]:
            accuracy, token_accuracy, ppl = evaluate(dataloader, tokenizer, ctx, teacher, args.max_new_tokens, epoch, args.save_model, split)
            print (f'Epoch: {epoch}, split: {split}, accuracy: {accuracy}')

if __name__ == "__main__":
    main()
