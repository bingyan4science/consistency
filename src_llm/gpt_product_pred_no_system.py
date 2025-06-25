from openai import OpenAI
import sys
import logging
import pandas as pd
import numpy as np
import argparse
import os
import torch

api_key = 'API_KEY'
client = OpenAI(api_key=api_key)

def extract_answer(text):
    split_pattern = '####'
    if split_pattern not in text:
        return text.strip().replace(',', '')
    else:
        _, ans = text.strip().split('####', 1)
        #ans = '####' + ans
        ans = ans.strip().replace(',', '')
        return ans


def pred_reaction_smiles(reaction, model, mode):
    if mode == "smiles":
        instruction = "Given the SMILES strings of reactants and reagents, predict the SMILES string of the product. Please output the product directly. For example:\nInput: CC1=CC=C(C#N)C([N+](=O)[O-])=C1.CN(C)C=O.CNC.O\nOutput: CN(C)C=CC1=CC=C(C#N)C=C1[N+](=O)[O-]\n"
    elif mode == "iupac":
        instruction = "Given the IUPAC names of reactants and reagents, predict the SMILES string of the product. Please output the product directly. For example:\nInput: 4-methyl-2-nitrobenzonitrile.N,N-dimethylformamide.N-methylmethanamine.oxidane\nOutput: CN(C)C=CC1=CC=C(C#N)C=C1[N+](=O)[O-]\n"
    message = reaction
    messages = []
    #import ipdb; ipdb.set_trace()
    if message:
        full_prompt = instruction + 'Input: ' + message
        messages.append({"role": "user", "content": full_prompt})
        #logging.info(f'user: {message}')
        chat = client.chat.completions.create(model=model, messages=messages)
        reply = chat.choices[0].message.content
        #logging.info(f'assistant: {reply}')
        messages.append({"role": "assistant", "content": reply})
    return reply


def pred_reaction_iupac(reaction, model, mode):
    if mode == "smiles":
        instruction = "Given the SMILES strings of reactants and reagents, predict the IUPAC name of the product. Please output the product directly. For example:\nInput: CC1=CC=C(C#N)C([N+](=O)[O-])=C1.CN(C)C=O.CNC.O\nOutput: 4-[2-(dimethylamino)ethenyl]-3-nitrobenzonitrile\n"
    elif mode == "iupac":
        instruction = "Given the IUPAC names of reactants and reagents, predict the IUPAC name of the product. Please output the product directly. For example:\nInput: 4-methyl-2-nitrobenzonitrile.N,N-dimethylformamide.N-methylmethanamine.oxidane\nOutput: 4-[2-(dimethylamino)ethenyl]-3-nitrobenzonitrile\n"
    message = reaction
    messages = []
    #import ipdb; ipdb.set_trace()
    if message:
        full_prompt = instruction + 'Input: ' + message
        messages.append({"role": "user", "content": full_prompt})
        #logging.info(f'user: {message}')
        chat = client.chat.completions.create(model=model, messages=messages)
        reply = chat.choices[0].message.content
        #logging.info(f'assistant: {reply}')
        messages.append({"role": "assistant", "content": reply})
    return reply

def pred_trans_smiles(smiles, model):
    instruction = "Given the SMILES string of a compound, output its IUPAC name. For example:\nInput: CN(C)C=CC1=CC=C(C#N)C=C1[N+](=O)[O-]\nOutput: 4-[2-(dimethylamino)ethenyl]-3-nitrobenzonitrile\n"
    message = smiles
    messages = []
    if message:
        full_prompt = instruction + 'Input: ' + message
        messages.append({"role": "user", "content": full_prompt})
        chat = client.chat.completions.create(model=model, messages=messages)
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
    return reply


def pred_trans_iupac(iupac, model):
    instruction = "Given the IUPAC name of a compound, output its SMILES string. For example:\nInput: 4-[2-(dimethylamino)ethenyl]-3-nitrobenzonitrile\nOutput: CN(C)C=CC1=CC=C(C#N)C=C1[N+](=O)[O-]\n"
    message = iupac
    messages = []
    if message:
        full_prompt = instruction + 'Input: ' + message
        messages.append({"role": "user", "content": full_prompt})
        chat = client.chat.completions.create(model=model, messages=messages)
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
    return reply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--base_model', type=str, default='gpt-4')
    parser.add_argument('--start_idx_smiles', type=int, default=0)
    parser.add_argument('--start_idx_iupac', type=int, default=0)
    parser.add_argument('--task', type=str, default='reaction_pred')
    args = parser.parse_args()
    print (args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_smiles = os.path.join(args.data_folder, 'test_smiles.txt')
    input_iupac = os.path.join(args.data_folder, 'test_iupac.txt')

    for split, input_file in [('smiles', input_smiles), ('iupac', input_iupac)]:
        #if split == 'smiles':
        #    continue
        if split == 'smiles':
            start_idx = args.start_idx_smiles
        elif split == 'iupac':
            start_idx = args.start_idx_iupac
        total_instances = 0
        total_correct = 0
        tgt_all = []
        predicted_all = []
        columns = ['Target', 'Predicted']
        df = pd.DataFrame(columns = columns)
        #import ipdb; ipdb.set_trace()
        with open(input_file, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.readlines() if (len(line) > 0 and not line.isspace()
                                                                                and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)
        for src, tgt in zip(src_lines[start_idx:], tgt_lines[start_idx:]):
            total_instances += 1
            #if total_instances < 175:
            #    continue
            src = src.strip()
            src = src[:-2]
            ans = extract_answer(tgt)
            if src is None:
                src = ''
            if ans is None:
                ans = ''
            tgt_all.append(ans)

            if args.task == 'translation':
                if split == 'smiles':
                    pred = pred_trans_smiles(src, model=args.base_model)
                elif split == 'iupac':
                    pred = pred_trans_iupac(src, model=args.base_model)
            elif args.task == 'reaction_pred':
                if split == 'smiles':
                    pred = pred_reaction_smiles(src, model=args.base_model, mode=args.mode)
                elif split == 'iupac': 
                    pred = pred_reaction_iupac(src, model=args.base_model, mode=args.mode)
            pred = pred.strip('\n')
            pred = pred.strip('\"')
            pred = pred.replace("'", "")
            pred = pred.replace(",", "")
            pred = pred.strip()
            
            predicted_all.append(pred)
            if pred == ans:
                total_correct += 1
            
            #if total_instances % 10 == 0:
            print (f'Input: {src}')
            print (f'Target: {ans}')
            print (f'Predicted: {pred}')
            print ('')
            sys.stdout.flush()
        
        accuracy = total_correct / total_instances
        print(f'accuracy is {accuracy}.')
        df['Target'] = tgt_all
        df[f'Predicted'] = predicted_all
        dirname = f'{args.save_model}'
        os.makedirs(dirname, exist_ok=True)
        df.to_csv(f'{dirname}/pred_results_mode{args.mode}_split{split}.csv', index=False)


if __name__ == "__main__":
    main()