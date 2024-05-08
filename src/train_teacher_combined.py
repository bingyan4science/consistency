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


def save_model(model, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, teacher, max_new_tokens, epoch, save_model, generate_onebatch=False):
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
        with ctx:
            outputs = teacher.compute_loss(input_ids=input_ids_all, labels=labels)
        total_loss += outputs.total_loss.item()
        total_correct_tokens += outputs.total_correct.item()
        total_tokens += outputs.total_tokens
        total_instances += batch_size

        # Generate
        if generate_onebatch and batch_id > 16:
            continue
        beam_output = teacher.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
        # Evaluate
        #import pdb; pdb.set_trace()
        for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_output)):
            end_idx = len(input_ids_all_i)-1
            while input_ids_all_i[end_idx] == tokenizer.eos_token_id:
                end_idx -= 1
            input_ids_all_i = input_ids_all_i[:end_idx+2]
            #sep_position = sep_positions[i].item()
            sep_position = [ii for ii, n in enumerate(input_ids_all_i.view(-1)) if n == tokenizer.eos_token_id][-3]
            tgt = input_ids_all_i[sep_position+1:]
            tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
            ans = extract_answer(tgt_text)
            pred_text = tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
            pred_ans = extract_answer(pred_text)
            if ans == pred_ans:
                total_correct += 1
            tgt_all.append(ans.split("#### ")[-1])
            predicted_all.append(pred_ans.split("#### ")[-1])
        
            if i == 0 or generate_onebatch:
                print (f'Input: {tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
                print (f'Target: {tgt_text}')
                print (f'Predicted: {pred_text}')
                print ('')
        if generate_onebatch:
            accuracy = total_correct / total_instances
    if not generate_onebatch:
        accuracy = total_correct / total_instances
    token_accuracy = total_correct_tokens / total_tokens
    loss = total_loss / total_tokens
    ppl = math.exp(loss)
    df['Target'] = tgt_all
    df['Predicted'] = predicted_all
    df.to_csv(f'{save_model}/valid_results_epoch{epoch}.csv', index=False)
    return accuracy, token_accuracy, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--base_model', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--generate_onebatch', action='store_true')
    parser.set_defaults(generate_onebatch=False)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--nopretrain', action='store_true')
    parser.set_defaults(nopretrain=False)
    args = parser.parse_args()

    print (args)

    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Create Model
    config = TeacherConfig(base_model=args.base_model)
    teacher = Teacher(config).to(device).to(ptdtype)
    if args.nopretrain:
        print ('reinitializing weights')
        teacher.base_model.apply(teacher.base_model._init_weights)

    # Load data
    tokenizer = teacher.tokenizer
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 1024)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, 1024)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    # set a breakpoint to see the tokenized data
    #ipdb.set_trace()
    for batch in train_dataloader:
        break

    # Create Optimizer
    trainable_params = list(teacher.parameters())
    use_fused = True
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    teacher.train()
    step = 0

    # Train
    best_ppl = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        teacher.train()
        for batch in tqdm.tqdm(train_dataloader):
            input_ids = batch['input_ids_all'].to(device)
            labels = batch['labels_all'].to(device)
            with ctx:
                outputs = teacher.compute_loss(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            if step % 100 == 0:
                print (f"loss: {loss}")
            token_accuracy = outputs.token_accuracy.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            ppl = loss.exp().item()
            if step % 100 == 0:
                print (f"Step: {step}. PPL: {ppl}. loss: {loss}. Token Accuracy: {token_accuracy}")
            step += 1
        accuracy, token_accuracy, ppl = evaluate(val_dataloader, tokenizer, ctx, teacher, args.max_new_tokens, epoch, args.save_model, args.generate_onebatch)
        if ppl < best_ppl:
            print ('best val ppl')
            best_ppl = ppl
        print (f'Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
        teacher.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))

if __name__ == "__main__":
    main()
