import math
import torch
from torch.utils.data import DataLoader
import argparse
import os
import sys
import tqdm
from models.model import Model
from models.configuration import Config
from data import Dataset, DataCollator, extract_answer
import logging
import pandas as pd
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import copy
import time  # Add time module

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere


def tensorize_batch(examples):
    # In order to accept both lists of lists and lists of Tensors
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)
    else:
        return pad_sequence(examples, batch_first=True, padding_value=-100)

@torch.no_grad()
def sample_sequences(model, tokenizer, max_new_tokens, num_samples, input_ids_1, input_ids_2, train=False):
    if not train:
        model.eval()
    import ipdb; ipdb.set_trace()
    sampled_sequences_1 = [[] for _ in range(num_samples)]
    sampled_labels_1 = [[] for _ in range(num_samples)]
    sampled_sequences_2 = [[] for _ in range(num_samples)]
    sampled_labels_2 = [[] for _ in range(num_samples)]
    beam_output = model.generate(
        input_ids=input_ids_1,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_samples
        )
    if model.config.base_model == "Salesforce/codet5-small":
        for beam_output_i in beam_output:
            assert num_samples == beam_output_i.shape[0]
            for j in range(num_samples):
                sampled_labels_1[j].append(beam_output_i[j])
        for j, label in enumerate(sampled_labels_1):
            sampled_sequences_1[j] = input_ids_1
            sampled_sequences_2[j] = input_ids_2
            sampled_labels_1[j] = torch.tensor(label)
            sampled_labels_2[j] = torch.tensor(label)
        return sampled_sequences_1, sampled_labels_1, sampled_sequences_2, sampled_labels_2



    for beam_output_i in beam_output:
        assert num_samples == beam_output_i.shape[0]
        for j in range(num_samples):
            sampled_sequences_1[j].append(beam_output_i[j])   
    
    for j, seq in enumerate(sampled_sequences_1):
        labels_1 = copy.deepcopy(seq)
        seq_2 = []
        labels_2 = []
        separator = tokenizer.eos_token_id
        assert len(seq) == len(input_ids_2)
        for i, (elem, input_2) in enumerate(zip(seq, input_ids_2)):
            end_idx = len(elem)-1
            if end_idx < 0:
                continue
            while elem[end_idx] == tokenizer.eos_token_id:
                end_idx -= 1
            elem = elem[:end_idx+2]
            labels_1[i][end_idx+2:] = -100

            end_idx = len(input_2)-1
            while input_2[end_idx] == tokenizer.eos_token_id:
                end_idx -= 1
            input_2 = input_2[:end_idx+2]
            try:
                sep_idx_1 = [i for i, n in enumerate(elem) if n == separator][-3] + 1
                sep_idx_2 = [i for i, n in enumerate(input_2) if n == separator][-3] + 1
            except:
                sep_idx_1 = len(labels_1[i])
                labels_1[i][sep_idx_1-1] = separator
                sep_idx_2 = len(input_2)
                input_2[sep_idx_2-1] = separator
            assert labels_1[i][sep_idx_1-1] == separator
            assert input_2[sep_idx_2-1] == separator
            for k in range(sep_idx_1):
                labels_1[i][k] = -100
            elem_2 = torch.cat((input_2[:sep_idx_2], labels_1[i][sep_idx_1:]), dim=0)
            seq_2.append(elem_2)
            label_2 = copy.deepcopy(elem_2)
            for k in range(sep_idx_2):
                label_2[k] = -100
            labels_2.append(label_2)
        seq = tensorize_batch(seq)
        seq[seq.lt(0)] = tokenizer.eos_token_id
        sampled_sequences_1[j] = seq

        labels_1 = tensorize_batch(labels_1)
        sampled_labels_1[j] = labels_1

        seq_2 = tensorize_batch(seq_2)
        seq_2[seq_2.lt(0)] = tokenizer.eos_token_id
        sampled_sequences_2[j] = seq_2

        labels_2 = tensorize_batch(labels_2)
        sampled_labels_2[j] = labels_2

    return sampled_sequences_1, sampled_labels_1, sampled_sequences_2, sampled_labels_2


def compute_consist_loss(model1, model2, tokenizer, max_new_tokens, num_samples, input_ids_a, input_ids_b, device_a, device_b, weight, ctx, accumulate, train=False):
    input_ids_b = input_ids_b.to(device_a)
    sampled_sequences11, sampled_labels11, sampled_sequences12, sampled_labels12 = sample_sequences(model1, tokenizer, max_new_tokens, num_samples, input_ids_a, input_ids_b, train=True)
    input_ids_a = input_ids_a.to(device_b)
    input_ids_b = input_ids_b.to(device_b)
    sampled_sequences22, sampled_labels22, sampled_sequences21, sampled_labels21 = sample_sequences(model2, tokenizer, max_new_tokens, num_samples, input_ids_b, input_ids_a, train=True)
    
    if not train:
        model1.eval()
        model2.eval()   
    
    kl1_list = []
    kl2_list = []
    kl_normalizer_1 = 0
    kl_normalizer_2 = 0
    for seq1, labels1, seq2, labels2 in zip(sampled_sequences11, sampled_labels11, sampled_sequences12, sampled_labels12):
        if seq1.shape[-1] > 340 or seq2.shape[-1] > 340:
            print ('skipped')
            continue
        
        with ctx:
            outputs_1 = model1.compute_loss(input_ids=seq1, labels=labels1, logits_only=True)
        seq2 = seq2.to(device_b)
        labels2 = labels2.to(device_b)
        with ctx:
            outputs_2 = model2.compute_loss(input_ids=seq2, labels=labels2, logits_only=True)
        log_probs_1 = outputs_1.logits[outputs_1.mask].log_softmax(dim=-1)
        probs_1 = outputs_1.logits[outputs_1.mask].softmax(dim=-1)
        log_probs_2 = outputs_2.logits[outputs_2.mask].log_softmax(dim=-1)
        log_probs_2 = log_probs_2.to(device_a)
        kl_div1 = (probs_1 * (log_probs_1 - log_probs_2)).sum(-1).sum()
        kl_normalizer_1 += outputs_1.mask.sum().item()
        kl1_list.append(kl_div1)

    if len(kl1_list) > 0:
        kl_loss1 = sum(kl1_list)
        kl_loss1 = kl_loss1 / kl_normalizer_1
        loss_1 = weight * kl_loss1
        loss_1.div(accumulate).backward()
        kl_loss1 = kl_loss1.item()
    else:
        kl_loss1 = 0
    
    for seq1, labels1, seq2, labels2 in zip(sampled_sequences21, sampled_labels21, sampled_sequences22, sampled_labels22):
        if seq1.shape[-1] > 340 or seq2.shape[-1] > 340:
            print ('skipped')
            continue
        seq1 = seq1.to(device_a)
        labels1 = labels1.to(device_a)
        with ctx:
            outputs_1 = model1.compute_loss(input_ids=seq1, labels=labels1, logits_only=True)
            outputs_2 = model2.compute_loss(input_ids=seq2, labels=labels2, logits_only=True)
        log_probs_1 = outputs_1.logits[outputs_1.mask].log_softmax(dim=-1)
        log_probs_1 = log_probs_1.to(device_b)
        log_probs_2 = outputs_2.logits[outputs_2.mask].log_softmax(dim=-1)
        probs_2 = outputs_2.logits[outputs_2.mask].softmax(dim=-1)
        kl_div2 = (probs_2 * (log_probs_2 - log_probs_1)).sum(-1).sum()
        kl_normalizer_2 += outputs_2.mask.sum().item()
        kl2_list.append(kl_div2)

    if len(kl2_list) > 0:
        kl_loss2 = sum(kl2_list)
        kl_loss2 = kl_loss2 / kl_normalizer_2
        loss_2 = weight * kl_loss2
        loss_2.div(accumulate).backward()
        kl_loss2 = kl_loss2.item() 
    else:
        kl_loss2 = 0
    
    return kl_loss1, kl_loss2

def save_model(model, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

@torch.no_grad()
def evaluate(device_a, device_b, dataloader_a, dataloader_b, tokenizer, ctx, model_a, model_b, max_new_tokens, epoch, save_model_a, save_model_b, num_samples, generate_onebatch=False):
    model_a.eval()
    model_b.eval()
    total_instances = 0
    total_tokens = 0
    total_correct_a = 0
    total_correct_b = 0
    total_correct_tokens_a = 0
    total_correct_tokens_b = 0
    total_loss_a = 0
    total_loss_b = 0
    total_kl_a = 0
    total_kl_b = 0
    kl_normalizer_1 = 0
    kl_normalizer_2 = 0
    tgt_all_a = []
    tgt_all_b = []
    predicted_all_a = []
    predicted_all_b = []
    df_a = pd.DataFrame(columns = ['Target', 'Predicted'])
    df_b = pd.DataFrame(columns = ['Target', 'Predicted'])
    batch_id = -1
    dataloader_length = len(dataloader_a)
    dataloader_length = len(dataloader_b)
    
    for batch_a, batch_b in tqdm.tqdm(zip(dataloader_a, dataloader_b), total=dataloader_length):
        batch_id += 1
        first_batch_a = True
        first_batch_b = True
        input_ids_all_a = batch_a['input_ids_all'].to(device_a)
        labels_a = batch_a['labels_all'].to(device_a)
        input_ids_all_b = batch_b['input_ids_all'].to(device_b)
        labels_b = batch_b['labels_all'].to(device_b)
        mask_a = labels_a[...,1:].ge(0)
        mask_b = labels_b[...,1:].ge(0)
        if mask_a.sum().item() != mask_b.sum().item():
            print ('skip')
            continue
        
        input_ids_a = input_ids_all_a
        input_ids_b = input_ids_all_b
        batch_size = input_ids_a.shape[0]
        with ctx:
            outputs_a = model_a.compute_loss(input_ids=input_ids_all_a, labels=labels_a, validation=True)
            outputs_b = model_b.compute_loss(input_ids=input_ids_all_b, labels=labels_b, validation=True)
            
        if generate_onebatch and batch_id > 16:
            continue
        input_ids_b = input_ids_b.to(device_a)
        sampled_sequences11, sampled_labels11, sampled_sequences12, sampled_labels12 = sample_sequences(model_a, tokenizer, max_new_tokens, num_samples, input_ids_a, input_ids_b)
        input_ids_a = input_ids_a.to(device_b)
        input_ids_b = input_ids_b.to(device_b)
        sampled_sequences22, sampled_labels22, sampled_sequences21, sampled_labels21 = sample_sequences(model_b, tokenizer, max_new_tokens, num_samples, input_ids_b, input_ids_a)
        
        # Compute log-probabilities of the sampled sequences
        kl1_list = []
        kl2_list = []
        
        for seq1, labels1, seq2, labels2 in zip(sampled_sequences11, sampled_labels11, sampled_sequences12, sampled_labels12):
            with ctx:
                outputs_1 = model_a.compute_loss(input_ids=seq1, labels=labels1, logits_only=True)
            seq2 = seq2.to(device_b)
            labels2 = labels2.to(device_b)
            with ctx:
                outputs_2 = model_b.compute_loss(input_ids=seq2, labels=labels2, logits_only=True)
            kl_div1 = 0
            log_probs_1 = outputs_1.logits[outputs_1.mask].log_softmax(dim=-1)
            probs_1 = outputs_1.logits[outputs_1.mask].softmax(dim=-1)
            log_probs_2 = outputs_2.logits[outputs_2.mask].log_softmax(dim=-1)
            log_probs_2 = log_probs_2.to(device_a)
            kl_div1 += (probs_1 * (log_probs_1 - log_probs_2)).sum(-1).sum(0)
            kl_normalizer_1 += outputs_1.mask.sum().item()
            kl1_list.append(kl_div1)
        kl_loss_1 = sum(kl1_list)
        for seq1, labels1, seq2, labels2 in zip(sampled_sequences21, sampled_labels21, sampled_sequences22, sampled_labels22):
            seq1 = seq1.to(device_a)
            labels1 = labels1.to(device_a)
            with ctx:
                outputs_1 = model_a.compute_loss(input_ids=seq1, labels=labels1, logits_only=True)
                outputs_2 = model_b.compute_loss(input_ids=seq2, labels=labels2, logits_only=True)
            kl_div2 = 0
            log_probs_1 = outputs_1.logits[outputs_1.mask].log_softmax(dim=-1)
            log_probs_1 = log_probs_1.to(device_b)
            log_probs_2 = outputs_2.logits[outputs_2.mask].log_softmax(dim=-1)
            probs_2 = outputs_2.logits[outputs_2.mask].softmax(dim=-1)
            kl_div2 += (probs_2 * (log_probs_2 - log_probs_1)).sum(-1).sum(0)
            kl_normalizer_2 += outputs_1.mask.sum().item()
            kl2_list.append(kl_div2)
        kl_loss_2 = sum(kl2_list)

        total_loss_a += outputs_a.total_loss.item()
        total_loss_b += outputs_b.total_loss.item()
        total_kl_a += kl_loss_1.item()
        total_kl_b += kl_loss_2.item()
        total_correct_tokens_a += outputs_a.total_correct.item()
        total_correct_tokens_b += outputs_b.total_correct.item()

        total_tokens += outputs_a.total_tokens
        total_instances += batch_size * num_samples

        for seq_a, seq_b in zip(sampled_sequences11, sampled_sequences22):
            for i, (input_ids_a_i,seq_a_i) in enumerate(zip(input_ids_a, seq_a)):
                end_idx = len(input_ids_a_i)-1
                while input_ids_a_i[end_idx] == tokenizer.eos_token_id:
                    end_idx -= 1
                input_ids_a_i = input_ids_a_i[:end_idx+2]
                sep_position = [ii for ii, n in enumerate(input_ids_a_i.view(-1)) if n == tokenizer.eos_token_id][-3]
                tgt = input_ids_a_i[sep_position+1:]
                tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
                ans = extract_answer(tgt_text)
                pred_text = tokenizer.decode(seq_a_i[sep_position+1:], skip_special_tokens=True)
                pred_ans = extract_answer(pred_text)
                if ans == pred_ans:
                    total_correct_a += 1
                tgt_all_a.append(ans.split("#### ")[-1])
                predicted_all_a.append(pred_ans.split("#### ")[-1])
                if len(pred_ans.split("#### ")[-1]) == 0:
                    print("empty prediction")
                    print(tokenizer.decode(seq_a_i[sep_position+1:], skip_special_tokens=True))
            
                if i == 0 or generate_onebatch:
                    if first_batch_a and batch_id % 10 == 0:
                        print (f'Input a: {tokenizer.decode(input_ids_a_i[:sep_position], skip_special_tokens=True)}')
                        print (f'Target a: {tgt_text}')
                        print (f'Predicted a: {pred_text}')
                        print ('')
                        first_batch_a = False
            
            for i, (input_ids_b_i,seq_b_i) in enumerate(zip(input_ids_b, seq_b)):
                end_idx = len(input_ids_b_i)-1
                while input_ids_b_i[end_idx] == tokenizer.eos_token_id:
                    end_idx -= 1
                input_ids_b_i = input_ids_b_i[:end_idx+2]
                sep_position = [ii for ii, n in enumerate(input_ids_b_i.view(-1)) if n == tokenizer.eos_token_id][-3]
                tgt = input_ids_b_i[sep_position+1:]
                tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
                ans = extract_answer(tgt_text)
                pred_text = tokenizer.decode(seq_b_i[sep_position+1:], skip_special_tokens=True)
                pred_ans = extract_answer(pred_text)
                if ans == pred_ans:
                    total_correct_b += 1
                tgt_all_b.append(ans.split("#### ")[-1])
                predicted_all_b.append(pred_ans.split("#### ")[-1])
                if len(pred_ans.split("#### ")[-1]) == 0:
                    print("empty prediction")
                    print(tokenizer.decode(seq_b_i[sep_position+1:], skip_special_tokens=True))
            
                if i == 0 or generate_onebatch:
                    if first_batch_b and batch_id % 10 == 0:
                        print (f'Input b: {tokenizer.decode(input_ids_b_i[:sep_position], skip_special_tokens=True)}')
                        print (f'Target b: {tgt_text}')
                        print (f'Predicted b: {pred_text}')
                        print ('')
                        first_batch_b = False
        if generate_onebatch:
            accuracy_a = total_correct_a / total_instances
            accuracy_b = total_correct_b / total_instances
            if kl_normalizer_1 != 0:
                kl_loss_a = total_kl_a / kl_normalizer_1
            else:
                kl_loss_a = total_kl_a
            if kl_normalizer_2 != 0:
                kl_loss_b = total_kl_b / kl_normalizer_2
            else:
                kl_loss_b = total_kl_b
        
    if not generate_onebatch:
        accuracy_a = total_correct_a / total_instances
        accuracy_b = total_correct_b / total_instances
        kl_loss_a = total_kl_a / kl_normalizer_1
        kl_loss_b = total_kl_b / kl_normalizer_2
    token_accuracy_a = total_correct_tokens_a / total_tokens
    token_accuracy_b = total_correct_tokens_b / total_tokens
    loss_a = total_loss_a / total_tokens
    loss_b = total_loss_b / total_tokens
    ppl_a = math.exp(loss_a)
    ppl_b = math.exp(loss_b)
    df_a['Target'] = tgt_all_a
    df_b['Target'] = tgt_all_b
    df_a['Predicted'] = predicted_all_a
    df_b['Predicted'] = predicted_all_b
    df_a.to_csv(f'{save_model_a}/valid_results_epoch{epoch}.csv', index=False)
    df_b.to_csv(f'{save_model_b}/valid_results_epoch{epoch}.csv', index=False)
    return accuracy_a, token_accuracy_a, ppl_a, accuracy_b, token_accuracy_b, ppl_b, kl_loss_a, kl_loss_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path_a', type=str, required=True)
    #parser.add_argument('--train_path_a_2', type=str, required=True)
    parser.add_argument('--val_path_a', type=str, required=True)
    parser.add_argument('--train_path_b', type=str, required=True)
    #parser.add_argument('--train_path_b_2', type=str, required=True)
    parser.add_argument('--val_path_b', type=str, required=True)
    parser.add_argument('--save_model_a', type=str, required=True)
    parser.add_argument('--save_model_b', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--base_model', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight', type=float, default=0)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--pretrain_epochs', type=int, default=0)
    parser.add_argument('--generate_onebatch', action='store_true')
    parser.set_defaults(generate_onebatch=False)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--nopretrain', action='store_true')
    parser.set_defaults(nopretrain=False)
    args = parser.parse_args()

    print (args)
    #import ipdb; ipdb.set_trace()
    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device_a = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_b = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #device_b = device_a
    torch.cuda.set_device(device_a)
    torch.cuda.empty_cache()
    torch.cuda.set_device(device_b)
    torch.cuda.empty_cache()
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device_a, device_b)

    # Create Model
    config = Config(base_model=args.base_model)
    if args.base_model == "mistralai/Mistral-7B-v0.1":
        model_a = Model(config)
        model_b = Model(config)
    else:
        model_a = Model(config).to(device_a).to(ptdtype)
        model_b = Model(config).to(device_b).to(ptdtype)
    
    if args.nopretrain:
        print ('reinitializing weights')
        model_a.base_model.apply(model_a.base_model._init_weights)
        model_b.base_model.apply(model_b.base_model._init_weights)
    
    # Load data
    tokenizer = model_a.tokenizer
    collate_fn = DataCollator(tokenizer)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dataset_a = Dataset(tokenizer, args.train_path_a, 1024, args.base_model, shuffle=True)
    train_dataloader_a = DataLoader(train_dataset_a, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    #train_dataset_a_2 = Dataset(tokenizer, args.train_path_a_2, 1024, shuffle=True)
    #train_dataloader_a_2 = DataLoader(train_dataset_a_2, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    val_dataset_a = Dataset(tokenizer, args.val_path_a, 1024, args.base_model)
    val_dataloader_a = DataLoader(val_dataset_a, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dataset_b = Dataset(tokenizer, args.train_path_b, 1024, args.base_model, shuffle=True)
    train_dataloader_b = DataLoader(train_dataset_b, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    #train_dataset_b_2 = Dataset(tokenizer, args.train_path_b_2, 1024, shuffle=True)
    #train_dataloader_b_2 = DataLoader(train_dataset_b_2, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    val_dataset_b = Dataset(tokenizer, args.val_path_b, 1024, args.base_model)
    val_dataloader_b = DataLoader(val_dataset_b, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    dataloader_length = len(train_dataloader_a)
    dataloader_length = len(train_dataloader_b)    

    # Create Optimizer
    trainable_params_a = list(model_a.parameters())
    trainable_params_b = list(model_b.parameters())
    if args.base_model == "mistralai/Mistral-7B-v0.1":
        use_fused = False
    else:
        use_fused = True
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer_a = torch.optim.AdamW(trainable_params_a, lr=args.lr, **extra_args)
    optimizer_b = torch.optim.AdamW(trainable_params_b, lr=args.lr, **extra_args)

    model_a.train()
    model_b.train()
    step = 0

    # Start timing total training
    total_start_time = time.time()
    
    # Train
    best_ppl_a = float('inf')
    best_ppl_b = float('inf')
    for epoch in range(args.epochs):
        # Start timing this epoch
        epoch_start_time = time.time()
        print(f"Epoch {epoch}")
        model_a.train()
        model_b.train()
        for batch_a, batch_b in tqdm.tqdm(zip(train_dataloader_a, train_dataloader_b), total=dataloader_length):
            input_ids_a = batch_a['input_ids_all'].to(device_a)
            labels_a = batch_a['labels_all'].to(device_a)
            input_ids_b = batch_b['input_ids_all'].to(device_b)
            labels_b = batch_b['labels_all'].to(device_b)
            mask_a = labels_a[...,1:].ge(0)
            mask_b = labels_b[...,1:].ge(0)
            
            if input_ids_a.shape[-1] > 300 or input_ids_b.shape[-1] > 300:
                print ('skipped')
                continue
            if mask_a.sum().item() != mask_b.sum().item():
                print ('skip')
                continue
            with ctx:
                outputs_a = model_a.compute_loss(input_ids=input_ids_a, labels=labels_a)
                loss_a = outputs_a.loss
            loss_a.div(args.accumulate).backward()
            
            with ctx:
                outputs_b = model_b.compute_loss(input_ids=input_ids_b, labels=labels_b)
                loss_b = outputs_b.loss
            loss_b.div(args.accumulate).backward()

            if epoch > args.pretrain_epochs-1:
                loss_consist_a, loss_consist_b = compute_consist_loss(model_a, model_b, tokenizer, args.max_new_tokens, args.num_samples, input_ids_a, input_ids_b, device_a, device_b, args.weight, ctx, args.accumulate, train=True)
                
            else:
                loss_consist_a = 0
                loss_consist_b = 0

            if step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params_a, args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(trainable_params_b, args.max_grad_norm)
                optimizer_a.step()
                optimizer_b.step()
                optimizer_a.zero_grad(set_to_none=True)
                optimizer_b.zero_grad(set_to_none=True)
        
            if step % 100 == 0:
                ppl_a = loss_a.exp().item()
                ppl_b = loss_b.exp().item()
                token_accuracy_a = outputs_a.token_accuracy.item()
                token_accuracy_b = outputs_b.token_accuracy.item()
                print (f"Step: {step}. PPL a: {ppl_a}. PPL b: {ppl_b}. loss a: {loss_a}. loss b: {loss_b}. Token Accuracy a: {token_accuracy_a}. Token Accuracy b: {token_accuracy_b}. Consist loss: {loss_consist_a}. {loss_consist_b}.")
                sys.stdout.flush()
            step += 1
            break

            """
            input_ids_a = batch_a_2['input_ids_all'].to(device_a)
            labels_a = batch_a_2['labels_all'].to(device_a)
            input_ids_b = batch_b_2['input_ids_all'].to(device_b)
            labels_b = batch_b_2['labels_all'].to(device_b)
            mask_a = labels_a[...,1:].ge(0)
            mask_b = labels_b[...,1:].ge(0)
            
            if input_ids_a.shape[-1] > 300 or input_ids_b.shape[-1] > 300:
                print ('skipped')
                continue
            if mask_a.sum().item() != mask_b.sum().item():
                print ('skip')
                continue
            with ctx:
                outputs_a = model_a.compute_loss(input_ids=input_ids_a, labels=labels_a)
                loss_a = outputs_a.loss
            loss_a.div(args.accumulate).backward()
            
            with ctx:
                outputs_b = model_b.compute_loss(input_ids=input_ids_b, labels=labels_b)
                loss_b = outputs_b.loss
            loss_b.div(args.accumulate).backward()

            if epoch > args.pretrain_epochs-1:
                loss_consist_a, loss_consist_b = compute_consist_loss(model_a, model_b, tokenizer, args.max_new_tokens, args.num_samples, input_ids_a, input_ids_b, device_a, device_b, args.weight, ctx, args.accumulate, train=True)
                
            else:
                loss_consist_a = 0
                loss_consist_b = 0
            
            if step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params_a, args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(trainable_params_b, args.max_grad_norm)
                optimizer_a.step()
                optimizer_b.step()
                optimizer_a.zero_grad(set_to_none=True)
                optimizer_b.zero_grad(set_to_none=True)
        
            if step % 100 == 0:
                ppl_a = loss_a.exp().item()
                ppl_b = loss_b.exp().item()
                token_accuracy_a = outputs_a.token_accuracy.item()
                token_accuracy_b = outputs_b.token_accuracy.item()
                print (f"Step: {step}. PPL a: {ppl_a}. PPL b: {ppl_b}. loss a: {loss_a}. loss b: {loss_b}. Token Accuracy a: {token_accuracy_a}. Token Accuracy b: {token_accuracy_b}. Consist loss: {loss_consist_a}. {loss_consist_b}.")
                sys.stdout.flush()
            step += 1
            #break
        """
        accuracy_a, token_accuracy_a, ppl_a, accuracy_b, token_accuracy_b, ppl_b, kl_loss_a, kl_loss_b = evaluate(device_a, device_b, val_dataloader_a, val_dataloader_b, tokenizer, ctx, model_a, model_b, args.max_new_tokens, epoch, args.save_model_a, args.save_model_b, 1*args.num_samples, args.generate_onebatch)
        if ppl_a < best_ppl_a:
            print ('best val ppl a')
            best_ppl_a = ppl_a
        if ppl_b < best_ppl_b:
            print ('best val ppl b')
            best_ppl_b = ppl_b
        print (f'Val. PPL a: {ppl_a}; PPL b: {ppl_b}; Accuracy a: {accuracy_a}; Accuracy b: {accuracy_b}; Token Accuracy a: {token_accuracy_a}; Token Accuracy b: {token_accuracy_b}. Consist loss: {kl_loss_a}. {kl_loss_b}.')
        model_a.save_pretrained(os.path.join(args.save_model_a, f'checkpoint_{epoch}'))
        model_b.save_pretrained(os.path.join(args.save_model_b, f'checkpoint_{epoch}'))
        
        # Calculate and print epoch timing
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")

    # Calculate and print total training time
    total_training_time = time.time() - total_start_time
    print(f"\nTotal training completed in {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")

if __name__ == "__main__":
    main()
