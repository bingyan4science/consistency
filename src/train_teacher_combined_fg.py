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
from fgcombineddata import CoTDataset, CoTDataCollator, extract_answer
import logging
import pandas as pd

from utils import get_sep_position
import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
torch.manual_seed(1234)

fgs = []
with open('function_groups.txt') as fin:
    for line in fin:
        fgs.append(line.strip())

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_loss(input_ids, probe_labels_all, hidden_states, tokenizer, model, ctx, layer):
    batch_size = input_ids.shape[0]
    with torch.no_grad():
        with ctx:
            hidden_states = hidden_states[layer]
            all_sel_states = []
            all_sel_labels = []
            reaction_molecule_ids_all = []
            for i in range(batch_size):
                input_ids_i = input_ids[i]
                end_idx = len(input_ids_i)-1
                while input_ids_i[end_idx] == tokenizer.eos_token_id:
                    end_idx -= 1
                input_ids_i = input_ids_i[:end_idx+2]
                sep_positions_i = [ii for ii, n in enumerate(input_ids_i.view(-1)) if n == tokenizer.eos_token_id][:-3]
                reaction_molecule_ids_all.append(sep_positions_i)
                sep_positions_i = torch.LongTensor(sep_positions_i).to(device).view(-1, 1).expand(-1, 768)
                hidden_states_i = hidden_states[i]
                sel_labels = probe_labels_all[i, :sep_positions_i.shape[0]]
                sel_states = hidden_states_i.gather(0, sep_positions_i)
                all_sel_states.append(sel_states)
                all_sel_labels.append(sel_labels)
            all_sel_states = torch.cat(all_sel_states, dim=0)
            all_sel_labels = torch.cat(all_sel_labels, dim=0)
    predicted_logits = model(all_sel_states)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(predicted_logits, all_sel_labels.float())
    normalized_logits = torch.nn.Sigmoid()(predicted_logits)
    predicted_labels = normalized_logits.gt(0.5).long()
    correct = predicted_labels == all_sel_labels
    correct_molecule_preds = correct.all(-1).long()
    correct_molecule_pred = correct_molecule_preds.sum().item()
    num_molecules = correct.shape[0]
    molecule_accuracy = correct_molecule_pred / num_molecules
    num_reactions = batch_size
    offset = 0
    correct_reactions = 0
    all_predictions = []
    all_targets = []
    #import pdb; pdb.set_trace()
    for i in range(batch_size):
        flag_correct = True
        predictions = []
        targets = []
        for _ in range(len(reaction_molecule_ids_all[i])):
            #predictions.append(':'.join([str(item) for item in predicted_labels[offset].cpu().tolist()]))
            #targets.append(':'.join([str(item) for item in all_sel_labels[offset].cpu().tolist()]))
            predictions.append(predicted_labels[offset].cpu())
            targets.append(all_sel_labels[offset].cpu())
            if correct_molecule_preds[offset] != 1:
                flag_correct = False
            offset += 1
        #all_predictions.append(' '.join(predictions))
        #all_targets.append(' '.join(targets))
        all_predictions.append(predictions)
        all_targets.append(targets)
        if flag_correct:
            correct_reactions += 1
    #import pdb; pdb.set_trace()
    reaction_accuracy = correct_reactions / num_reactions
    tp = (correct * all_sel_labels.gt(0)).sum().item()
    total = all_sel_labels.shape[0] * all_sel_labels.shape[1]
    pp = predicted_labels.gt(0).sum().item()
    precision = tp / max(pp, 1e-5)
    gp = all_sel_labels.gt(0).sum().item()
    recall = tp / max(gp, 1e-5)
    accuracy = correct.sum().item() / total
    stats = {
            "loss": [loss.item() * total, total],
            "precision": [tp, pp],
            "recall": [tp, gp],
            "accuracy": [correct.sum().item(), total],
            "molecule_accuracy": [correct_molecule_pred, num_molecules],
            "reaction_accuracy": [correct_reactions, num_reactions],
            }
    return loss, precision, recall, accuracy, molecule_accuracy, reaction_accuracy, all_predictions, all_targets, stats

def save_model(model, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, teacher, model, max_new_tokens, epoch, save_model, generate_onebatch=False, split='valid', layer=None):
    teacher.eval()
    tgt_all = []
    predicted_all = []
    df = pd.DataFrame(columns = ['Target', 'Predicted'])
    batch_id = -1
    all_stats = {}
    for batch in tqdm.tqdm(dataloader):
        batch_id += 1
        input_ids = batch['input_ids_all'].to(device)
        outputs = teacher.base_model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        labels = batch['labels_all'].to(device)
        batch_size = input_ids.shape[0]
        probe_labels_all = batch['probe_labels_all'].to(device)
        loss, precision, recall, accuracy, molecule_accuracy, reaction_accuracy, all_predictions, all_targets, stats = compute_loss(input_ids, probe_labels_all, hidden_states, tokenizer, model, ctx, layer)
        for target in all_targets:
            ttt = []
            for t in target:
                #import pdb; pdb.set_trace()
                nnz = t.nonzero().view(-1)
                items = [fgs[item] for item in nnz]
                if len(items) == 0:
                    items = ['<s>']
                ttt.append('.'.join(items))
            tgt_all.append(' '.join(ttt))
        for target in all_predictions:
            ttt = []
            for t in target:
                #import pdb; pdb.set_trace()
                nnz = t.nonzero().view(-1)
                items = [fgs[item] for item in nnz]
                if len(items) == 0:
                    items = ['<s>']
                ttt.append('.'.join(items))
            predicted_all.append(' '.join(ttt))
        #tgt_all.extend(all_targets)
        #predicted_all.extend(all_predictions)
        for k in stats:
            if k not in all_stats:
                all_stats[k] = [0, 0]
            all_stats[k] = [stats[k][0] + all_stats[k][0], stats[k][1] + all_stats[k][1]]
    precision = all_stats['precision'][0] / all_stats['precision'][1]
    recall = all_stats['recall'][0] / all_stats['recall'][1]
    token_accuracy = all_stats['accuracy'][0] / all_stats['accuracy'][1]
    molecule_accuracy = all_stats['molecule_accuracy'][0] / all_stats['molecule_accuracy'][1]
    reaction_accuracy = all_stats['reaction_accuracy'][0] / all_stats['reaction_accuracy'][1]
    loss = all_stats['loss'][0] / all_stats['loss'][1]
    ppl = math.exp(loss)

    df['Target'] = tgt_all
    df['Predicted'] = predicted_all
    df.to_csv(f'{save_model}/probe_{split}_results_epoch{epoch}.csv', index=False)
    return precision, recall, token_accuracy, molecule_accuracy, reaction_accuracy, ppl, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--from_pretrained', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--layer', type=int, default=5)
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
    #config = TeacherConfig(base_model=args.base_model)
    teacher = Teacher.from_pretrained(args.from_pretrained)
    #teacher = Teacher(config).to(device).to(ptdtype)
    teacher = teacher.to(device).to(ptdtype)

    model = torch.nn.Linear(768, 30).to(device).to(ptdtype)
    #if args.nopretrain:
    #    print ('reinitializing weights')
    #    teacher.base_model.apply(teacher.base_model._init_weights)

    # Load data
    tokenizer = teacher.tokenizer
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 1024)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, 1024)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    test_dataset = CoTDataset(tokenizer, args.test_path, 1024)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    # set a breakpoint to see the tokenized data
    #ipdb.set_trace()
    for batch in train_dataloader:
        break

    # Create Optimizer
    #trainable_params = list(teacher.parameters())
    trainable_params = list(model.parameters())
    use_fused = True
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    #teacher.train()
    teacher.eval()
    step = 0

    # Train
    best_ppl = float('inf')
    args.epochs = 1
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        #teacher.train()
        for batch in tqdm.tqdm(train_dataloader):
            #import pdb; pdb.set_trace()
            input_ids = batch['input_ids_all'].to(device)
            labels = batch['labels_all'].to(device)
            batch_size = input_ids.shape[0]
            probe_labels_all = batch['probe_labels_all'].to(device)
            with torch.no_grad():
                with ctx:
                    outputs = teacher.base_model(input_ids=input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states

            loss, precision, recall, accuracy, molecule_accuracy, reaction_accuracy, _, _, _ = compute_loss(input_ids, probe_labels_all, hidden_states, tokenizer, model, ctx, args.layer)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            ppl = loss.exp().item()
            if step % 100 == 0:
                print (f"ppl: {ppl}, loss: {loss}, precision: {precision}, recall: {recall}, token accuracy: {accuracy}, molecule accuracy: {molecule_accuracy}, reaction accuracy: {reaction_accuracy}")
                #print (f"Step: {step}. PPL: {ppl}. loss: {loss}. Token Accuracy: {token_accuracy}")
                sys.stdout.flush()
            step += 1
        precision, recall, token_accuracy, molecule_accuracy, reaction_accuracy, ppl, loss = evaluate(val_dataloader, tokenizer, ctx, teacher, model, args.max_new_tokens, epoch, args.save_model, args.generate_onebatch, split=f'valid_layer{args.layer}', layer=args.layer)
        if ppl < best_ppl:
            print ('best val ppl')
            best_ppl = ppl
            print (f"Val. PPL: {ppl}; loss: {loss}, precision: {precision}, recall: {recall}, token accuracy: {accuracy}, molecule accuracy: {molecule_accuracy}, reaction accuracy: {reaction_accuracy} (best)")
        else:
            print (f"Val. PPL: {ppl}; loss: {loss}, precision: {precision}, recall: {recall}, token accuracy: {accuracy}, molecule accuracy: {molecule_accuracy}, reaction accuracy: {reaction_accuracy}")
        precision, recall, token_accuracy, molecule_accuracy, reaction_accuracy, ppl, loss = evaluate(test_dataloader, tokenizer, ctx, teacher, model, args.max_new_tokens, epoch, args.save_model, args.generate_onebatch, split=f'test_layer{args.layer}', layer=args.layer)
        print (f"Test. PPL: {ppl}; loss: {loss}, precision: {precision}, recall: {recall}, token accuracy: {accuracy}, molecule accuracy: {molecule_accuracy}, reaction accuracy: {reaction_accuracy}")
        #teacher.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))
        torch.save(model, os.path.join(args.save_model, f'probe_checkpoint_{epoch}'))

if __name__ == "__main__":
    main()
