import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList, AutoModelForSeq2SeqLM

from .configuration import Config
import sys
sys.path.append("..")
from utils import get_sep_position, DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.base_model == "Salesforce/codet5-small":
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)
        elif config.base_model == "mistralai/Mistral-7B-v0.1":
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                device_map="auto",
                torch_dtype=torch.float32,
                load_in_4bit=True)
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model)
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def forward(self, input_ids, decoder_input_ids=None):
        if self.config.base_model == "Salesforce/codet5-small":
            outputs = self.base_model.forward(input_ids=input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=False)
        else:
            outputs = self.base_model.forward(input_ids=input_ids, output_hidden_states=False)
        return outputs

    def compute_loss(self, input_ids, labels, logits_only=False, validation=False):
        if self.config.base_model == "Salesforce/codet5-small":
            #import ipdb; ipdb.set_trace()
            shifted = labels.new_zeros(labels.shape)
            eos_mask = labels.ne(-100)
            labels_masked = labels.masked_fill(~eos_mask, self.tokenizer.pad_token_id)
            shifted[:, 1:] = labels_masked[:, :-1]
            shifted[:, 0] = self.tokenizer.pad_token_id
            outputs = self.forward(input_ids=input_ids, decoder_input_ids=shifted)
            del shifted, eos_mask, labels_masked
            logits = outputs.logits
            labels_pred = logits.argmax(-1)
            mask = labels.ge(0)
            correct_tokens = ((labels_pred == labels) * mask).sum()
            total_tokens = mask.sum()
            token_accuracy = correct_tokens / total_tokens
            if logits_only:
                outputs.mask = mask
                outputs.logits = logits
                return outputs
            
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            if not validation:
                outputs.loss = loss
                outputs.token_accuracy = token_accuracy
                return outputs
            
            outputs.total_correct = correct_tokens
            outputs.total_loss = loss * total_tokens
            outputs.total_tokens = total_tokens
            
            return outputs

        else:
            outputs = self.forward(input_ids=input_ids)
            logits = outputs.logits
            labels_pred = logits.argmax(-1)
            mask = labels[...,1:].ge(0)
            correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
            total_tokens = mask.sum()
            token_accuracy = correct_tokens / total_tokens
            shift_logits = logits[..., :-1, :].contiguous()

            if logits_only:
                outputs.mask = mask
                outputs.logits = shift_logits
                return outputs

            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if not validation:
                outputs.loss = loss
                outputs.token_accuracy = token_accuracy
                return outputs
            
            outputs.total_correct = correct_tokens
            outputs.total_loss = loss * total_tokens
            outputs.total_tokens = total_tokens
            
            return outputs

    def generate(self, input_ids, max_new_tokens=512, num_beams=1, stop_on_two_eos=True, num_return_sequences=1):
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        batch_size = input_ids.shape[0]

        # Since there's one eos after CoT and another after final answer, we need to wait for two eos
        generation_config = GenerationConfig.from_model_config(self.base_model.config)
        if stop_on_two_eos:
            generation_config.eos_token_id = -1
            logits_processor = LogitsProcessorList([DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)])
            stopping_criteria = StoppingCriteriaList([DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)])
        else:
            logits_processor = None
            stopping_criteria = None
        
        if self.config.base_model == "Salesforce/codet5-small":
            attention_mask_tensor = input_ids.data.clone()
            for i in range(batch_size):
                input_ids_i = input_ids[i]
                end_idx = len(input_ids_i)-1
                while input_ids_i[end_idx] == self.tokenizer.eos_token_id:
                    end_idx -= 1
                attention_mask_tensor[i, :end_idx+2] = 1
                attention_mask_tensor[i, end_idx+2:] = 0
            
            beam_output = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask_tensor,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                do_sample=True,
                num_return_sequences=num_return_sequences,
            )
        
        else:
            beam_output = []
            input_ids_all = []
            for i in range(batch_size):
                if stop_on_two_eos:
                    generation_config.eos_token_id = -1
                    logits_processor = LogitsProcessorList([DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)])
                    stopping_criteria = StoppingCriteriaList([DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)])

                input_ids_i = input_ids[i]
                end_idx = len(input_ids_i)-1
                while input_ids_i[end_idx] == self.tokenizer.eos_token_id:
                    end_idx -= 1
                input_ids_i = input_ids_i[:end_idx+2]
                sep_positions_i = [ii for ii, n in enumerate(input_ids_i.view(-1)) if n == self.tokenizer.eos_token_id][-3]
                input_ids_i = input_ids_i.view(1, -1)[:, :sep_positions_i+1]
                input_ids_all.append(input_ids_i)
            max_seq_len = max([ele.shape[-1] for ele in input_ids_all])
            input_ids_tensor = torch.zeros(batch_size,max_seq_len).long().to(input_ids.device)
            attention_mask_tensor = input_ids_tensor.data.clone()
            input_ids_tensor.fill_(self.tokenizer.eos_token_id)
            for i in range(batch_size):
                pad_len = max_seq_len - input_ids_all[i].shape[-1]
                input_ids_tensor[i, pad_len:] = torch.Tensor(input_ids_all[i]).view(-1).to(input_ids.device)
                attention_mask_tensor[i, pad_len:] = 1
            
            beam_output = self.base_model.generate(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                do_sample=True,
                num_return_sequences=num_return_sequences,
            )
        
        beam_output = beam_output.view(batch_size,num_return_sequences,-1)
        beam_output_list = []
        #import ipdb; ipdb.set_trace()
        for i in range(batch_size):
            if self.config.base_model == "Salesforce/codet5-small":
                pad_pos = beam_output.shape[-1] - 1
                while beam_output[i,:,pad_pos] == self.tokenizer.eos_token_id:
                    pad_pos -= 1
                beam_output_list.append(beam_output[i,:,:pad_pos+1])
            else:
                pad_len = max_seq_len - input_ids_all[i].shape[-1]
                beam_output_list.append(beam_output[i,:,pad_len:])
        return beam_output_list

    @classmethod
    def from_pretrained(self, pretrained_path):
        config = Config.from_pretrained(pretrained_path)
        model = Model(config)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))
