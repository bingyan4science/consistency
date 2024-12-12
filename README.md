# consistency

# Inconsistency of LLMs

This repo provides code for "Inconsistency of Large Language Models In Molecular Representations" by Bing Yan, Angelica Chen, and Kyunghyun Cho. Note that this repo can be found at https://github.com/bingyan4science/consistency.

## Dependencies

The code has been tested on Python 3.10 and PyTorch 2.1.0.

* Python 3.10.13: `conda create --prefix ../conda_consist python=3.10`
* PyTorch: https://pytorch.org/get-started/locally/ `conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia`

## Finetuning experiments: Train a forward reaction prediction model based on GPT-2 Small

For the finetuning, we train a forward reaction prediction model by finetuning the GPT-2 Small model on one-to-one mapped SMILES vs IUPAC input representations, and equal chance of generating SMILES and IUPAC outputs:

To finetunewithout KL divergence loss
```
export MODE_A=smiles
export MODE_B=iupac
export FOLDER_A=data/reaction_${MODE_A}_80k
export FOLDER_B=data/reaction_${MODE_B}_80k
export MODEL=gpt2
export EPOCHS=20
export LR=1e-4
export WEIGHT=1.0
export NUM_SAMPLES=1
export BSZ=32
export PRETRAIN_EPOCHS=20
export SAVE=train_models_nokl
echo $SAVE
mkdir -p ${SAVE}/${MODE_A}
mkdir -p ${SAVE}/${MODE_B}
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICESNOTUSED=0 stdbuf -oL -eL python src/train.py \
    --train_path_a ${FOLDER_A}/train.txt \
    --val_path_a ${FOLDER_A}/valid.txt \
    --train_path_b ${FOLDER_B}/train.txt \
    --val_path_b ${FOLDER_B}/valid.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --weight $WEIGHT \
    --num_samples $NUM_SAMPLES \
    --base_model $MODEL \
    --batch_size $BSZ \
    --pretrain_epochs ${PRETRAIN_EPOCHS} \
    --save_model_a ${SAVE}/${MODE_A} \
    --save_model_b ${SAVE}/${MODE_B} \
    --generate_onebatch \
    > ${SAVE}/log.train 2>&1&

To finetune with KL divergence loss, change ''PRETRAIN_EPOCHS=1''.

Next, we test the trained model.
Taking the model trained without KL divergence loss as an example:
```
# SMILES representation for inputs
export MODE=smiles
export FOLDER=data/reaction_${MODE}_80k
export MODEL=gpt2
export EPOCHS=20
export LR=1e-4
export SAVE=train_models_nokl/${MODE}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/evaluate.py \
    --data_folder ${FOLDER} \
    --epochs $EPOCHS \
    --base_model $MODEL \
    --save_model $SAVE \
    > ${SAVE}/log.gen 2>&1&

# IUPAC representation for inputs
export MODE=iupac
export FOLDER=data/reaction_${MODE}_80k
export MODEL=gpt2
export EPOCHS=20
export LR=1e-4
export SAVE=train_models_nokl/${MODE}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/evaluate.py \
    --data_folder ${FOLDER} \
    --epochs $EPOCHS \
    --base_model $MODEL \
    --save_model $SAVE \
    > ${SAVE}/log.gen 2>&1&
```

Then we evaluate the prediction and calculate the consistency between SMILES and IUPAC inputs.

```
python calc_consistency.py
```
