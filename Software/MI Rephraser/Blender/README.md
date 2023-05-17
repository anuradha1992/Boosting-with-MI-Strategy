To train Blender-based rephraser models, you need to clone and install the ParlAI repository from: [https://github.com/facebookresearch/ParlAI](https://github.com/facebookresearch/ParlAI). We used the development installation for our experiments.

To train the different rephraser models using Blender, use the following scripts (The scripts indicated are for training on the template-based PP corpus and testing on the combined PP corpus):

# Building the dictionary before fine-tuning
$ python3 parlai/scripts/build_dict.py --task fromfile:parlaiformat --fromfile-datapath training_data/PP/template/advise --fromfile-datatype-extension true --dict-file dict/PP/template.dict

# Script for fine-tuning
$ python3 parlai/scripts/multiprocessing_train.py --model transformer/generator --task fromfile:parlaiformat --fromfile-datapath training_data/PP/template/advise --fromfile-datatype-extension true --multitask-weights 1,3,3,3 --num-epochs 200 --init-model zoo:blender/blender_90M/model --dict-file dict/PP/template.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --force-fp16-tokens true --text-truncate 100 --label-truncate 100 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 8 -vmt ppl -vmm min --save-after-valid True --model-file models/PP/blender_finetune_template/blender_finetune_template --load-from-checkpoint True --eval-batchsize 8 --dynamic-batching full --truncate 100

# Display model predictions on the testset
$ python3 parlai/scripts/display_model.py --task fromfile:parlaiformat --fromfile-datapath training_data/PP/template/advise --fromfile-datatype-extension true --model-file models/PP/blender_finetune_template/blender_finetune_template --datatype test --skip-generation False --num-examples 10 --beam-min-length 20 --beam-block-ngram 3 --beam-context-block-ngram 3 --beam-size 10 --inference beam

# Write model predictions into a csv file
$ python3 display_model.py --task fromfile:parlaiformat --fromfile-datapath training_data/PP/template_retrieve/advise --fromfile-datatype-extension true --model-file models/PP/blender_finetune_template/blender_finetune_template --datatype test --skip-generation False --num-examples 200 --output-file output/train_PP_test_PP/template.csv --beam-min-length 20 --beam-block-ngram 3 --beam-context-block-ngram 3 --beam-size 10 --inference beam

# Evaluating the model
$ python3 parlai/scripts/eval_model.py --task fromfile:parlaiformat --fromfile-datapath training_data/PP/template_retrieve/advise --fromfile-datatype-extension true --model-file models/PP/blender_finetune_template/blender_finetune_template --report-filename eval/PP/template_report.csv --world-logs eval/PP/template_world_logs.csv --num-examples 200 --display-examples True --metrics ppl,rouge,bleu --datatype test --skip-generation False --beam-min-length 20 --beam-block-ngram 3 --beam-context-block-ngram 3 --beam-size 10 --inference beam