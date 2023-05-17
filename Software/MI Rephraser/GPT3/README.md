To train GPT-3 rephraser models, you need to use the paid api by OpenAI [https://openai.com/](https://openai.com/), following their API: [https://beta.openai.com/docs/guides/fine-tuning](https://beta.openai.com/docs/guides/fine-tuning). We used GPT3's smallest but fastest model 'Ada' in all our experiments.

To train the different rephraser models using GPT3, use the following scripts (The scripts indicated are for training on the template-based PP corpus and testing on the combined PP corpus):

# Script for fine-tuning
$ openai api fine_tunes.create -t ./training_data/PP/template/advise_train.jsonl -v ./training_data/PP/template/advise_valid.jsonl -m ada --n_epochs 4 --suffix advise_PP_template

# Display model predictions for a given prompt
$ openai api completions.create -m <MODEL_ID> -p <YOUR_PROMPT>

# Write model predictions into a csv file
For this you can use the output.py python script included in this folder

# Evaluating the model
$ openai api fine_tunes.results -i <MODEL_ID> > ./eval/PP/template.csv