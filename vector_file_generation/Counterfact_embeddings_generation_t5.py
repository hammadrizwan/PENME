from transformers import T5Tokenizer, T5Model,T5ForConditionalGeneration
import torch
import nethook
import json
from tqdm import tqdm
import linecache
from torch.utils.data import Dataset, DataLoader

class CounterFactDataset(Dataset):
    def __init__(self, json_file_path,tokenizer):
        with open(json_file_path, 'r') as jsonl_file:
          lines = jsonl_file.readlines()
        self.data = [json.loads(line) for line in lines]
        self.tokenizer=tokenizer
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        item = self.data[idx]

        return item
    

def get_model(PATH: str,device):

    if(PATH is None):
        model = T5Model.from_pretrained("google/t5-small-ssm-nq")
        tokenizer = T5Tokenizer.from_pretrained("google/t5-small-ssm-nq")
    else:
        model = T5Model.from_pretrained(PATH, local_files_only=True)
        tokenizer = T5Tokenizer.from_pretrained(PATH, local_files_only=True)
    
    model.to(device)
    return model,tokenizer

                    
                    
def get_embedding_dataset_special_token(file_path,model,tokenizer,dataset,file_save_path,device):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    counter=1
    l=["encoder.block.2.layer.1.DenseReluDense.wo"]
    # with open(file_path, 'r') as jsonl_file_reader:
    with open(file_save_path, 'w') as jsonl_file_writer:
        with nethook.TraceDict(model, l) as ret:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Processing batches", leave=False):
                    data_entry = json.loads(linecache.getline(file_path, counter).strip())
                    data_entry["vector_edited_prompt"]=[]
                    inputs=tokenizer(data_entry["edited_prompt"][0], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, decoder_input_ids=inputs, output_hidden_states=True)
                   
                    data_entry["vector_edited_prompt"]=[ret[layer_fc1_vals].output[:,-1, :] for layer_fc1_vals in ret][0][0].detach().cpu().numpy().tolist()
                    torch.cuda.empty_cache()
                    data_entry["vector_edited_prompt_paraphrases_processed"]=[]
                    inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed"], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, decoder_input_ids=inputs, output_hidden_states=True)
                    data_entry["vector_edited_prompt_paraphrases_processed"]=[ret[layer_fc1_vals].output[:,-1, :] for layer_fc1_vals in ret][0][0].detach().cpu().numpy().tolist()
                    
                    data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[]
                    torch.cuda.empty_cache()
                    inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed_testing"], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, decoder_input_ids=inputs, output_hidden_states=True)
                    data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[ret[layer_fc1_vals].output[:,-1, :] for layer_fc1_vals in ret][0][0].detach().cpu().numpy().tolist()

                    data_entry["vectors_neighborhood_prompts_train"]=[]
                    for string in data_entry["neighborhood_prompts_train"]:
                        torch.cuda.empty_cache()
                        inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
                        outputs = model(inputs, decoder_input_ids=inputs, output_hidden_states=True)
                        data_entry["vectors_neighborhood_prompts_train"].append([ret[layer_fc1_vals].output[:,-1, :] for layer_fc1_vals in ret][0][0].detach().cpu().numpy().tolist())

                    data_entry["vectors_neighborhood_prompts_test"]=[]
                    for string in data_entry["neighborhood_prompts_test"]:
                        torch.cuda.empty_cache()
                        inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
                        outputs = model(inputs, decoder_input_ids=inputs, output_hidden_states=True)
                        data_entry["vectors_neighborhood_prompts_test"].append([ret[layer_fc1_vals].output[:,-1, :] for layer_fc1_vals in ret][0][0].detach().cpu().numpy().tolist())

                    for string in data_entry["openai_usable_paraphrases"]:
                        torch.cuda.empty_cache()
                        inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)
                        outputs = model(inputs, decoder_input_ids=inputs, output_hidden_states=True)
                        if( "openai_usable_paraphrases_embeddings" not in data_entry.keys()):
                            data_entry["openai_usable_paraphrases_embeddings"]=[]
                        data_entry["openai_usable_paraphrases_embeddings"].append([ret[layer_fc1_vals].output[:,-1, :] for layer_fc1_vals in ret][0][0].detach().cpu().numpy().tolist())
                    counter+=1
                    json.dump(data_entry, jsonl_file_writer)
                    jsonl_file_writer.write('\n')
                    print(counter)
                 

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path=None #change if local path is needed
    model,tokenizer=get_model(model_path,device)
    # print(model)
    file_path=""#path to the counterfact dataset
    counterfact_dataset=CounterFactDataset(file_path,tokenizer)
    file_save_path=""#save path for counterfact dataset with embeddings
    get_embedding_dataset_special_token(file_path,model,tokenizer,counterfact_dataset,file_save_path,device)
    
