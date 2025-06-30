from transformers import GPT2Tokenizer, GPT2Model
import torch
import nethook
import json
from tqdm import tqdm
import linecache
from torch.utils.data import Dataset, DataLoader

class CounterFactDataset(Dataset):
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as jsonl_file:
          lines = jsonl_file.readlines()
        self.data = [json.loads(line) for line in lines]
        # self.tokenizer=tokenizer
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        item = self.data[idx]
        return item
    

def get_model(PATH: str,device ):

    # Model and tokenizer
    # model_name = "google/t5-small-ssm-nq"
    if(PATH is None):
        model = GPT2Model.from_pretrained("gpt2-xl")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    else:
        model = GPT2Model.from_pretrained(
            pretrained_model_name_or_path=PATH, 
            local_files_only=True,
            use_safetensors=False
        )
        tokenizer = GPT2Tokenizer.from_pretrained(
        pretrained_model_name_or_path=PATH, 
        local_files_only=True,
        use_safetensors=False
        )
    
    model.to(device)
    return model,tokenizer

def get_embedding_dataset_avg_token(file_path,model,tokenizer,dataset,file_save_path,device):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    counter=1
    # 
    l=["h.2.mlp.act"]
    # with open(file_path, 'r') as jsonl_file_reader:
    with open(file_save_path, 'w') as jsonl_file_writer:
        with nethook.TraceDict(model, l) as ret:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Processing batches", leave=False):
                    data_entry = json.loads(linecache.getline(file_path, counter).strip())
                    # print(data_entry.keys())
                    torch.cuda.empty_cache()
                    # print(data_entry["edited_prompt"])
                    data_entry["vector_edited_prompt"]=[]
                    inputs=tokenizer(data_entry["edited_prompt"][0], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    # print([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0].shape)
                    
                    data_entry["vector_edited_prompt"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                    torch.cuda.empty_cache()
                    data_entry["vector_edited_prompt_paraphrases_processed"]=[]
                    inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed"], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    data_entry["vector_edited_prompt_paraphrases_processed"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]
                    
                    data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[]
                    torch.cuda.empty_cache()
                    inputs=tokenizer(data_entry["edited_prompt_paraphrases_processed_testing"], return_tensors="pt")["input_ids"].to(device)    
                    outputs = model(inputs, output_hidden_states=True)
                    data_entry["vector_edited_prompt_paraphrases_processed_testing"]=[ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0]

                    data_entry["vectors_neighborhood_prompts_train"]=[]
                    for string in data_entry["neighborhood_prompts_train"]:
                        torch.cuda.empty_cache()
                        inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
                        outputs = model(inputs, output_hidden_states=True)
                        data_entry["vectors_neighborhood_prompts_train"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])

                    data_entry["vectors_neighborhood_prompts_test"]=[]
                    for string in data_entry["neighborhood_prompts_test"]:
                        torch.cuda.empty_cache()
                        inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)    
                        outputs = model(inputs, output_hidden_states=True)
                        data_entry["vectors_neighborhood_prompts_test"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])

                    for string in data_entry["openai_usable_paraphrases"]:
                        torch.cuda.empty_cache()
                        inputs=tokenizer(string, return_tensors="pt")["input_ids"].to(device)
                        outputs = model(inputs, output_hidden_states=True)
                        if( "openai_usable_paraphrases_embeddings" not in data_entry.keys()):
                            data_entry["openai_usable_paraphrases_embeddings"]=[]
                        data_entry["openai_usable_paraphrases_embeddings"].append([ret[layer_fc1_vals].output.mean(dim=1).detach().cpu().numpy().tolist() for layer_fc1_vals in ret][0])
                    counter+=1
                    json.dump(data_entry, jsonl_file_writer)
                    jsonl_file_writer.write('\n')

if __name__ == '__main__':
    
    device = torch.device("cpu")
    print(device)
    model_path=None# change if local path is needed
    model,tokenizer=get_model(model_path,device)
    print(model)
  
    # file_path="/Users/hammadrizwan/Documents/Model_Editing2/counterfact_dataset_openai_5000_set_"+str(index)+ ".jsonl"
    file_path=""# path to the counterfact dataset
    counterfact_dataset=CounterFactDataset(file_path)

    file_save_path=""# 
    get_embedding_dataset_avg_token(file_path,model,tokenizer,counterfact_dataset,file_save_path,device)
    
  