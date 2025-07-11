from imports import Dataset,torch,np,random,DataLoader,util,tqdm,json
from torch.utils.data.sampler import Sampler
# DATASET CLASSES PYTORCH 
import helper_functions as hp
import itertools
import linecache

class CustomDataset(Dataset):
    def __init__(self,dataset,openai_vectors_dict, edit_vectors_dict, neighbourhood_vectors_dict, paraphrase_vectors_dict,device):
        self.dataset=np.array(dataset,dtype=object)
        self.openai_vectors_dict=openai_vectors_dict
        self.edit_vectors_dict=edit_vectors_dict
        self.neighbourhood_vectors_dict=neighbourhood_vectors_dict
        self.paraphrase_vectors_dict=paraphrase_vectors_dict
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def total_indexes(self):
        # print(self.dataset[0][2:])
        return np.unique(self.dataset[:, 3])

    def get_row_indexes(self,target_sample_index):
        return np.where(self.dataset[:, 3] == target_sample_index)[0]

    def get_samples_at_data_index(self,target_sample_index):
        row_indexes = np.where(self.dataset[:, 3] == target_sample_index)[0]
        emb1=[]
        emb2=[]
        label=[]
        row_index=[]
        sent1=[]
        sent2=[]
        for index in row_indexes:
        
          emb1.append(hp.to_tensor(self.dataset[index][0][0]))
          emb2.append(hp.to_tensor(self.dataset[index][1][0]))
          label.append(hp.to_tensor(self.dataset[index][2]))
          row_index.append(hp.to_tensor(self.dataset[index][3]))
          sent1.append(self.dataset[index][4])
          sent2.append(self.dataset[index][5])
        return emb1.to(self.device), emb2.to(self.device), label.to(self.device),row_index, sent1, sent2

    def __getitem__(self, index):
        data_row=self.dataset[index]
     
        if(data_row[-2]==0):#open ai paraphrase
            emb1 = hp.to_tensor(self.edit_vectors_dict[data_row[0]]).to(self.device)#, dtype=torch.float)
            emb2 = hp.to_tensor(self.openai_vectors_dict[data_row[0]][data_row[1]]).to(self.device)#, dtype=torch.float)
            label = hp.to_tensor(self.dataset[index][2]).to(self.device)#, dtype=torch.long)
            sample_index=self.dataset[index][3]
            sent1=self.dataset[index][4]
            sent2=self.dataset[index][5]
            pair_type=self.dataset[index][6]
            negative_sample_cntrl=self.dataset[index][7]
            emb1_index=data_row[0]
            emb2_index=data_row[1]

        elif(data_row[-2]==1):#paraphrase
            emb1 = hp.to_tensor(self.edit_vectors_dict[data_row[0]]).to(self.device)#, dtype=torch.float)
            emb2 = hp.to_tensor(self.paraphrase_vectors_dict[data_row[0]]).to(self.device)#, dtype=torch.float)
            label = hp.to_tensor(self.dataset[index][2]).to(self.device)#, dtype=torch.long)
            sample_index=self.dataset[index][3]
            sent1=self.dataset[index][4]
            sent2=self.dataset[index][5]
            pair_type=self.dataset[index][6]#neighbour,openai,paraphrase
            negative_sample_cntrl=self.dataset[index][7]
            emb1_index=data_row[0]#both should be the same
            emb2_index=data_row[1]#both should be the same

        else:#neighbour
            emb2 = hp.to_tensor(self.edit_vectors_dict[data_row[1]]).to(self.device)#, dtype=torch.float)
            # print(data_row[0],data_row[0])
            emb1 = hp.to_tensor(self.neighbourhood_vectors_dict[data_row[1]][data_row[0]]).to(self.device)#, dtype=torch.float)
            label = hp.to_tensor(self.dataset[index][2]).to(self.device)#, dtype=torch.long)
            sample_index=self.dataset[index][3]
            sent1=self.dataset[index][4]
            sent2=self.dataset[index][5]
            pair_type=self.dataset[index][6]
            negative_sample_cntrl=self.dataset[index][7]
            emb1_index=data_row[0]
            emb2_index=data_row[1]

        return emb1, emb2, label,sample_index, sent1, sent2,pair_type,emb1_index,emb2_index, negative_sample_cntrl

def get_data_loader(dataset_paired,openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, paraphrase_train_vectors_dict,batch_size=512,shuffle=True,device="cpu"):
  """
    dataset: dataset to be used
    shuffle: dataset shuffle per iteration

  """

  dataset_pt=CustomDataset(dataset_paired,openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, paraphrase_train_vectors_dict,device=device)
  data_loader=DataLoader(dataset_pt, batch_size=batch_size, shuffle=shuffle)
  return data_loader


def create_dataset_pairs(dataset,neightbour_control=0,label_reversal=False,eval_type=""):
    """
    Modes:
        0 high sim as train and low sim as test
        1 low sim as test and high sim as test
        2 random assigment
    """
    paraphrase=1
    neightbour=0


    openai_vectors_dict={}
    edit_vectors_dict={}
    neighbourhood_train_vectors_dict={}
    neighbourhood_test_vectors_dict={}
    paraphrase_train_vectors_dict={}
    paraphrase_test_vectors_dict={}

    dataset_paired_train=[]
    dataset_paired_test=[]
    for row_index,row in enumerate(dataset):
        if(eval_type=="counterfact_llama" or eval_type=="counterfact_gpt2xl"):
        
            edit_vectors_dict[row_index]=row["vector_edited_prompt"][0]
            paraphrase_train_vectors_dict[row_index]=row["vector_edited_prompt_paraphrases_processed"][0]
            paraphrase_test_vectors_dict[row_index]=row["vector_edited_prompt_paraphrases_processed_testing"][0]

            num_elements_to_select = min(3, len(row["openai_usable_paraphrases_embeddings"]))#add 5 max open ai paraphrases
            # return  None, None, None, None, None, None, None, None

            sampled_indices, sampled_elements = zip(*random.sample(list(enumerate(row["openai_usable_paraphrases_embeddings"])), num_elements_to_select))# sample and get indexes
            for index,vector in zip(sampled_indices, sampled_elements):#create postive label with edit vector
                if(row_index not in openai_vectors_dict.keys()):
                    openai_vectors_dict[row_index]={}
                openai_vectors_dict[row_index][index]=vector[0]
                # print(vector[:5],row["openai_usable_paraphrases"][index],"openai")
                dataset_paired_train.append([row_index,index,paraphrase,row_index,
                                        row["edited_prompt"][0],row["openai_usable_paraphrases"][index],0,0])


            dataset_paired_train.append([row_index,row_index,paraphrase,row_index,
                                        row["edited_prompt"][0],row["edited_prompt_paraphrases_processed"],1,1])
            # print(row["edited_prompt_paraphrases_processed"])
            dataset_paired_test.append([row_index,row_index,paraphrase,row_index,
                                        row["edited_prompt"][0],row["edited_prompt_paraphrases_processed_testing"],1,0])
            
            if(neightbour_control==0):
                for index,vector in enumerate(row["vectors_neighborhood_prompts_high_sim"]):
                    if(row_index not in neighbourhood_train_vectors_dict.keys()):
                        neighbourhood_train_vectors_dict[row_index]={}
                    neighbourhood_train_vectors_dict[row_index][index]=vector[0]   
                    dataset_paired_train.append([index,row_index,neightbour,row_index,
                                            row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index],2,0])

                 
                for index,vector in enumerate(row["vectors_neighborhood_prompts_low_sim"]):
                    if(row_index not in neighbourhood_test_vectors_dict.keys()):
                        neighbourhood_test_vectors_dict[row_index]={}
                    neighbourhood_test_vectors_dict[row_index][index]=vector[0] 
                    dataset_paired_test.append([index,row_index,neightbour,row_index,
                                            row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index],2,0])

        else:
            edit_vectors_dict[row_index]=row["vector_edited_prompt"]
            paraphrase_train_vectors_dict[row_index]=row["vector_edited_prompt_paraphrases_processed"]
            paraphrase_test_vectors_dict[row_index]=row["vector_edited_prompt_paraphrases_processed_testing"]

            num_elements_to_select = min(3, len(row["openai_usable_paraphrases_embeddings"]))#add 5 max open ai paraphrases


            sampled_indices, sampled_elements = zip(*random.sample(list(enumerate(row["openai_usable_paraphrases_embeddings"])), num_elements_to_select))# sample and get indexes
            for index,vector in zip(sampled_indices, sampled_elements):#create postive label with edit vector
                if(row_index not in openai_vectors_dict.keys()):
                    openai_vectors_dict[row_index]={}
                openai_vectors_dict[row_index][index]=vector
                # print(vector[:5],row["openai_usable_paraphrases"][index],"openai")
                dataset_paired_train.append([row_index,index,paraphrase,row_index,
                                        row["edited_prompt"][0],row["openai_usable_paraphrases"][index],0,0])


            dataset_paired_train.append([row_index,row_index,paraphrase,row_index,
                                        row["edited_prompt"][0],row["edited_prompt_paraphrases_processed"],1,1])
            # print(row["edited_prompt_paraphrases_processed"])
            dataset_paired_test.append([row_index,row_index,paraphrase,row_index,
                                        row["edited_prompt"][0],row["edited_prompt_paraphrases_processed_testing"],1,0])
            
            if(neightbour_control==0):
                for index,vector in enumerate(row["vectors_neighborhood_prompts_high_sim"]):
                    if(row_index not in neighbourhood_train_vectors_dict.keys()):
                        neighbourhood_train_vectors_dict[row_index]={}
                    neighbourhood_train_vectors_dict[row_index][index]=vector    
                    dataset_paired_train.append([index,row_index,neightbour,row_index,
                                            row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index],2,0])

                    # print(vector[:5],row["neighborhood_prompts_high_sim"][index],"high")
                for index,vector in enumerate(row["vectors_neighborhood_prompts_low_sim"]):
                    if(row_index not in neighbourhood_test_vectors_dict.keys()):
                        neighbourhood_test_vectors_dict[row_index]={}
                    neighbourhood_test_vectors_dict[row_index][index]=vector
                    dataset_paired_test.append([index,row_index,neightbour,row_index,
                                            row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index],2,0])                               

        
    return  openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test


def data_construct_high_sim(openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, paraphrase_train_vectors_dict, dataset_paired_train,neightbour_control=0,label_reversal=False,comparison="dist",topk_neg=0,pos_sim=0.65,loss="contrastive",device='cpu'):
    if(label_reversal==True):
        paraphrase=0
        neighbour=1
    else:
        paraphrase=1
        neighbour=0


    dataset_processed=[]

    vector_list_edits=[]
    vector_tensor_list_edits=[]
    edit_prompts=[]
    row_indexes_edits=[]
    data_index_edit=[]

    vector_list_neighbours=[]
    vector_tensor_list_neighbours=[]
    neighbours_prompt=[]
    neighbours_edits=[]
    data_index_edit_neighbours=[]

    data_loader=get_data_loader(dataset_paired_train,openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict,paraphrase_train_vectors_dict,batch_size=1,shuffle=False,device=device)
    
    for sample in data_loader:
        
        if(sample[2].item()==paraphrase):# only perfom for counterfact paraphrase not for openai
            if(sample[-1].item()!=1):#only add edit phrase once
                continue
            # print(sample[3],sample[-3])
            # print(sample[2:])
            vector_list_edits.append(sample[0][0].detach().cpu().numpy().tolist())
            vector_tensor_list_edits.append(sample[0][0])
            edit_prompts.append(sample[4][0])
          
            row_indexes_edits.append(sample[3][0].item())
            data_index_edit.append(sample[-3][0].item())
    
        else:
            vector_list_neighbours.append(sample[0][0].detach().cpu().numpy().tolist())
            vector_tensor_list_neighbours.append(sample[0][0])
            neighbours_prompt.append(sample[5][0])
            data_index_edit_neighbours.append(sample[-3][0].item())



    print(vector_tensor_list_neighbours[0])

    c=0
    print("topk_neg",topk_neg)
    if(topk_neg!=0):
        vectors = torch.stack(vector_tensor_list_neighbours)
        for index_vector,(target_vector, edit_index) in enumerate(zip(vector_tensor_list_edits,data_index_edit)):# for each edit construct negative pairs across the dataset
            # print(target_vector,vectors[0])
            metric = util.cos_sim(target_vector,vectors)
            # print(metric)
            top_indices = torch.topk(metric, k=topk_neg).indices
            for index in top_indices[0].cpu().numpy().tolist():
            #   print(data_index_edit_neighbours[index])
                dataset_processed.append([data_index_edit_neighbours[index],edit_index,neighbour,row_indexes_edits[index_vector],
                                                edit_prompts[edit_index],neighbours_prompt[index],2,0])


    vectors_tensor = torch.stack(vector_tensor_list_edits)
    # for index,vector in enumerate(vector_list_edits):
    print("vectors_tensor",vectors_tensor.shape)
    # Compute the magnitudes of each vector
    magnitudes = torch.norm(vectors_tensor, dim=1, keepdim=True)

    # Normalize each vector by dividing by its magnitude
    normalized_vectors = vectors_tensor / magnitudes
    # Compute pairwise cosine similarity matrix
    similarity_matrix = torch.matmul(normalized_vectors, normalized_vectors.t())

    # Fill diagonal with zeros to avoid self-similarity
    similarity_matrix.fill_diagonal_(0)
    # print("diag filled")
    # Find indices where similarity is greater than 0.80
    indices = torch.nonzero(similarity_matrix > pos_sim, as_tuple=False)
    # print(indices[:5])
    indices = [[i.item(), j.item()] for i, j in indices if i < j]  # Ensure only one direction is considered
    # print(len(indices))
    # Iterate over the indices and update dataset_processed
    for i, j in tqdm(indices):

        neighbourhood_train_vectors_dict[i][data_index_edit[j]]=vector_tensor_list_edits[j]#add to dict
        dataset_processed.append([
            data_index_edit[j],data_index_edit[i], neighbour, row_indexes_edits[i],
            edit_prompts[i], edit_prompts[j], 2,0
        ])
    
    print("Pairings complete")
    return dataset_processed,neighbourhood_train_vectors_dict
