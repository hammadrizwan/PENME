from imports import json,os, random, np, torch, linecache
import trainer
import data_processing_counterfact as dpcounter
import data_processing_zsre as dpzsre
import models
import results_processing as rp
import helper_functions as hp
import sys
import json
import linecache
import random
from argparsers import counterfact_llama, counterfact_gpt2xl, counterfact_t5, zsre_t5, zsre_gpt2xl, zsre_llama
import argparse
import importlib
np.random.seed(42)# fix seed
random.seed(42)
import os
from datetime import datetime

def read_dataset_reduced_split(file_path_read_dataset: str, data_size_train=2000,data_size_test=2000):
    total_samples = 5000
    indices = list(range(1, total_samples + 1))

    # Randomly shuffle indices to get two non-overlapping sets
    # random.shuffle(indices)
    first_indices = indices[:data_size_train]
    second_indices = indices[data_size_train:data_size_train+data_size_test]

    dataset1 = []
    dataset2 = []

    # Retrieve data for the first dataset
    for index, number in enumerate(first_indices):
        try:
            data_entry = json.loads(linecache.getline(file_path_read_dataset, number).strip())
            dataset1.append(data_entry)
        except Exception as e:
            print(f"Error at index {number} in dataset1: {e}")
    print("load data 2")
    # Retrieve data for the second dataset
    for index, number in enumerate(second_indices):
        try:
            data_entry = json.loads(linecache.getline(file_path_read_dataset, number).strip())
            dataset2.append(data_entry)
        except Exception as e:
            print(f"Error at index {number} in dataset2: {e}")

    return dataset1, dataset2


def read_dataset_reduced(file_path_read_dataset: str,data_size):
    dataset=[]
    values_list = list(range(1, data_size+1))
    for index,number in enumerate(values_list):
        
        # print(json.loads(linecache.getline(file_path_read_dataset, number)))
        # print(linecache.getline(file_path_read_dataset, number).strip())
        try:
            data_entry = json.loads(linecache.getline(file_path_read_dataset, number).strip())
            dataset.append(data_entry)
        except Exception as e:
            print(index)
            print(e)
    return dataset

def read_dataset(file_path: str,data_size):
    dataset=[]
    random_numbers=random.sample(range(1, 21000), data_size) 
    for number in random_numbers:
        data_entry = json.loads(linecache.getline(file_path, number).strip())
        dataset.append(data_entry)
    return dataset

def read_dataset_zsre(file_path):
    # Open the file and read its content
    dataset=[]
    with open(file_path, 'r') as file:
        # Read each line and parse it as JSON
        for line in file:
            data_row=json.loads(line.strip())
            dataset.append(data_row)

    return dataset


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def calculate_class_ratios(targets, predictions):
    """
    Calculate the ratio of correct classification for each class.

    Args:
    - targets: Array-like, true class labels (ground truth).
    - predictions: Array-like, predicted class labels.

    Returns:
    - class_ratios: Dictionary containing the ratio of correct classification for each class.
    """
    # Convert targets and predictions to numpy arrays if they are not already
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    # Calculate class-wise correct predictions for class 0 and class 1
    correct_predictions_0 = np.logical_and(predictions == 0, targets == 0)
    correct_predictions_1 = np.logical_and(predictions == 1, targets == 1)
    
    # Calculate total samples for each class
    class_counts_0 = np.sum(targets == 0)
    class_counts_1 = np.sum(targets == 1)
    
    # Calculate correct classification ratio for each class
    class_ratio_0 = np.sum(correct_predictions_0) / class_counts_0 if class_counts_0 != 0 else 0
    class_ratio_1 = np.sum(correct_predictions_1) / class_counts_1 if class_counts_1 != 0 else 0
    
    return {'locality': class_ratio_0, 'paraphrase_sucess': class_ratio_1}

def generate_results_zsre(model, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test,path_to_folder,mode,comparison,device,loss_function,threshold,args):
    device="cuda"
    print("writing results indivisual matching")
    file_path=file_path=path_to_folder+"/results_vectorlist_indivisual_threshold_testset"+str(threshold)+".jsonl"
    predictions,targets=rp.write_results_indivisual_threshold_zsre(edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test,model,comparison,mode,file_path,threshold,args,device,loss_function)
    print(len(predictions),len(targets))
    heat_map_file=path_to_folder+'ConfusionMatrix_Test_Indivisual_Threshold_'+loss_function+"_"+str(threshold)+".png"
    classification_report_file=path_to_folder+'Classificaiton_Report_Test_Indivisual_Threshold_'+loss_function+"_"+str(threshold)+".png"
    _=hp.write_classificaiton_report(targets,predictions,heat_map_file,classification_report_file)
    print("writing results indivisual matching completed")

    edit_success=100# since we are storing the vectors for the edits the edit sucess would be 1 by default
    ratios=calculate_class_ratios(targets,predictions)
    paraphrase_success=ratios["paraphrase_sucess"]
    locality=ratios["locality"]

    content = f"edit_success={edit_success}\n"
    content += f"paraphrase_success={paraphrase_success}\n"
    content += f"locality={locality}\n"
    locality_paraphrase_file=path_to_folder+'results'+str(threshold)+'.txt'
    print(paraphrase_success,locality)
    with open(locality_paraphrase_file, "w") as file:
        file.write(content)


def generate_results_counterfact(model,openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test,path_to_folder,mode,comparison,device,loss_function,threshold,args):
    device="cuda"
    print("starting evaluation")
    file_path=file_path=path_to_folder+"/results_vectorlist_indivisual_threshold_testset"+str(threshold)+".jsonl"
    predictions,targets=rp.write_results_indivisual_threshold(openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test,model,comparison,mode,file_path,threshold,args,device)
    print(len(predictions),len(targets))
    heat_map_file=path_to_folder+'ConfusionMatrix_Test_Indivisual_Threshold_'+loss_function+"_"+str(threshold)+".png"
    classification_report_file=path_to_folder+'Classificaiton_Report_Test_Indivisual_Threshold_'+loss_function+"_"+str(threshold)+".png"
    _=hp.write_classificaiton_report(targets,predictions,heat_map_file,classification_report_file)
    print("writing results indivisual matching completed")

    edit_success=100# since we are storing the vectors for the edits the edit sucess would be 1 by default
    ratios=calculate_class_ratios(targets,predictions)
    paraphrase_success=ratios["paraphrase_sucess"]
    locality=ratios["locality"]

    content = f"edit_success={edit_success}\n"
    content += f"paraphrase_success={paraphrase_success}\n"
    content += f"locality={locality}\n"
    locality_paraphrase_file=path_to_folder+'results'+str(threshold)+'.txt'
    print(f"paraphrase sucess:{paraphrase_success} locality:{locality}")
    with open(locality_paraphrase_file, "w") as file:
        file.write(content)


import argparse
import sys

def get_mode_parser():
    """
    Parses --eval_type and checks if --evaluation_mode is present and valid.
    For 'zsre_*' eval_type, allowed evaluation_modes are: 'batch', 'transferlearning'.
    For 'counterfact_*' eval_type, allowed evaluation_modes are: 'stream', 'batch'.
    """

    parser = argparse.ArgumentParser(description="Main Dispatcher", add_help=False)
    parser.add_argument(
        '--eval_type', 
        type=str, 
        required=True, 
        choices=[
            'counterfact_llama', 
            "counterfact_gpt2xl", 
            "counterfact_t5", 
            "zsre_t5", 
            "zsre_gpt2xl", 
            "zsre_llama"
        ]
    )

    # Parse only known args to preserve --evaluation_mode
    args, remaining_argv = parser.parse_known_args()

    # Check if --evaluation_mode is present and extract its value
    if '--evaluation_mode' not in remaining_argv:
        print("ERROR: --evaluation_mode is required.", file=sys.stderr)
        sys.exit(1)

    try:
        mode_index = remaining_argv.index('--evaluation_mode')
        eval_mode = remaining_argv[mode_index + 1]
    except (ValueError, IndexError):
        print("ERROR: --evaluation_mode must be followed by a value.", file=sys.stderr)
        sys.exit(1)

    # Define allowed evaluation modes for each category
    if args.eval_type.startswith("zsre"):
        allowed_modes = ['batch', 'transferlearning']
    elif args.eval_type.startswith("counterfact"):
        allowed_modes = ['stream', 'batch']
    else:
        print(f"ERROR: Unrecognized eval_type: {args.eval_type}", file=sys.stderr)
        sys.exit(1)

    if eval_mode not in allowed_modes:
        print(f"ERROR: Invalid evaluation_mode '{eval_mode}' for eval_type '{args.eval_type}'. "
              f"Allowed values: {allowed_modes}", file=sys.stderr)
        sys.exit(1)

    return args.eval_type, remaining_argv

def editing_counterfact(remaining_argv, parser_module):
    args=parser_module.parse_args(remaining_argv)
    print("parser_module", args)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("current_dir", current_dir)
    print("overall path",current_dir+"/"+args.data_file)
    file_path_dataset=current_dir+"/"+args.data_file
    print("args.data_file",args.data_file.split("/")[-1])
    dataset_fileset=[args.data_file.split("/")[-1]]


    control=0#version of datasplit to be used, 0 based on high sim, 1 on low sim and 3 on random
    label_reversal=False# no longer required cost functions updated
    loss="contrastive" # cosine, cosine_crossentropy, contrastive, triplet
    print("Loading data Counterfact")

    data_size=2000
    dataset_train, dataset_test = read_dataset_reduced_split(file_path_dataset,data_size_train=data_size,data_size_test=data_size) 
    print("Loading data completed",len(dataset_test))


    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_to_folder="./results/"+loss+"_mode_file_"+str(eval_type)+"_"+str(timestamp_str)+"/"       
    create_folder(path_to_folder)
    path_to_data="./results/"+loss+"_mode_file_"+str(eval_type)+"_"+str(timestamp_str)+"/data.jsonl"
    
     
    num_epochs=args.epochs# The code snippet provided seems to be setting the
    path_to_model=current_dir+"/"+args.pretrained_proj_path
    if os.path.isfile(path_to_model) and args.force_train==False:
        print("Loading pretrained model from", path_to_model)
        if(args.evaluation_mode=="stream"):
            openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test=dpcounter.create_dataset_pairs(dataset_test,control,label_reversal,eval_type)#
            threshold=args.similatiry_thresh_codebook_stream
        else:
            openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test=dpcounter.create_dataset_pairs(dataset_train,control,label_reversal,eval_type)#
            threshold=args.similatiry_thresh_codebook_batch
        model=models.SiameseNetwork(args.in_features ,args.out_features)#load a random initialized projector network
        print(model)
        #load pretrained and evaluate
        state_dict = torch.load(path_to_model)
        model.load_state_dict(state_dict)
        mode="similarity"
        comparison="dist"
        
        
        generate_results_counterfact(model,openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test,path_to_folder,mode,comparison,device,loss,threshold,args)
   
    else:
        sim_pos=args.threshold_sim_pos_pairs
        neighbors=0  #[2,3,4,5,6,7,8]   
        openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test=dpcounter.create_dataset_pairs(dataset_train,control,label_reversal,eval_type)#
        dataset_paired_train_aug,neighbourhood_train_vectors_dict=dpcounter.data_construct_high_sim(openai_vectors_dict=openai_vectors_dict, edit_vectors_dict=edit_vectors_dict, neighbourhood_train_vectors_dict=neighbourhood_train_vectors_dict,
                                                                            paraphrase_train_vectors_dict=paraphrase_train_vectors_dict, dataset_paired_train=dataset_paired_train,
                                                                            neightbour_control=0,label_reversal=label_reversal,comparison="sim",topk_neg=neighbors,pos_sim=sim_pos,
                                                                            loss=loss,device=device)
        early_stop_patience=args.patience
        margin_loss=args.margin_loss
      
        model,mode,comparison,file_path=trainer.train_control_counterfact(openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, paraphrase_train_vectors_dict, dataset_paired_train,args,path_to_folder=path_to_folder,cls_weights=False,device=device)
        
        if(args.evaluation_mode=="stream"):
            openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test=dpcounter.create_dataset_pairs(dataset_test,control,label_reversal,eval_type)#
        else:
            openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test=dpcounter.create_dataset_pairs(dataset_train,control,label_reversal,eval_type)#
        
        print("Result Generation Start")
        threshold=args.similatiry_thresh_codebook_batch
        generate_results_counterfact(model,openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test,path_to_folder,mode,comparison,device,loss,threshold,args)
        print("Result Generation Completed")

def editing_zsre(remaining_argv, parser_module):
    
    args=parser_module.parse_args(remaining_argv)
    print("parser_module", args)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("current_dir", current_dir)
    print("overall path",current_dir+"/"+args.data_file)
    file_path_dataset=current_dir+"/"+args.data_file
    print("args.data_file",args.data_file.split("/")[-1])
    print("file_path_dataset",file_path_dataset)
    data_zsre=read_dataset_zsre(file_path_dataset) 
    

    control=0#version of datasplit to be used, 0 based on high sim, 1 on low sim and 3 on random 
    label_reversal=False# no longer required cost functions updated
    loss="contrastive" # cosine, cosine_crossentropy, contrastive, triplet
    # print("Loading data")
    print("Loading data completed",len(data_zsre))
    high_sim_neighbours_list=[0]

    neighbors=0  #[2,3,4,5,6,7,8]



    # for n in high_sim_neighbours_list:
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_to_folder="./results/"+loss+"_mode_file_"+str(eval_type)+"_"+str(timestamp_str)+"/"       
    create_folder(path_to_folder)
    path_to_data="./results/"+loss+"_mode_file_"+str(eval_type)+"_"+str(timestamp_str)+"/data.jsonl"
    

    if(args.evaluation_mode=="batch"):
        path_to_model=current_dir+"/"+args.pretrained_proj_path
        threshold=args.similatiry_thresh_codebook_batch
    else:
        path_to_model=current_dir+"/"+args.pretrained_proj_path_transferlearning
        threshold=args.similatiry_thresh_codebook_transferlearning
        
    if os.path.isfile(path_to_model) and args.force_train==False:
        print("Loading pretrained model from", path_to_model)
        edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test=dpzsre.create_dataset_pairs(data_zsre,control,label_reversal,eval_type)#
        
        model=models.SiameseNetwork(args.in_features ,args.out_features)#load a random initialized projector network
        state_dict = torch.load(path_to_model)
        model.load_state_dict(state_dict)
        mode="similarity"
        comparison="dist"
        
        generate_results_zsre(model, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test,path_to_folder,mode,comparison,device,loss,threshold,args)
   
    else:
        sim_pos=args.threshold_sim_pos_pairs
        neighbors=0  #[2,3,4,5,6,7,8]   
        print("TRAINING PATH")
        
        edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test=dpzsre.create_dataset_pairs(data_zsre,control,label_reversal,eval_type)#


        dataset_paired_train_aug,neighbourhood_train_vectors_dict=dpzsre.data_construct_high_sim(edit_vectors_dict=edit_vectors_dict, neighbourhood_train_vectors_dict=neighbourhood_train_vectors_dict,
                                                                            paraphrase_train_vectors_dict=paraphrase_train_vectors_dict, dataset_paired_train=dataset_paired_train,
                                                                            neightbour_control=0,label_reversal=label_reversal,comparison="sim",topk_neg=neighbors,pos_sim=sim_pos,
                                                                            loss=loss,device=device)
        dataset_paired_train=dataset_paired_train+dataset_paired_train_aug
        model,mode,comparison,file_path=trainer.train_control_zsre(edit_vectors_dict, neighbourhood_train_vectors_dict, paraphrase_train_vectors_dict, dataset_paired_train,args,path_to_folder=path_to_folder,
             cls_weights=False,device=device)#cosine, cosine_crossentropy, contrastive
       
        print("Result Generation Start")
        threshold=args.similatiry_thresh_codebook_batch
        generate_results_zsre(model, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test,path_to_folder,mode,comparison,device,loss,threshold,args)
        print("Result Generation Completed")

if __name__ == "__main__":
    eval_type, remaining_argv = get_mode_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device)
    print("eval_type", eval_type)
    try:
        parser_module = importlib.import_module(f"argparsers.{eval_type}")
    except ModuleNotFoundError:
        raise ValueError(f"No parser module found for mode '{eval_type}'")
    
    if("counterfact" in eval_type):
        editing_counterfact(remaining_argv,parser_module)
    else:
        editing_zsre(remaining_argv,parser_module)