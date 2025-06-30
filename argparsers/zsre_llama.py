import argparse
import os

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ZsRE Editing Run on Llama-2-7b")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--result_dir", type=str, default="result", help="directory of all outputs"
    )
    parser.add_argument("--dataset", type=str, default="counterfact", help="dataset")
    
    parser.add_argument(
        "--data_file",
        type=str,
        default="vector_data_files/zsre_llama.jsonl",#add file path to vectors file
        help="dir to dataset",
    )
    parser.add_argument("--threshold_sim_pos_pairs", default=0.6, type=float, help="edit to edit pairinig threshold")
    ##################################### Evaluation ############################################
    #paired use
    parser.add_argument("--evaluation_mode", default="transferlearning", type=str, help="evaluation mode \"batch\" or \"transferlearning\"")
    parser.add_argument("--similatiry_thresh_codebook_transferlearning", type=int, default= 6, help="tau adjustment")
    parser.add_argument("--path_to_codebook_transferlearning", type=str, default= "trained_projectors/llama/Llama_zsre_codebook_transferlearning.json", help="type of projector to use")
    #
    # paired use
    parser.add_argument("--similatiry_thresh_codebook_batch", type=int, default= 23, help="tau adjustment")
    parser.add_argument("--path_to_codebook_batch", type=str, default= "trained_projectors/llama/Llama_zsre_codebook_batch.json", help="type of projector to use")
    #
    ##################################### Projector ############################################
    parser.add_argument("--force_train", type=bool, default= False, help="tau adjustment")# KEEP FALSE TO AVOID RETRAINING
    parser.add_argument("--pretrained_proj_path", type=str, default= "trained_projectors/llama/best_model_weights_zsre_llama.pth", help="path to pretrained projector")

    #only use with trainin off and transferlearning
    parser.add_argument("--pretrained_proj_path_transferlearning", type=str, default= "trained_projectors/llama/best_model_weights_counterfact_llama.pth", help="path to pretrained projector")
    
    
    
    parser.add_argument("--in_features", type=int, default= 4096, help="number of input features")
    parser.add_argument("--out_features", type=int, default= 2048, help="number of input features")#can be set to 4032
    

    ##################################### Training setting ############################################
    parser.add_argument("--margin_loss", default=70.0, type=float, help="margin for contrastive loss")#unnormalized
    parser.add_argument("--lr", default=0.009, type=float, help="initial learning rate")
    parser.add_argument("--patience", default=8, type=int, help="number patience")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="weight decay")
    parser.add_argument("--batch_size_training", type=int, default=32768, help="batch size")
    parser.add_argument(
        "--epochs", default=50, type=int, help="number of total epochs to run"
    )


    
    return parser.parse_args(argv)