import argparse
import os

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Counterfact Editing Run on GPT2-XL")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--result_dir", type=str, default="result", help="directory of all outputs"
    )
    parser.add_argument("--dataset", type=str, default="counterfact", help="dataset")
    
    parser.add_argument(
        "--data_file",
        type=str,
        default="vector_data_files/counterfact_gpt2xl.jsonl",#add file path to vectors file
        help="dir to dataset",
    )
    parser.add_argument("--threshold_sim_pos_pairs", default=0.5, type=float, help="edit to edit pairinig threshold")
    ##################################### Evaluation ############################################
    #paired use
    parser.add_argument("--evaluation_mode", default="stream", type=str, help="evaluation mode \"batch\" or \"stream\"")
    parser.add_argument("--similatiry_thresh_codebook_stream", type=int, default= 6, help="tau adjustment")
    parser.add_argument("--path_to_codebook_stream", type=str, default= "trained_projectors/gpt2xl/gpt2xl_counterfact_codebook_stream.json", help="type of projector to use")
    
    # paired use
    parser.add_argument("--similatiry_thresh_codebook_batch", type=int, default= 12, help="tau adjustment")
    parser.add_argument("--path_to_codebook_batch", type=str, default= "trained_projectors/gpt2xl/gpt2xl_counterfact_codebook_batch.json", help="type of projector to use")
    #
    ##################################### Projector ############################################
    parser.add_argument('--force_train', action='store_true', help="Force retraining regardless of saved projector.")# KEEP FALSE TO AVOID RETRAINING
    parser.add_argument("--pretrained_proj_path", type=str, default= "trained_projectors/gpt2xl/best_model_weights_counterfact_gpt2xl.pth", help="path to pretrained projector")
    parser.add_argument("--in_features", type=int, default= 6400, help="number of input features")
    parser.add_argument("--out_features", type=int, default= 3200, help="number of input features")#can be set to 6400
    
    
   
    

    ##################################### Training setting ############################################
    parser.add_argument("--margin_loss", default=60.0, type=float, help="margin for contrastive loss")#unnormalized
    parser.add_argument("--lr", default=0.009, type=float, help="initial learning rate")
    parser.add_argument("--patience", default=8, type=int, help="number patience")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="weight decay")
    parser.add_argument("--batch_size_training", type=int, default=4096, help="batch size")
    parser.add_argument(
        "--epochs", default=100, type=int, help="number of total epochs to run"
    )
    
    return parser.parse_args(argv)