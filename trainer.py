from imports import *
from models import SiameseNetwork
from loss_functions import WeightedContrastiveLoss,CosineSimilarityLoss_SentenceTransformers,ContrastiveLoss,TripletLoss
import data_processing_counterfact as dpcounter
import torch.optim as optim
from torch.optim import lr_scheduler
import data_processing_zsre as dpzsre

def train_model_combined(model,optimizer,data_loader,criterion_similarity,
                         criterion_classification=None,scheduler=None,mode="simlarity",
                         path_to_folder="./",file_name="best_model_weights.pth",epochs=50,early_stop_patience=5,device="cpu"):
  # try:
  num_epochs = epochs
  lowest_error = float('inf')
  best_model_weights = None
  counter_early_stop=0
  print("num_epochs",num_epochs)
  for epoch in range(num_epochs):
      total_loss = 0.0
      total_batches = len(data_loader)
      for batch_idx, batch in tqdm(enumerate(data_loader), desc="Processing Batches", total=len(data_loader)):
          # print(batch_idx)
          embs1, embs2, labels, _ , _, _,_ ,_,_,_= batch
          optimizer.zero_grad()# zero out gradients
          if(mode=="classificaiton"):# old not used
            output1, output2, output3 = model(embs1,embs2)
            loss_classification = criterion_classification(output3, labels)
            labels_cosine = torch.where(labels == 0, torch.tensor(0.5).to(device), torch.tensor(0.9).to(device))
            loss_semantic_sim = criterion_similarity(output1,output2, labels_cosine)
            alpha = 0.0
            beta = 1.0
            combined_loss = (alpha * loss_classification) + (beta * loss_semantic_sim)
            combined_loss.backward()
          else:
            output1, output2 = model(embs1,embs2)
            combined_loss = criterion_similarity(output1, output2, labels)#semantic sim loss
            combined_loss.backward()

          optimizer.step()

          total_loss += combined_loss.item()
      # Calculate average loss after the epoch
      if(scheduler != None):
         scheduler.step()
      epoch_loss = total_loss / total_batches
      

      # Check for early stopping
      if epoch_loss < lowest_error:
          lowest_error = epoch_loss
          best_model_weights = model.state_dict()
          counter_early_stop = 0  # Reset the counter when there is an improvement
      else:
          counter_early_stop += 1
      print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} Early Stop: {counter_early_stop}')
      # Check for early stopping
      if counter_early_stop >= early_stop_patience:
          print(f'Early stopping triggered at epoch {epoch + 1} as the loss did not improve.')
          break
  torch.save(best_model_weights, path_to_folder+file_name)
  state_dict = torch.load(path_to_folder+file_name)
  model.load_state_dict(state_dict)
  print('Training finished.')
  return model,path_to_folder+file_name






def train_control_zsre(edit_vectors_dict, neighbourhood_train_vectors_dict, paraphrase_train_vectors_dict, dataset_paired_train,args,path_to_folder="./"
             ,cls_weights=True,device="cpu"):
  """
    loss_function: loss function to be used cosine, cosine_crossentropy, contrastive
    learning_rate: learning rate for Adam optimizer
  """
#32768

  train_data_loader=dpzsre.get_data_loader(dataset_paired_train, edit_vectors_dict, neighbourhood_train_vectors_dict,paraphrase_train_vectors_dict,batch_size=args.batch_size_training,shuffle=True,device=device)
  
  model=SiameseNetwork(input_size=args.in_features, hidden_size1=args.out_features).to(device)
  print(model)
  mode="similarity"
  comparison="dist"
  #12.0=MARGIN
  # Define learning rate scheduler
  class_weights=[1.0,1.0]
  criterion_similarity=WeightedContrastiveLoss(margin=args.margin_loss, positive_weight=class_weights[1], negative_weight=class_weights[0]).to(device)
  criterion_classification=None

  step_size = 5  # Adjust step size as needed
  gamma = 0.1    # Factor by which to reduce learning rate
  
  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)# for cosine use 0.00001 else 0.0001
  # scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
  scheduler=None
  model,file_path=train_model_combined(model,optimizer,train_data_loader,criterion_similarity,criterion_classification,scheduler,mode,path_to_folder,"best_model_weights.pth",early_stop_patience=args.patience,device=device,epochs=args.epochs)
  return model,mode,comparison,file_path









def train_control_counterfact(openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, paraphrase_train_vectors_dict, dataset_paired_train,args,path_to_folder="./"
             ,cls_weights=True,device="cpu"):
  """
    loss_function: loss function to be used cosine, cosine_crossentropy, contrastive
    learning_rate: learning rate for Adam optimizer
  """

  
  class_weights=[1.0,1.0]

  
  # if(loss_function=="cosine" or  loss_function=="cosine_crossentropy"):
  #   train_data_loader=dpcounter.get_data_loader(dataset_paired_train,batch_size=8192,shuffle=True,device=device)
  #   if(loss_function=="cosine_crossentropy"):
  #     mode="classificaiton"
  #     class_weights=torch.tensor(class_weights,dtype=torch.float)
  #     criterion_classification = nn.CrossEntropyLoss(weight=class_weights).to(device)
  #     model=SiameseClassificationNetwork(512).to(device)
  #   else:
  #     criterion_classification=None
  #     mode="similarity"
  #     model=SiameseNetwork(input_size=512, hidden_size1=256).to(device)
  #   criterion_similarity = CosineSimilarityLoss_SentenceTransformers().to(device)
  #   comparison="sim"
  # elif(loss_function=="contrastive"):
    # print("inside contrastive")
    # dataset = dpcounter.LargeDataset(data_path=dataset_paired_train,device=device)
    # print(dataset.__len__())
    # sampler = dpcounter.ChunkSampler(dataset.__len__(), chunk_size=8192)
    # train_data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler,batch_size=8192)
            #
    # train_data_loader=dpcounter.get_data_loader(dataset_paired_train,batch_size=8192,shuffle=True,device=device)
    # print("LOADER 2")

  train_data_loader=dpcounter.get_data_loader(dataset_paired_train,openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict,paraphrase_train_vectors_dict,batch_size=args.batch_size_training,device=device)
  
  
  model=SiameseNetwork(input_size=args.in_features, hidden_size1=args.out_features).to(device)
  print(model)
  mode="similarity"
  comparison="dist"
  criterion_similarity=WeightedContrastiveLoss(margin=args.margin_loss, positive_weight=class_weights[1], negative_weight=class_weights[0]).to(device)
  criterion_classification=None
  # else:
  #   train_data_loader=dpcounter.get_data_loader_triplet(dataset_paired_train,batch_size=512,shuffle=True,device=device)
  #   model=SiameseNetworkTriplet(512).to(device)
  #   mode="similarity"
  #   comparison="dist"
  #   #12.0=MARGIN
  #   criterion_similarity = nn.TripletMarginLoss(margin=10.0, p=2, eps=1e-7).to(device)
  #   criterion_classification=None
  #   optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)# for cosine use 0.00001 else 0.0001
  #   model,file_path=train_model_triplet(model,optimizer,train_data_loader,criterion_similarity,path_to_folder,"best_model_weights.pth",early_stop_patience=early_stop_patience,device=device,epochs=epochs)
  # return model,mode,comparison,file_path
  
  step_size = 5  # Adjust step size as needed
  gamma = 0.1    # Factor by which to reduce learning rate
  
  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)# for cosine use 0.00001 else 0.0001
  # scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
  scheduler=None
  model,file_path=train_model_combined(model,optimizer,train_data_loader,criterion_similarity,criterion_classification,scheduler,mode,path_to_folder,"best_model_weights.pth",early_stop_patience=args.patience,device=device,epochs=args.epochs)
  return model,mode,comparison,file_path
