from imports import torch,F,util,json,np,f1_score,tqdm,os
import helper_functions as hp
import data_processing_counterfact as dpcounter
import data_processing_zsre as dpzsre



def get_model_results(model,data_loader,threshold=0.5,comparison="sim",mode="similarity",device="cuda"):
  model.to(device)
  model.eval()
  
  predictions=[]
  targets=[]
  threshold_predictions=[]
  distances_sims=[]
  
  with torch.no_grad():
    
    for batch in data_loader:
      embs1,embs2, labels,index, sent1, sent2,_,_,_,_ = batch

      if(mode=="similarity"):
        output1, output2 = model(embs1.to(device),embs2.to(device))
      else:
        output1, output2,_ = model(embs1.to(device),embs2.to(device))

      if(comparison=="dist"):
        distance=(F.pairwise_distance(output1, output2, keepdim=True))[0][0].item()#/ (torch.norm(output1) + torch.norm(output2)))[0][0].item()
        predictions.append(distance)
      else:
        cosine_scores = util.cos_sim(output1, output2)
        sim=cosine_scores[0][0].to(device).numpy()
        predictions.append(sim)

      label=labels[0].item()
      # print(label)
      # if(label==1):
      #   print("aloha",label)
      targets.append(label)


      if(comparison=="dist"):
        dist=(F.pairwise_distance(embs1,embs2, keepdim=True))[0][0].item()#[0][0].item()/ (torch.norm(embs1) + torch.norm(embs2)))[0][0].item()
        distances_sims.append([distance,dist,label])
        threshold_predictions.append(1 if (distance < threshold) else 0)#classes are reversed
      else:
        distances_sims.append([sim,util.cos_sim(embs1,embs2)[0][0].item(),label])
        threshold_predictions.append(0 if (sim < threshold) else 1)
  return distances_sims,predictions,threshold_predictions,targets

def get_best_threshold(precision, recall, thresholds_pr,predictions,targets,comparison):
  
  f1_scores = 2 * (precision * recall) / (precision + recall)
  f1_scores = np.nan_to_num(f1_scores, nan=0)

  try:
    best_threshold = thresholds_pr[np.argmax(f1_scores)]
  except:
    best_threshold = thresholds_pr[-1]
  print(f"Best Threshold: {best_threshold}")
  # Use the best threshold to make predictions
  # y_pred = (predictions >= best_threshold).astype(int)
  if(comparison=="dist"):
    y_pred = (predictions <= best_threshold).astype(int)
  else:
    y_pred = (predictions >= best_threshold).astype(int)

  # Evaluate the model with the best threshold
  f1 = f1_score(targets, y_pred)
  print(f"F1 Score with Best Threshold: {f1}")
  return best_threshold


def write_result_to_file(model,dataset_test,comparison,mode,file_path,device="cpu",loss_function="contrastive"):
  with open(file_path, 'w') as jsonl_file_writer:
    with torch.no_grad():
      if(loss_function=="triplet"):
        data_loader_test_temp=dpcounter.get_data_loader_triplet(dataset_test,batch_size=1,shuffle=False,device=device)
      else:
        data_loader_test_temp=dpcounter.get_data_loader(dataset_test,batch_size=1,shuffle=False,device=device)

      model.to(device)
      for batch in data_loader_test_temp:
        if(loss_function=="triplet"):
          embs1,embs2, embs3,index, sent1, sent2,sent3,_ = batch
          output1, output2, output3 = model(embs1,embs2,embs3)
          after_projection_distance_neightbour=(F.pairwise_distance(output1, output2, keepdim=True))[0][0].item()
          before_projection_distance_neightbour=(F.pairwise_distance(embs1, embs2, keepdim=True))[0][0].item()
        
          after_projection_distance_paraphrase=(F.pairwise_distance(output2, output3, keepdim=True))[0][0].item()
          before_projection_distance_paraphrase=(F.pairwise_distance(embs2, embs3, keepdim=True))[0][0].item()
          data_entry={"sentence_edit":sent1,
                                "sentence_neighbour":sent2,
                                "sentence_paraphrase":sent3,
                                "index":index.numpy().tolist(),
                                "after_projection_distance_neightbour":after_projection_distance_neightbour,
                                "before_projection_distance_neightbour":before_projection_distance_neightbour,
                                "after_projection_distance_paraphrase":after_projection_distance_paraphrase,
                                "before_projection_distance_paraphrase":before_projection_distance_paraphrase}
          json.dump(data_entry, jsonl_file_writer)
          jsonl_file_writer.write('\n')
        else:
          embs1,embs2, label,index, sent1, sent2,_ = batch
          if(mode=="similarity"):
            output1, output2 = model(embs1,embs2)
          else:#classification cosine_crossentropy
            output1, output2,_ = model(embs1,embs2)

          if(comparison=="sim"):
            cosine_scores =util.cos_sim(embs1,embs2)
            before_projection_sim=cosine_scores[0][0].numpy().tolist()
            cosine_scores =util.cos_sim(output1, output2)
            after_projection_sim=cosine_scores[0][0].numpy().tolist()
            data_entry={"sentence1":sent1,
                                  "sentence2":sent2,
                                  "label":label.numpy().tolist(),
                                  "similarity_before_projection":before_projection_sim,
                                  "similarity_after_projection":after_projection_sim}
          else:
            after_projection_distance=(F.pairwise_distance(output1, output2, keepdim=True))[0][0].item()
            before_projection_distance=(F.pairwise_distance(embs1, embs2, keepdim=True))[0][0].item()
            data_entry={"sentence1":sent1,
                                  "sentence2":sent2,
                                  "label":label.numpy().tolist(),
                                  "distance_before_projection":before_projection_distance,
                                  "distance_after_projection":after_projection_distance}
          json.dump(data_entry, jsonl_file_writer)
          jsonl_file_writer.write('\n')

def write_results_to_file_vector_list(dataset_train,dataset_test,model,comparison,mode,file_path,device="cpu"):
  data_loader_test=dpcounter.get_data_loader(dataset_train,batch_size=1,shuffle=False,device=device)
  vector_list=[]
  string_list=[]
  with torch.no_grad():
    model.to(device)
    for sample in data_loader_test:
      if(mode=="similarity"):
        output1, _ = model(sample[0],sample[0])
      else:
        output1, _, _ = model(sample[0],sample[0])
      vector_list.append(output1[0])
      string_list.append(sample[4][0])

  data_loader_train=dpcounter.get_data_loader(dataset_test,batch_size=1,shuffle=False)
  counter=0
  
  with torch.no_grad():
    with open(file_path, 'w') as jsonl_file_writer:
      for sample in data_loader_train:
        if(mode=="similarity"):
          output1, output2= model(sample[0],sample[1])
        else:
          output1, output2, _ = model(sample[0],sample[1])
        if(sample[2].item()==0):
          vector,index,distance_sim=hp.find_nearest_vector(output1[0],vector_list,comparison)
        else:
          vector,index,distance_sim=hp.find_nearest_vector(output2[0],vector_list,comparison)
        # print(sample[3].numpy().tolist(),int(index/6),index%6) alternative for checking if matching is correct
        
        if(string_list[index]==sample[4][0]):
          matching=1
        else:
          matching=0
          counter+=1

        sim1=distance_sim.numpy().tolist()
        sim2=util.cos_sim(output1[0],output2[0]).numpy()[0][0].tolist()
        sim_difference=abs(sim1-sim2)
        data_entry={"label":sample[2].numpy().tolist(),"Matching":matching,"Test Sentence":sample[5][0],
                                  "Matched Sentence":string_list[index],
                                  "Comaprison String":sample[4][0],
                                  "Distance/Sim predicted":sim1,
                                  "Distance/Sim Dataset":sim2,
                                    "Sim Difference":sim_difference}
        json.dump(data_entry, jsonl_file_writer)
        jsonl_file_writer.write('\n')

def read_codebook(file_path):
  with open(file_path, "r") as f:
    data_loaded = json.load(f)

  # Reconstruct the variables
  vector_list = [torch.tensor(item["vector"]) for item in data_loaded]  # List of tensors
  threshold_list = [float(item["threshold"]) for item in data_loaded]   # List of floats
  string_list = [item["label"] for item in data_loaded] 
  return  vector_list, threshold_list, string_list

def write_codebook(file_path,vector_list,threshold_list,string_list):
  data_to_save = []
  for vector, threshold, label in zip(vector_list, threshold_list, string_list):
      data_to_save.append({
          "vector": vector.tolist(),       # Convert tensor to list
          "threshold": float(threshold),   # Ensure float type
          "label": label                   # String label
      })

  # Write to JSON file
  with open(file_path, "w") as f:
      json.dump(data_to_save, f, indent=2)   

def write_results_indivisual_threshold(openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test,model,comparison,mode,file_path,d,args,device="cpu",loss_function="contrastive"):
  if(args.evaluation_mode=="stream"):
    codebook_file_path=args.path_to_codebook_stream
  else:
    codebook_file_path=args.path_to_codebook_batch
  if(os.path.isfile(codebook_file_path)):#force recompute codebook
    print("loading precomputed codebook")
    vector_list,threshold_list,string_list=read_codebook(codebook_file_path)
  else:
    print("computing codebook")
    data_loader=dpcounter.get_data_loader(dataset_paired_train,openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict,paraphrase_train_vectors_dict,batch_size=1,shuffle=False,device=device)
    
    distances_sims,predictions_sim,_,targets=get_model_results(model,data_loader,threshold=0.5,comparison=comparison,mode=mode,device=device)
    
    
    vector_list,threshold_list,string_list=hp.threshold_per_index(model,dpcounter.CustomDataset(dataset_paired_train,openai_vectors_dict, edit_vectors_dict, neighbourhood_train_vectors_dict, paraphrase_train_vectors_dict,device=device)
                                                                ,targets,predictions_sim,comparison,mode,d)
    write_codebook(codebook_file_path,vector_list,threshold_list,string_list)
                                                           
 
  model.to(device)
  model.eval()
  
  data_loader=dpcounter.get_data_loader(dataset_paired_test,openai_vectors_dict, edit_vectors_dict, neighbourhood_test_vectors_dict,paraphrase_test_vectors_dict,batch_size=1,shuffle=False,device=device)
    
  counter=0
  predictions=[]
  targets=[]
  with torch.no_grad():
    with open(file_path, 'w') as jsonl_file_writer:
      for data in tqdm(data_loader, desc="Computing results", unit="one unit"):
        label=data[2].item()
        targets.append(label)
        # (check the dataloaders in data processing files for format)
        if(label==1):
          vector1= data[1]#pick the paraphrase vector 
          vector2= data[0]#orignal prompt
        else:
          vector1= data[0]#pick the neighbourhood vector
          vector2= data[1]#orignal prompt
        vector1=vector1.to(device)
        vector2=vector2.to(device)
        # vector1 = vector1.squeeze(1) 
        # vector2 = vector2.squeeze(1)
    
        if(mode!="classificaiton"):
          output1, output2 = model(vector1,vector2)
        else:
          output1, output2,_ = model(vector1,vector2)

        if(comparison=="sim"):
          sim2=util.cos_sim(output1[0],output2[0]).numpy()[0][0].tolist()
        else:
          sim2=F.pairwise_distance(output1, output2, keepdim=True)[0][0].item()
          
        _,best_distance_index,best_distance=hp.find_nearest_vector(output1,vector_list,comparison,device)


        if(best_distance.item()<=threshold_list[best_distance_index]):# difference in distance vs sim
          if(comparison=="sim"):
            pred=0
          else:
            pred=1# this one
        else:
          if(comparison=="sim"):
            pred=1
          else:
            pred=0# this one

        
        sim_difference=abs(best_distance.item()-sim2)#sim difference should be zero if correct edit is matched

        if(string_list[best_distance_index]==data[4][0]):#exact match
          matching=1
          predictions.append(pred)
        else:
          matching=0
          if(label==0):# if label is negative we want the prediction as is
            predictions.append(pred)
          else:# if label is 1 we want to check if we picked the right one
            if pred == 1:
              pred=0 
            predictions.append(pred)



        data_entry={"label":label,"Matching":matching,"Prediction":pred,"Threshold":threshold_list[best_distance_index],"Distance/Sim predicted":best_distance.item(),
                                        "Distance/Sim Dataset":sim2,
                                        "Sim Difference":sim_difference,
                                        "Test Sentence":data[5][0],
                                        "Matched Sentence":string_list[best_distance_index],
                                        "Comaprison String":data[4][0],
                                        }
        json.dump(data_entry, jsonl_file_writer)
        jsonl_file_writer.write('\n')

  return predictions,targets





def write_results_indivisual_threshold_zsre(edit_vectors_dict, neighbourhood_train_vectors_dict, neighbourhood_test_vectors_dict, paraphrase_train_vectors_dict, paraphrase_test_vectors_dict, dataset_paired_train, dataset_paired_test,model,comparison,mode,file_path,threshold,args,device="cpu",loss_function="contrastive"):
  if(args.evaluation_mode=="transferlearning"):
    codebook_file_path=args.path_to_codebook_transferlearning
  else:
    codebook_file_path=args.path_to_codebook_batch

  if(os.path.isfile(codebook_file_path)):#force recompute codebook
    print("loading precomputed codebook")
    vector_list,threshold_list,string_list=read_codebook(codebook_file_path)
  else:
    print("computing codebook")
    data_loader=dpzsre.get_data_loader(dataset_paired_train, edit_vectors_dict, neighbourhood_train_vectors_dict,paraphrase_train_vectors_dict,batch_size=1,shuffle=False,device=device)
    
    distances_sims,predictions_sim,_,targets=get_model_results(model,data_loader,threshold=0.5,comparison=comparison,mode=mode,device=device)
    
    
    vector_list,threshold_list,string_list=hp.threshold_per_index(model,dpzsre.CustomDataset(dataset_paired_train, edit_vectors_dict, neighbourhood_train_vectors_dict, paraphrase_train_vectors_dict,device=device)
                                                              ,targets,predictions_sim,comparison,mode,threshold)

    write_codebook(codebook_file_path,vector_list,threshold_list,string_list)

  
  model.to(device)
  model.eval()
  data_loader=dpzsre.get_data_loader(dataset_paired_test, edit_vectors_dict, neighbourhood_test_vectors_dict,paraphrase_test_vectors_dict,batch_size=1,shuffle=False,device=device)
    
  counter=0
  predictions=[]
  targets=[]
  with torch.no_grad():
    with open(file_path, 'w') as jsonl_file_writer:
      for data in tqdm(data_loader, desc="Processing results", unit="one unit"):
        label=data[2].item()
        targets.append(label)
        if(label==1):
          vector1= data[1]#pick the paraphrase vector
          vector2= data[0]#orignal prompt
        else:
          vector1= data[0]#pick the neighbourhood vector
          vector2= data[1]#orignal prompt
        if(mode!="classificaiton"):
          output1, output2 = model(vector1,vector2)
        else:
          output1, output2,_ = model(vector1,vector2)

        if(comparison=="sim"):
          sim2=util.cos_sim(output1[0],output2[0]).numpy()[0][0].tolist()
        else:
          sim2=F.pairwise_distance(output1, output2, keepdim=True)[0][0].item()
          
        _,best_distance_index,best_distance=hp.find_nearest_vector(output1,vector_list,comparison,device)
        
        if(best_distance.item()<=threshold_list[best_distance_index]):# difference in distance vs sim
          if(comparison=="sim"):
            pred=0
          else:
            pred=1
        else:
          if(comparison=="sim"):
            pred=1
          else:
            pred=0
        # print(pred,best_distance.item(),label,threshold_list[best_distance_index])
        
        sim_difference=abs(best_distance.item()-sim2)#sim difference should be zero if correct edit is matched
        if(string_list[best_distance_index]==data[4][0]):
          matching=1
          predictions.append(pred)
        else:
          matching=0
          if(label==0):# if label is negative we want the prediction as is
            predictions.append(pred)
          else:# if label is 1 we want to check if we picked the right one
            if pred == 1:
              pred=0 
            predictions.append(pred)

        data_entry={"label":label,"Matching":matching,"Prediction":pred,"Threshold":threshold_list[best_distance_index],"Distance/Sim predicted":best_distance.item(),
                                        "Distance/Sim Dataset":sim2,
                                        "Sim Difference":sim_difference,
                                        "Test Sentence":data[5][0],
                                        "Matched Sentence":string_list[best_distance_index],
                                        "Comaprison String":data[4][0],
                                        }
        json.dump(data_entry, jsonl_file_writer)
        jsonl_file_writer.write('\n')

  return predictions,targets