from imports import nn,torch,F

class CosineSimilarityLoss_SentenceTransformers(nn.Module):
  def __init__(self):
        super(CosineSimilarityLoss_SentenceTransformers, self).__init__()
        self.loss_fct = torch.nn.MSELoss()
        self.cos_score_transformation=nn.Identity()

  def forward(self, output1, output2, target):
      score= torch.cosine_similarity(output1.to(dtype=torch.float), output2.to(dtype=torch.float))
      score = self.cos_score_transformation(score)
      loss=self.loss_fct(score, target.to(dtype=torch.float).view(-1))
      return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - target) * 0.5 * torch.pow(euclidean_distance, 2) +
                                      (target) * 0.5 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        
        # Use inverted similarity (1 - similarity) for the positive term
        loss = F.relu((1 - distance_positive) + distance_negative + self.margin)
        return torch.mean(loss)
    
class WeightedContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, positive_weight=1.0, negative_weight=1.0):
        super(WeightedContrastiveLoss, self).__init__()
        self.margin = margin
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Weighted contributions from positive and negative samples
        loss_contrastive = torch.mean(
            label * self.positive_weight  * torch.pow(euclidean_distance, 2) +
            (1 - label) * self.negative_weight  * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

