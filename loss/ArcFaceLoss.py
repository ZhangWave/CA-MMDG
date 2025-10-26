import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdditiveAngularMarginLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, scale=30.0, margin=0.5):
        super(AdditiveAngularMarginLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize the embeddings and weights
        embeddings_norm = F.normalize(embeddings, dim=1)
        weights_norm = F.normalize(self.weight, dim=1)

        # Compute cosine similarity between embeddings and weights
        cos_theta = torch.matmul(embeddings_norm, weights_norm.T)

        # Get one-hot encoding of labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Apply additive angular margin to the target class
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Ensure that cos(theta + m) is within [-1, 1]
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)

        # Scale the features and apply margin only to the correct class
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        output *= self.scale

        # Compute cross entropy loss
        loss = F.cross_entropy(output, labels)

        return loss

# Example usage
num_classes = 2  # 二分类问题
embedding_dim = 512
batch_size = 64

# Initialize model and loss function
model = nn.Linear(embedding_dim, embedding_dim)  # Dummy model to generate embeddings
criterion = AdditiveAngularMarginLoss(num_classes=num_classes, embedding_dim=embedding_dim, scale=30.0, margin=0.35)

# Dummy data
embeddings = torch.randn(batch_size, embedding_dim)  # Batch size of 64
labels = torch.randint(0, num_classes, (batch_size,))

# Forward pass through the dummy model
features = model(embeddings)

# Compute loss
loss = criterion(features, labels)
print(f"Loss: {loss.item()}")



