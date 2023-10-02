import torch
from copy_of_signature_classification_using_siamese import SiameseNetwork
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()  # Set the model to evaluation mode (important for inference).




# Example image paths
image1_path = "photos\image1.png"
image2_path = "photos\image2.png"

# Load and preprocess the images
transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
img1 = Image.open(image1_path).convert('L')
img2 = Image.open(image2_path).convert('L')
img1 = transform(img1).to(device)
img2 = transform(img2).to(device)

# Forward pass to get the embeddings for the images
with torch.no_grad():
    output1, output2 = model(img1.unsqueeze(0), img2.unsqueeze(0))

# Calculate the similarity score
similarity_score = F.pairwise_distance(output1, output2)
print(similarity_score)
# You can use this similarity score for your application.


