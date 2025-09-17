import os, json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_model import SerbianBanknoteCNN

with open("./configs/cnn_config.json", "r") as f:
    config = json.load(f)

test_dir = config['data']['test_dir']
save_dir = config['model']['save_dir']
weights_file = config['model']['weights_file']
img_size = config['model']['img_size']
class_names = config['classes']
num_classes = len(class_names)
batch_size = config['training']['batch_size']

transform = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SerbianBanknoteCNN(num_classes).to(device)
model.load_state_dict(torch.load(os.path.join(save_dir, weights_file), map_location=device))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_names = [class_names[i] for i in predicted.tolist()]
        actual_names = [class_names[i] for i in labels.tolist()]
        print(f"Predict:  {predicted_names}")
        print(f"Actually: {actual_names}")

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
