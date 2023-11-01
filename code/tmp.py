import torch

from torchvision import transforms, datasets

dataset = datasets.SUN397(root='/D_data/kaqiu/datasets/SUN397', download=True, transform=transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

print(len(dataset))

for i, data in enumerate(dataloader):
    print(i, data[0].shape, data[1])
    # break