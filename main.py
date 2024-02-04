import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import diffusionNet
import os
from config import config_dict
from torch.optim.lr_scheduler import OneCycleLR
import math
from exponentialMovingAverage import ExponentialMovingAverage
from torchvision.utils import save_image
from advdiffuser import AdvDiffuser
from classifier import Classifier
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hyper params
batch_size = config_dict['batch_size']
epochs = config_dict['total_epochs']
classify_part_epochs = config_dict['classify_part_epochs']
timesteps = config_dict['timesteps']
base_dim = config_dict['base_dim']
model_ema_steps = config_dict['model_ema_steps']
model_ema_decay = config_dict['model_ema_decay']
lr = config_dict['lr']
n_samples = config_dict['n_samples']
clip_flag = config_dict['clip_flag']
log_freq = config_dict['log_freq']
num_classes = config_dict['num_classes']


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()


def main():
    model = diffusionNet.DiffusionNet(timesteps=timesteps,
                                      image_size=28,
                                      in_channels=1,
                                      base_dim=base_dim,
                                      dim_mults=[2, 4]).to(device)

    adjust = 1 * batch_size * model_ema_steps / epochs
    alpha = 1.0 - model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    ckpt_path = "results/diff_model.pt"

    # checkpoint
    if os.path.exists(ckpt_path):
        print("Loading diffusion model from checkpoint...")
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        model_ema.load_state_dict(ckpt['model_ema'])
        model_ema.eval()
        print("Loading diffusion model completed!")
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = OneCycleLR(optimizer, lr, total_steps=epochs * len(train_loader), pct_start=0.25,
                               anneal_strategy='cos')
        loss_fn = nn.MSELoss(reduction='mean')

        global_steps = 0
        for i in range(epochs):
            model.train()
            for j, (image, target) in enumerate(train_loader):
                noise = torch.randn_like(image).to(device)
                image = image.to(device)
                pred = model(image, noise)
                loss = loss_fn(pred, noise)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if global_steps % model_ema_steps == 0:
                    model_ema.update_parameters(model)
                global_steps += 1
                if j % log_freq == 0:
                    print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i + 1, epochs, j,
                                                                                  len(train_loader),
                                                                                  loss.detach().cpu().item(),
                                                                                  scheduler.get_last_lr()[0]))

            model_ema.eval()
            samples = model_ema.module.sampling(n_samples, clipped_reverse_diffusion=clip_flag, device=device)
            save_image(samples, "results/steps_{:0>8}.png".format(global_steps), nrow=int(math.sqrt(n_samples)))

        ckpt = {"model": model.state_dict(),
                "model_ema": model_ema.state_dict()}

        os.makedirs("results", exist_ok=True)
        torch.save(ckpt, "results/diff_model.pt")

    # classify
    classifier = Classifier().to(device)
    classifier.train_model(device, train_loader, test_loader, epochs=10)
    adv_ori_images, adv_labels = utils.random_loader_sampling(test_loader)
    adv_ori_images, adv_labels = adv_ori_images.to(device), adv_labels.to(device)
    dataset = TensorDataset(adv_ori_images, adv_labels)
    sampled_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    accuracy = utils.evaluate_accuracy(classifier, sampled_data_loader)

    # adversarial attack
    adv_model = AdvDiffuser(model, num_classes, device).to(device)
    # pretrain for grad cam
    adv_model.pretrain_classify_part(classify_part_epochs, train_loader, device)
    adv_images = []
    # generate adversarial samples by batch
    for images_batch, labels_batch in sampled_data_loader:
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
        adv_model.denoise_test_data(images_batch, timesteps)
        adv_batch = adv_model.generate_adversarial_example(images_batch, labels_batch, timesteps)
        adv_images.append(adv_batch.cpu())
    adv_ori_images = torch.cat(adv_images, dim=0)
    # update adversarial dataset
    dataset = TensorDataset(adv_ori_images.to(device), adv_labels)
    sampled_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # adversarial accuracy
    adv_accuracy = utils.evaluate_accuracy(classifier, sampled_data_loader)

    print(f'Accuracy for adversarial samples: {adv_accuracy:.2f}%')
    print(f'Accuracy for original samples: {accuracy:.2f}%')


if __name__ == "__main__":
    main()
