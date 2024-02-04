import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import os
import tqdm


class AdvDiffuser(nn.Module):
    def __init__(self, diffusion_model, num_classes, device):
        super(AdvDiffuser, self).__init__()
        self.diffusion_model = diffusion_model
        self.device = device
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(diffusion_model.model.final_conv.out_channels, num_classes)
        self.target_layer = "model.final_conv"

    def forward(self, x):
        noise = torch.randn_like(x).to(self.device)
        x = self.diffusion_model(x, noise)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def classify(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def pretrain_classify_part(self, epochs, train_loader, device):
        ckpt_path = "results/grad_cam_classifier.pt"
        if os.path.exists(ckpt_path):
            print("Loading grad_cam_classifier from checkpoint...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            self.load_state_dict(checkpoint['model_state'])
            print("Loading grad_cam_classifier completed!")
            return

        print("Start training grad_cam_classifier.")
        # freeze diffusion model part
        # for param in self.diffusion_model.parameters():
        #     param.requires_grad = False

        optimizer = torch.optim.Adam(list(self.gap.parameters()) + list(self.fc.parameters()), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        # save check point
        if not os.path.isdir('results'):
            os.makedirs('results')
        torch.save({
            'model_state': self.state_dict(),
        }, ckpt_path)

    def generate_grad_cam_heatmap(self, x, class_idx):
        target_layer = dict(self.diffusion_model.named_modules())[self.target_layer]
        activations = []
        gradients = []

        def save_activation(module, input, output):
            activations.append(output)

        def save_gradient(module, input_grad, output_grad):
            gradients.append(output_grad[0])

        # regester hook
        hook_a = target_layer.register_forward_hook(save_activation)
        hook_b = target_layer.register_backward_hook(save_gradient)

        # forward
        output = self(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1)

        # backward
        self.diffusion_model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output.scatter_(1, class_idx.unsqueeze(1), 1)
        output.backward(gradient=one_hot_output, retain_graph=True)

        # remove hook
        hook_a.remove()
        hook_b.remove()

        # cal grad cam
        activation = activations[0]
        grad = gradients[0]
        pooled_grad = torch.mean(grad, dim=[0, 2, 3], keepdim=True)
        grad_cam = torch.mul(activation, pooled_grad).sum(dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)
        grad_cam = grad_cam / grad_cam.max()
        return grad_cam

    def sample_xt_minus_1(self, x, alpha_t_minus_1):
        variance = (1 - alpha_t_minus_1)
        mean = torch.sqrt(alpha_t_minus_1) * x
        noise = torch.randn_like(x)
        x_t_minus_1_obj = mean + torch.sqrt(variance) * noise
        return x_t_minus_1_obj

    def generate_adversarial_example(self, x, class_idx, timesteps, iterations=2, clipped_reverse_diffusion=True):
        # Generate the inverse of the Grad-CAM heatmap
        heatmap = self.generate_grad_cam_heatmap(x, class_idx)
        utils.save_grad_cam_heatmap(heatmap, 'results/grad_cam_heatmap.png')
        inverse_heatmap = 1 - heatmap

        t = torch.randint(0, timesteps, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x).to(self.device)
        x_t = self.diffusion_model._forward_diffusion(x, t, noise)

        for batch_idx in range(x.shape[0]):
            current_t = t[batch_idx].item()
            current_x = x_t[batch_idx].unsqueeze(0)
            for time_step in range(current_t, -1, -1):
                # denoising
                current_noise = torch.randn_like(current_x).to(self.device)
                current_t_tensor = torch.tensor([time_step], device=current_x.device, dtype=torch.long)
                if clipped_reverse_diffusion:
                    z_t = self.diffusion_model._reverse_diffusion_with_clip(current_x, current_t_tensor, current_noise)
                else:
                    z_t = self.diffusion_model._reverse_diffusion(current_x, current_t_tensor, current_noise)

                if time_step < current_t / 4:
                    for _ in range(iterations):
                        # predict
                        z_t.requires_grad_()
                        logits = self.classify(z_t)
                        pred_class = logits.argmax(dim=1)
                        if pred_class != class_idx[batch_idx]:
                            break

                        # carry on adding perturbation
                        loss = F.cross_entropy(logits, class_idx[batch_idx].unsqueeze(0))
                        loss.backward()

                        with torch.no_grad():
                            perturbation = z_t.grad.sign()
                            z_t = z_t + perturbation

                        self.diffusion_model.zero_grad()
                        z_t.grad = None

                    # sample
                    alpha = self.diffusion_model.alphas_cumprod.gather(-1, t)
                    x_sample = self.sample_xt_minus_1(z_t, alpha[batch_idx])

                    # combine
                    current_x = x_sample * heatmap[batch_idx] + z_t * inverse_heatmap[batch_idx]
                else:
                    current_x = z_t

            current_x = torch.clamp(current_x, min=0, max=1).detach()
            x_t[batch_idx] = current_x
        utils.save_adversarial_example(x_t, 'results/adversarial_example.png')
        return x_t

    def denoise_test_data(self, x, timesteps, clipped_reverse_diffusion=True):
        t = torch.randint(0, timesteps, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x).to(self.device)
        x_t = self.diffusion_model._forward_diffusion(x, t, noise)

        for batch_idx in range(x.shape[0]):
            current_t = t[batch_idx].item()
            current_x = x_t[batch_idx].unsqueeze(0)

            for time_step in range(current_t, -1, -1):
                current_noise = torch.randn_like(current_x).to(self.device)
                # denoising
                current_t_tensor = torch.tensor([time_step], device=current_x.device, dtype=torch.long)
                if clipped_reverse_diffusion:
                    current_x = self.diffusion_model._reverse_diffusion_with_clip(current_x, current_t_tensor,
                                                                                  current_noise)
                else:
                    current_x = self.diffusion_model._reverse_diffusion(current_x, current_t_tensor, current_noise)

            current_x = (current_x + 1.) / 2.
            current_x = torch.clamp(current_x, min=0, max=1).detach()
            x_t[batch_idx] = current_x

        utils.save_adversarial_example(x_t, 'results/denoise_test.png')
        return x_t

