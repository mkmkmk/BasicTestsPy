"""

https://towardsdatascience.com/understanding-autoencoders-with-an-example-a-step-by-step-tutorial-693c3a4e9836

"""


import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader


def draw_circle(radius, center_x=0.5, center_y=0.5, size=28):
    # draw a circle using coordinates for the center, and the radius
    circle = plt.Circle((center_x, center_y), radius, color='k', fill=False)
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.add_patch(circle)
    ax.axis('off')
    buf = fig.canvas.print_to_buffer()
    plt.close()
    # converts matplotlib figure into PIL image, make it grayscale, and resize it
    return np.array(Image.frombuffer('RGBA', buf[1], buf[0]).convert('L').resize((int(size), int(size))))

def gen_circles(n, size=28):
    # generates random coordinates around (0.5, 0.5) as center points
    center_x = np.random.uniform(0.0, 0.03, size=n).reshape(-1, 1)+.5
    center_y = np.random.uniform(0.0, 0.03, size=n).reshape(-1, 1)+.5
    # generates random radius sizes between 0.03 and 0.47
    radius = np.random.uniform(0.03, 0.47, size=n).reshape(-1, 1)
    sizes = np.ones((n, 1))*size

    coords = np.concatenate([radius, center_x, center_y, sizes], axis=1)
    # generates circles using draw_circle function
    circles = np.apply_along_axis(func1d=lambda v: draw_circle(*v), axis=1, arr=coords)
    return circles, radius


def figure1(dataset):
    real = dataset.tensors[0][:10].numpy()
    real = np.rollaxis(real, 1, 4)

    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    axs = axs.flatten()
    for i in range(10):
        axs[i].set_title(f'Image #{i}')
        axs[i].imshow(real[i].squeeze(), cmap='gray', vmin=0, vmax=1)
    fig.tight_layout()
    return fig


np.random.seed(42)
# generates 1,000 circles
circles, radius = gen_circles(1000)
circles_ds = TensorDataset(torch.as_tensor(circles).unsqueeze(1).float()/255, torch.as_tensor(radius))
circles_dl = DataLoader(circles_ds, batch_size=32, shuffle=True, drop_last=True)


fig = figure1(circles_ds)

plt.pause(0)

import torch.nn as nn

def set_seed(self, seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

class Encoder(nn.Module):
    def __init__(self, input_shape, z_size, base_model):
        super().__init__()
        self.input_shape = input_shape
        self.z_size = z_size
        self.base_model = base_model
        
        # appends the "lin_latent" linear layer to map from "output_size" 
        # given by the base model to desired size of the representation (z_size)
        output_size = self._get_output_size()
        self.lin_latent = nn.Linear(output_size, z_size)
        
    def _get_output_size(self):
        # builds a dummy batch containing one dummy tensor
        # full of zeroes with the same shape as the inputs
        device = next(self.base_model.parameters()).device.type
        dummy = torch.zeros(1, *self.input_shape, device=device)
        # sends the dummy batch through the base model to get 
        # the output size produced by it
        size = self.base_model(dummy).size(1)
        return size
        
    def forward(self, x):
        # forwards the input through the base model and then the "lin_latent" layer 
        # to get the representation (z)
        base_out = self.base_model(x)
        out = self.lin_latent(base_out)        
        return out



set_seed(13)

# we defined our representation (z) as a vector of size one
z_size = 1
# our images are 1@28x28
input_shape = (1, 28, 28) # (C, H, W)

base_model = nn.Sequential(
    # (C, H, W) -> C*H*W
    nn.Flatten(),
    # C*H*W -> 2048
    nn.Linear(np.prod(input_shape), 2048),
    nn.LeakyReLU(),
    # 2048 -> 2048
    nn.Linear(2048, 2048),
    nn.LeakyReLU(),
)

encoder = Encoder(input_shape, z_size, base_model)

encoder

x, _ = circles_ds[7]
z = encoder(x)
z

decoder = nn.Sequential(
    # z_size -> 2048
    nn.Linear(z_size, 2048),
    nn.LeakyReLU(),
    # 2048 -> 2048
    nn.Linear(2048, 2048),
    nn.LeakyReLU(),
    # 2048 -> C*H*W
    nn.Linear(2048, np.prod(input_shape)),
    # C*H*W -> (C, H, W)
    nn.Unflatten(1, input_shape)
)

decoder

x_tilde = decoder(z)

x_tilde 
x_tilde.shape

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder
        
    def forward(self, x):
        # when encoder met decoder
        enc_out = self.enc(x)
        return self.dec(enc_out)
    
model_ae = AutoEncoder(encoder, decoder)

model_ae


set_seed(13)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_ae.to(device)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model_ae.parameters(), 0.0003)

num_epochs = 10

train_losses = []


for epoch in range(1, num_epochs+1):
    batch_losses = []
    for i, (x, _) in enumerate(circles_dl):
        model_ae.train()
        x = x.to(device)

        # Step 1 - Computes our model's predicted output - forward pass
        yhat = model_ae(x)
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, x) 
        # Step 3 - Computes gradients
        loss.backward()
        # Step 4 - Updates parameters using gradients and the learning rate
        optim.step()
        optim.zero_grad()
        
        batch_losses.append(np.array([loss.data.item()]))

    # Average over batches
    train_losses.append(np.array(batch_losses).mean(axis=0))

    print(f'Epoch {epoch:03d} | Loss >> {train_losses[-1][0]:.4f}')



def show(tensor, ax=None):
    img = np.rollaxis(tensor.detach().cpu().numpy(), 0, 3)
    kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1} if img.shape[-1] == 1 else {}
    if ax is None:
        plt.imshow(img.squeeze(), **kwargs)
    else:
        ax.imshow(img.squeeze(), **kwargs)

def set_fontsize(ax, size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)


def figure2(autoencoder, image, device):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    show(image, ax=axs[0])
    z = autoencoder.enc(image.to(device))
    show(autoencoder.dec(z)[0], ax=axs[2])
    axs[0].set_title('Original')
    axs[1].axis('off')
    axs[1].annotate(f'z = [{z.item():.4f}]', (0.25, .5), fontsize=20)
    axs[1].set_title('Latent Space')
    axs[2].set_title('Reconstructed')
    for i in range(3):
        set_fontsize(axs[i], 20)
    return fig

fig = figure2(model_ae, circles_ds[7][0], device)


plt.pause(0)













