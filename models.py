import torch
import torch.nn as nn

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, im_dim_mul,
                 conditional=False, onehot=True, num_labels=10):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.im_dim_mul = im_dim_mul

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, onehot, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, onehot, num_labels)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, self.im_dim_mul)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    # proposed in original vae paper 
    # epsilon allows vae to be trained end by end because now mu+eps*std is a learnable parameter
    # very low value thereby not causing the network to shift away too much from the true distribution
    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, onehot, num_labels=10):

        super().__init__()

        self.conditional = conditional
        self.num_labels = num_labels
        if self.conditional:
            layer_sizes[0] += num_labels
        self.onehot = onehot
        self.MLP = nn.Sequential()

        #print("\n\nencoder loop: ", list(zip(layer_sizes[:-1], layer_sizes[1:])),"\n")
        # [(794, 256)] 

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            if self.onehot:
                c = idx2onehot(c, n=self.num_labels)
            x = torch.cat((x, c), dim=-1)
            # x shape becomes [64,794] from [64,784] (28x28 im -> 1x784 vector)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, onehot, num_labels=10):

        super().__init__()

        self.MLP = nn.Sequential()
        self.num_labels = num_labels
        self.conditional = conditional
        self.onehot = onehot
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size


        #print("\n\ndecoder loop: ", list(zip([input_size]+layer_sizes[:-1], layer_sizes)), "\n")
        #[(12, 256), (256, 784)] -> input and output shapes of decoder 

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

        #print(self.MLP)

    def forward(self, z, c):

        if self.conditional:
            if self.onehot:
                c = idx2onehot(c, n=self.num_labels)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
