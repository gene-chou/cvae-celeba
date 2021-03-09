import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST, CelebA
from torch.utils.data import DataLoader
from dataloader import get_celeba_selected_dataset, mnist_check
from collections import defaultdict

from models import VAE


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    if args.dataset == 'mnist':
        mnist_check() # required to avoid MNIST http 403 download error 
        dataset = MNIST(
            root='data', train=True, transform=transforms.ToTensor(),
            download=True)
        im_dim = (28, 28)
        encoder_layer_sizes = [784, 256] # 28x28 for mnist 
        decoder_layer_sizes = [256, 784]

    elif args.dataset == 'celeba_torchvision':
        dataset = CelebA(
            root='data', split='train', target_type='attr', transform=transforms.ToTensor(),
            download=True)
        im_dim = (218, 178, 3)
        encoder_layer_sizes = [116412, 256] #(218, 178, 3) for celebA = 116412
        decoder_layer_sizes = [256, 116412]

    elif args.dataset == 'celeba':
        dataset = get_celeba_selected_dataset()
        im_dim = (218, 178, 3)
        encoder_layer_sizes = [116412, 256]
        decoder_layer_sizes = [256, 116412]

    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)


    def im_dim_multiply(im_dim):
        dim = 1
        for i in im_dim:
            dim *= i    
        return dim 

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, im_dim_multiply(im_dim)), x.view(-1, im_dim_multiply(im_dim)), reduction='mean')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=decoder_layer_sizes,
        im_dim_mul = im_dim_multiply(im_dim),
        conditional=args.conditional,
        onehot = args.dataset=='mnist',
        num_labels=10 if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            # each y is an array of batch size of true labels 
            if args.dataset == 'mnist':
                for i, yi in enumerate(y):
                    id = len(tracker_epoch)
                    tracker_epoch[id]['x'] = z[i, 0].item()
                    tracker_epoch[id]['y'] = z[i, 1].item()
                    tracker_epoch[id]['label'] = yi.item()

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

                if args.conditional:

                    if args.dataset == 'mnist':
                        c = torch.arange(0, 10).long().unsqueeze(1).to(device) # array of [[0],[1],...,[9]]
                    elif args.dataset == 'celeba':
                        c = torch.randint(low=0, high=2, size=(1,10)) #populated with 0s and 1s
                        for i in range(9):
                            c = torch.cat((c, torch.randint(low=0, high=2, size=(1,10))), 0)
                        c = c.to(device)
                    z = torch.randn([c.size(0), args.latent_size]).to(device)
                    x = vae.inference(z, c=c)

                else:
                    z = torch.randn([10, args.latent_size]).to(device)
                    x = vae.inference(z)

                # inference for both conditional and non-conditional returns tensor of shape (len(c) x original im size (784 not 794))
                # where len(c) corresponds to the len(c) images created that is later layed out and printed in grid 
                # randn returns random numbers from standard normal distribution; then inference decodes each distribution
                # under conditional, labels/attributes concatenated to help with decoding under supervision


                plt.figure()
                plt.figure(figsize=(5, 10))
                for p in range(10):
                    plt.subplot(5, 2, p+1) #row, col, index starting from 1
                    if args.conditional:
                        if args.dataset=='mnist':
                            plt.text(
                                0, 0, "c={:d}".format(c[p].item()), color='black',
                                backgroundcolor='white', fontsize=8)
                        elif args.dataset=='celeba':
                            plt.text(
                                0, 0, "c={}".format(c[p]), color='black',
                                backgroundcolor='white', fontsize=8)

                    # this order is required for visualizing data; otherwise easy to mess up color channels
                    p = x[p].view(3,218,178)
                    image = torch.transpose(p,0,2)
                    image = torch.transpose(image,0,1)
                    plt.imshow(image.cpu().data.numpy())

                    plt.axis('off')

                if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                    if not(os.path.exists(os.path.join(args.fig_root))):
                        os.mkdir(os.path.join(args.fig_root))
                    os.mkdir(os.path.join(args.fig_root, str(ts)))

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                 "E{:d}||{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')

        if args.dataset == 'mnist':
            df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
            g = sns.lmplot(
                x='x', y='y', hue='label', data=df.groupby('label').head(100),
                fit_reg=False, legend=True)
            g.savefig(os.path.join(
                args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
                dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    #parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])  #(218, 178, 3) for celebA = 116412
    #parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])  
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')
    parser.add_argument("--dataset", type=str, default='mnist')

    args = parser.parse_args()

    main(args)
