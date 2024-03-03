# training mnist
from fastai.vision.all import * 
from fastbook import * 
matplotlib.rc('image', cmap='Greys')

#================================================================
# 1. Load data
#================================================================

path = untar_data(URLs.MNIST)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#================================================================
# 2. Load data into dataloaders
#================================================================

# define dataloaders
mnist = DataBlock(blocks=(ImageBlock, CategoryBlock),
                    get_items=get_image_files,
                    splitter=GrandparentSplitter(train_name='training', valid_name='testing'),
                    get_y=parent_label)
dls = mnist.dataloaders(path, device=device)
# get a batch of data
# dls.show_batch(max_n=9, figsize=(4,4))

#================================================================
# 3. Training
#================================================================
# simple_net = nn.Sequential(
#     # flatten the input
#     nn.Flatten(1),
#     nn.Linear(28*28*3,30),
#     nn.ReLU(),
#     nn.Linear(30,10)
# )

bn = True
def conv(ni, nf, ks=3, act=True): 
    layers = []
    layers.append(nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks//2))
    if act: layers.append(nn.ReLU())
    if bn: layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)

def simple_cnn():
    return nn.Sequential(
        conv(3, 8, ks=5), #14
        conv(8, 16), #7
        conv(16, 32), #4
        conv(32, 32), #2
        conv(32, 10, act=False), #1
        nn.Flatten()
    )

learn = Learner(dls, simple_cnn(), opt_func=SGD,
                loss_func=F.cross_entropy,
                metrics=accuracy)
learn.fit_one_cycle(5, 0.1)

# save the model
learn.save('mnist_cnn')