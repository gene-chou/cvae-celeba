# cvae-celeba

an implementation of cvae that allows setting attributes for celeba and setting labels for mnist 

---

## Results 

---

## How to run 
reference data directory tree  
use functions in dataloader.py to select desired attributes (e.g. in this repo 10 were chosen) and corresponding images  
then run train.py with appropriate arguments  
when running inference, adjust array c to create image with desired attributes (change ~line 117 in train.py, currently c is random)  

---

### Credits
starting code with MNIST was taken from https://github.com/timbmg/VAE-CVAE-MNIST
