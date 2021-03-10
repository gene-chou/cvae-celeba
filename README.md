# cvae-celeba

an implementation of cvae that allows setting attributes for celeba and setting labels for mnist 

---

## Results 

---

## How to run 
data directory tree:  
data/  
&nbsp;&nbsp;celeba/  
&nbsp;&nbsp;&nbsp;&nbsp;selected_list_attr_celeba.txt/  
&nbsp;&nbsp;&nbsp;&nbsp;original_list_attr_celeba.txt/  
&nbsp;&nbsp;&nbsp;&nbsp;img_align_celeba/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;selected_images/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;original_images/  
  
use functions in dataloader.py to select desired attributes (e.g. in this repo 10 were chosen) and corresponding images  
then run train.py with appropriate arguments  
when running inference, adjust array c to create image with desired attributes (change ~line 117 in train.py, currently c is random)  
setting reduction parameter in BCE loss to "sum" rather than "mean" produces high loss but better reconstruction results due to emphasis on reconstruction rather than divergence of distribution  

---

## Credits
starting code with MNIST was taken from https://github.com/timbmg/VAE-CVAE-MNIST
