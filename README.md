# PointNet
PyTorch Implementation of PointNet paper.  
The [paper](https://arxiv.org/pdf/1612.00593.pdf).
Blog post: [here]().  

To train the model you have put the ModelNet10 or ModelNet40 datasets to dataset preprocessing that is explained in the blog post.  
Once you've done it run the following to train:  
```python train.py```  
``` --did ```: Device id to train the model on (0)  
```--lr```: Learning rate (0.001)  
```n_points```: Number of sampled points (1024)  
```n_class```: Number of classes (10)  
```batch_size```: Mini batch size (32)  
```epochs```: Number of epochs (100)  
```wd```: Weight decay (0)  
```dropout```: Dropout rate (0.3)  
