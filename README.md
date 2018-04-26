# FC<sup>4</sup>:<br> Fully Convolutional Color Constancy with Confidence-weighted Pooling
#### [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Hu_FC4_Fully_Convolutional_CVPR_2017_paper.pdf)]
[Yuanming Hu](http://taichi.graphics/me/)<sup>1,2</sup>, [Baoyuan Wang](https://sites.google.com/site/zjuwby/)<sup>1</sup>, [Stephen Lin](https://www.microsoft.com/en-us/research/people/stevelin/)<sup>1</sup>

<sup>1</sup>Microsoft Research <sup>2</sup>Tsinghua University (now MIT CSAIL)


**Change log:**
 - April 15, 2018: Started preparing for code release. TODO:
   - Upgrade `python` version from `2.7` to `3.5+`. **Please use python 2.7 for now.**
   - Upgrade `tensorflow` version from `1.1` to `1.7`
   - Remove internal project code
   - Retrain on datasets to get a tensorflow `1.7` compatible model

<img src="web/images/teaser.png">

# FAQ
1) **Datasets**

1.a) **Where to get the datasets?**

*Shi's Re-processing of Gehler's Raw Dataset*: [here](http://www.cs.sfu.ca/~colour/data/shi_gehler/)

*NUS-8 Camera Dataset*: [here](http://www.comp.nus.edu.sg/~whitebal/illuminant/illuminant.html)

1.b) **The input images look purely black. What's happening?**

The input photos from the ColorChecker dataset are 16-bit `png` files and some image viewer may not support them, as `png`s are typically 8-bit. 
Also, since these photos are linear (RAW sensor activations) and modern displays have a `2.2` gamma value (instead of linear gamma), they will appear even darker when displayed. A exposure correction is also necessary.

1.c) **I corrected the gamma. Now most images appear green. Is there anything wrong?**

It's common that RAW images appear green. A possibility is that the color filters of digital cameras may have a stronger activation on the green channel.

2) **How to preprocess the data?**

[*Shi's Re-processing of Gehler's Raw Dataset*:](http://www.cs.sfu.ca/~colour/data/shi_gehler/)
 - Download the 4 zip files from the website
 - Extract the `png` images into `fc4/data/gehler/images/`, without creating subfolders.
 - `python dataset.py`, and wait for it to finish
 - `python show_patches.py` to view **data-augmented** patches. Press any key to see the next patch. You can use this data provider to train your own network.

# Bibtex
```
@inproceedings{hu2017fc,
  title={FC 4: Fully Convolutional Color Constancy with Confidence-weighted Pooling},
  author={Hu, Yuanming and Wang, Baoyuan and Lin, Stephen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4085--4094},
  year={2017}
}
```

# Related Research Projects and Implementations 
 - [Exposure](https://github.com/yuanming-hu/exposure) General-propose photo postprocessing
 - ...
 
# Acknoledgements 
 - The SqueezeNet model is taken from [here](https://github.com/DeepScale/SqueezeNet). Thank Yu Gu for his great efforts in converting the `Caffe` models into a `TensorFlow`-readable version! 
