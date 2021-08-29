# MultilingualGAN

#### The official source code for the paper *"Multilingual-GAN: A Multilingual GAN-based Approach for Handwritten Generation"*

**Abstract:** Handwritten Text Recognition (HTR) is a difficult problem because of the diversity of calligraphic styles. To enhance the accuracy of HTR systems, a large amount of training data is required. The previous methods aim at generating handwritten images from input strings via RNN models such as LSTM or GRU. However, these methods require a predefined alphabet corresponding to a given language. Thus, they can not well adapt to a new languages. To address this problem, we propose an Image2Image-based method named Multilingual-GAN, which translates a printed text image into a handwritten style one. The main advantage of this approach is that the model does notdepend on any language alphabets. Therefore, our model can be used on a new language without re-training on a new dataset.The quantitative results demonstrate that our proposed method outperforms other state-of-the-art models.

**1. Requirements**
- Python 3.6 or above
- Pytorch 1.6 or above
- Torchvision 0.7.0 or above

**2. Data preparation:** put all train data and test data on `data/train/` and `data/test/` respectively. Each sample should contain a pair samples between two domains as below

![0246_samples](https://user-images.githubusercontent.com/32817741/131226858-f00ef91b-24da-4f57-a1f2-1348a880975f.png)
![0247_samples](https://user-images.githubusercontent.com/32817741/131226875-3b084723-d9cf-4f44-b575-6c5060073790.png)
![0252_samples](https://user-images.githubusercontent.com/32817741/131226895-e3c5c061-f87f-4fa0-9d5d-087bac6dbb6e.png)

**3. Run the code**
- Check `config.yaml` file and change it to your desire configuration
- Run the command `python Image2Image.py` to start training with `config.yaml` configuration

**4. The overall of MultilingualGAN**

![framework_color](https://user-images.githubusercontent.com/32817741/131227019-2b462735-5231-4c67-a923-621a8bff7d60.png)

**5. Example results**
- On vietnamese

![Dataset_fig](https://user-images.githubusercontent.com/32817741/131227026-82095cbc-7384-4406-ae8f-5b5806f4f253.png)
- On multiple languages within one model

![other_language](https://user-images.githubusercontent.com/32817741/131227040-66b6e3c3-75e7-40ec-ba82-12e802e4f232.png)

**6. Quantitative result**

![image](https://user-images.githubusercontent.com/32817741/131227100-13359388-892c-4522-a6d0-c41f5ac0f71d.png)

