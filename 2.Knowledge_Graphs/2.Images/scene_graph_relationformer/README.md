### KG using Relationformer
The code here uses [relationformer](https://github.com/suprosanna/relationformer?tab=readme-ov-file) framework for scene graph generation

you can also see [this](https://github.com/anant37289/relationformer), i've made some changes to the code to adapth to higher pytorch version.

---

For detailed setup refer to the original repository

For checkpoints see [this](https://drive.google.com/file/d/1Q1Nfvi2Frro6aVqRR8M_mD4OuZ6yhNRv/view?usp=sharing). I only trained this for 5 epochs and i also change the model dimension to fit my 12GB gpus. I have added my custom configs and the cuda 11.2 conda environent for trainig. The python version is 3.9.19.

If you wish to train here is the completely [compiled dataset to run traing](https://drive.google.com/drive/folders/1-77DLCL__TBx7PxvMK73Q6s49fcu5XXw?usp=sharing). Ater downloading unzip the images into VG_100K folder.

