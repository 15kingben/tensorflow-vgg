# Transfer Learning on VGG-16

This project is intended to be a plug-and-play setup for transfer learning with tensorflow and vgg that hopefully someone will find useful. It is based on the repository https://github.com/machrisaa/tensorflow-vgg.

>To use the VGG networks, the npy files for [VGG16 NPY](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) has to be downloaded.

## Usage
Edit train_vgg16.py to change the data_dirs variable to the directories containing your images and their labels. (0 being the first label up to the number of classes.

Next change the output path as desired. Other variables can be changed such as the batch size and frequency of saving copies.

python train_vgg16.py will train the network

python run_vgg16.py will give the network's predictions on a directory of images.
