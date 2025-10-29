# Brain MRI Segmentation with Improved UNet

## Description
This project implements a UNet based convolutional neural network for automatic segmentation of brain structures in MRI scans using the OASIS dataset. The problem addressed is medical image segmentation**, where each pixel in an MRI slice must be classified into anatomical categories. Accurate segmentation is essential for clinical diagnosis and neuroimaging research.

The algorithm works by passing input MRI slices through a U-Net encoder–decoder network with skip connections, which preserves spatial information while learning high-level features. The network outputs pixel-wise class probabilities that are compared to ground-truth masks using cross-entropy loss and evaluated using the dice coefficient.

## How it Works
### Preprocessing and Augmentation ###
This step ensures that the MRI data is prepared for robust training. Each MRI slice is loaded as a grayscale image and normalized to values between 0 and 1. Segmentation masks are mapped into class IDs, where 0 represents background and 1 and 2 represent different tissue classes. To improve generalization, data augmentation techniques such as horizontal flipping, small random rotations of up to +-5 degrees, brightness and contrast jittering, and Gaussian noise injection are applied.

### Model ###
The model architecture is based on UNet, a popular convolutional neural network for biomedical image segmentation. The encoder repeatedly downsamples the input using convolutional layers combined with Instance Normalization and ReLU activations. At the bottleneck, the model encodes the deepest feature representation of the input slice. The decoder then reconstructs the segmentation map through upsampling with transposed convolutions and incorporates skip connections from the encoder to preserve spatial details. Each block includes residual connections to stabilize training, and the final 1×1 convolution maps the features into three output classes.

### Training ###
Training is done using the cross-entropy loss funciton and an Adam optimizer with learning rate of 0.00001. The primary evaluation metric during training is the mean dice coefficient across all classes, which measures overlap between predictions and ground truth. At each epoch, validation dice scores are monitored, and the best-performing model checkpoint is saved automatically.

### Evaluation ###
Here, the trained model is tested on a test dataset, and outputs the mean dice coefficient on that set. Additionally, predictions for several inputs are visualized alongside ground truth masks.