## ACUITY Model Zoo

ACUITY model zoo contains a set of popular neurarl-network models created or converted (from Caffe or Tensorflow) by ACUITY toolset.

### Model Viewer
ACUITY uses JSON format to represent a neural-network model, and we provide an [online model viewer](https://verisilicon.github.io/acuity-models/viewer/index.html) to help visualized data flow graphs. The model viewer is inspired by [netscope](http://ethereon.github.io/netscope/quickstart.html).

### Classification
 - [Alexnet](https://verisilicon.github.io/acuity-models/viewer/#../models/alexnet/alexnet.json)
 - [Inception-v1](https://verisilicon.github.io/acuity-models/viewer/#../models/inception_v1/inception_v1.json)
 - [Inception-v2](https://verisilicon.github.io/acuity-models/viewer/#../models/inception_v2/inception_v2.json)
 - [Inception-v3](https://verisilicon.github.io/acuity-models/viewer/#../models/inception_v3/inception_v3.json)
 - [Inception-v4](https://verisilicon.github.io/acuity-models/viewer/#../models/inception_v4/inception_v4.json)
 - [Mobilenet-v1](https://verisilicon.github.io/acuity-models/viewer/#../models/mobilenet_v1/mobilenet_v1.json)
 - [Mobilenet-v1-025](https://verisilicon.github.io/acuity-models/viewer/#../models/mobilenet_v1_025/mobilenet_v1_025.json)
 - [Nasnet-Large](https://verisilicon.github.io/acuity-models/viewer/#../models/nasnet_large/nasnet_large.json)
 - [Nasnet-Mobile](https://verisilicon.github.io/acuity-models/viewer/#../models/nasnet_mobile/nasnet_mobile.json)
 - [Resnet-50](https://verisilicon.github.io/acuity-models/viewer/#../models/resnet50/resnet50.json)
 - [Resnext-50](https://verisilicon.github.io/acuity-models/viewer/#../models/resnext50/resnext50.json)
 - [Senet-50](https://verisilicon.github.io/acuity-models/viewer/#../models/senet50/senet50.json)
 - [Squeezenet](https://verisilicon.github.io/acuity-models/viewer/#../models/squeezenet/squeezenet.json)
 - [VGG-16](https://verisilicon.github.io/acuity-models/viewer/#../models/vgg16/vgg16.json)
 - [Xception](https://verisilicon.github.io/acuity-models/viewer/#../models/xception/xception.json)

### Detection
 - [Faster-RCNN-ZF](https://verisilicon.github.io/acuity-models/viewer/#../models/faster_rcnn_zf/faster_rcnn_zf.json)
 - [Mobilenet-SSD](https://verisilicon.github.io/acuity-models/viewer/#../models/mobilenet_ssd/mobilenet_ssd.json)
 - [MTCNN PNet](https://verisilicon.github.io/acuity-models/viewer/#../models/mtcnn/mtcnn_pnet.json) [RNet](https://verisilicon.github.io/acuity-models/viewer/#../models/mtcnn/mtcnn_rnet.json) [ONet](https://verisilicon.github.io/acuity-models/viewer/#../models/mtcnn/mtcnn_onet.json) [LNet](https://verisilicon.github.io/acuity-models/viewer/#../models/mtcnn/mtcnn_lnet.json)
 - [SSD](https://verisilicon.github.io/acuity-models/viewer/#../models/ssd/ssd.json) 
 - [Tiny-YOLO](https://verisilicon.github.io/acuity-models/viewer/#../models/tiny_yolo/tiny_yolo.json)
 - [YOLO-v1](https://verisilicon.github.io/acuity-models/viewer/#../models/yolo_v1/yolo_v1.json)
 - [YOLO-v2](https://verisilicon.github.io/acuity-models/viewer/#../models/yolo_v2/yolo_v2.json)

### Segmentation
 - [ENET](https://verisilicon.github.io/acuity-models/viewer/#../models/enet/enet.json)
 - [SegNet](https://verisilicon.github.io/acuity-models/viewer/#../models/segnet/segnet.json)

## About ACUITY

Acuity is a python based neural-network framework built on top of Tensorflow, it provides a set of easy to use high level layer API as well as infrastructure for optimizing neural networks for deployment.

 - Importing from popular frameworks such as Caffe and Tensorflow 

 
   > AcuityNet natively supports Caffe and Tensorflow imports, although it can be expanded to other NN frameworks.  


 - Fixed Point Quantization  


   > AcuityNet provides accurate Fixed Point Quantization from floating point 32 with a calibration dataset and produces accuracy numbers before and after quantization for comparison  


 - Graph Optimization  


   > Neural-network graph optimization is performed to reduce graph complexity for inference, such as Layer Merging, Layer Removal and Layer Swapping  


   - Merge consective layers into dense layers, such as ConvolutionReluPool, FullyConnectedRelu, etc.   
   - Fold BatchNrom layers into Convolution  
   - Swap layer ordering when suitable to reduce output size  
   - Remove Concatenation layers  
   - Intelligent layer optimization when mathamatically equivalent  

 - Tensor Pruning  


   > Pruning neural networks tensors to remove ineffective connections and neurons to create sparse matrix  


 - Training and Validation  


   > Acuitynet provides capability to train and validate Neural Networks  


 - Inference Code Generator  


   > Generates OpenVX Neural Network inference code which can run on any OpenVX enabled platforms  



