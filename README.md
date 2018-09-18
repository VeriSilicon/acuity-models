## Acuity Model Zoo

Acuity model zoo contains a set of popular neural-network models created or converted (from Caffe, Tensorflow, TFLite, DarkNet or ONNX) by Acuity toolset.

### Model Viewer
Acuity uses JSON format to describe a neural-network model, and we provide an [online model viewer](https://verisilicon.github.io/acuity-models/viewer/index.html) to help visualized data flow graphs. The model viewer is inspired by [netscope](http://ethereon.github.io/netscope/quickstart.html).

### Classification
 - [Alexnet](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/alexnet/alexnet.json)
 - [Inception-v1](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/inception_v1/inception_v1.json)
 - [Inception-v2](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/inception_v2/inception_v2.json)
 - [Inception-v3](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/inception_v3/inception_v3.json)
 - [Inception-v4](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/inception_v4/inception_v4.json)
 - [Mobilenet-v1](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/mobilenet_v1/mobilenet_v1.json)
 - [Mobilenet-v2](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/mobilenet_v2/mobilenet_v2.json)
 - [Nasnet-Large](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/nasnet_large/nasnet_large.json)
 - [Nasnet-Mobile](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/nasnet_mobile/nasnet_mobile.json)
 - [Resnet-50](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/resnet50/resnet50.json)
 - [Resnext-50](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/resnext50/resnext50.json)
 - [Senet-50](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/senet50/senet50.json)
 - [Squeezenet](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/squeezenet/squeezenet.json)
 - [VGG-16](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/vgg16/vgg16.json)
 - [Xception](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/xception/xception.json)

### Detection
 - [Faster-RCNN-ZF](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/faster_rcnn_zf/faster_rcnn_zf.json)
 - [Mobilenet-SSD](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/mobilenet_ssd/mobilenet_ssd.json)
 - [MTCNN PNet](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/mtcnn/mtcnn_pnet.json) [RNet](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/mtcnn/mtcnn_rnet.json) [ONet](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/mtcnn/mtcnn_onet.json) [LNet](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/mtcnn/mtcnn_lnet.json)
 - [SSD](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/ssd/ssd.json) 
 - [Tiny-YOLO](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/tiny_yolo/tiny_yolo.json)
 - [YOLO-v1](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/yolo_v1/yolo_v1.json)
 - [YOLO-v2](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/yolo_v2/yolo_v2.json)
 - [YOLO-v3](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/yolo_v3/yolo_v3.json)

### Segmentation
 - [ENET](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/enet/enet.json)
 - [SegNet](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/segnet/segnet.json)

### Pixel Processing
 - [Denoiser](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/denoiser/denoiser.json)
 - [Super Resolution](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/vdsr/vdsr.json)
 - [Fast Style Transfer](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/fast_style_transfer/fast_style_transfer.json)

### Recurrent Net
 - [LSTM - Command Recognition](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/lstm/lstm.json)
 - [LSTM - Speech Recognition](https://verisilicon.github.io/acuity-models/viewer/render.html#../models/deepspeech2/deepspeech2.json)

## About Acuity

Acuity is a python based neural-network framework built on top of Tensorflow, it provides a set of easy to use high level layer API as well as infrastructure for optimizing neural networks for deployment on [Vivante Vision IP](http://www.verisilicon.com/IPPortfolio_2_122_1_VisionIP.html) powered hardware platforms. Going from a pre-trained model to hardware inferencing can be as simple as 3 automated steps.

![Acuity Workflow](/docs/acuity_123.png)


 - Importing from popular frameworks such as Caffe and Tensorflow 

 
   > AcuityNet natively supports Caffe, Tensorflow, TFLite, DarkNet and ONNX imports, it can also be expanded to support other NN frameworks.  


 - Fixed Point Quantization  


   > AcuityNet provides accurate Fixed Point Quantization from floating point 32 with a calibration dataset and produces accuracy numbers before and after quantization for comparison  


 - Graph Optimization  


   > Neural-network graph optimization is performed to reduce graph complexity for inference, such as Layer Merging, Layer Removal and Layer Swapping  


   - Merge consective layers into dense layers, such as ConvolutionReluPool, FullyConnectedRelu, etc.   
   - Fold BatchNrom layers into Convolution  
   - Swap layer ordering when suitable to reduce output size  
   - Remove Concatenation and Split layers
   - Horizontal layer fusion   
   - Intelligent layer optimization when mathamatically equivalent

 - Tensor Pruning  


   > Pruning neural networks tensors to remove ineffective synapses and neurons to create sparse matrix  


 - Training and Validation  


   > Acuitynet provides capability to train and validate Neural Networks  


 - Inference Code Generator  


   > Generates OpenVX Neural Network inference code which can run on any OpenVX enabled platforms  



