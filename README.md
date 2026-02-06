## Acuity Model Zoo

Acuity model zoo contains a set of popular neural-network models created or converted (from Caffe, Tensorflow, PyTorch, TFLite, DarkNet or ONNX) by Acuity toolkits.

### Model Viewer
Acuity uses JSON format to describe a neural-network model, and we provide an [online model viewer](https://verisilicon.github.io/acuity-models/viewer/index.html) to help visualized data flow graphs. The model viewer is part of [netron](https://github.com/lutzroeder/netron) since 4.6.8.

### Classification
 - [Alexnet][]([OriginModel][OriginAlexnet])
 - [Inception-v1][] ([OriginModel][OriginInception-v1])
 - [Inception-v2][] ([OriginModel][OriginInception-v2])
 - [Inception-v3][] ([OriginModel][OriginInception-v3])
 - [Inception-v4][] ([OriginModel][OriginInception-v4])
 - [Mobilenet-v1][] ([OriginModel][OriginMobilenet-v1])
 - [Mobilenet-v2][] ([OriginModel][OriginMobilenet-v2])
 - [Mobilenet-v3][] ([OriginModel][OriginMobilenet-v3])
 - [EfficientNet][] ([OriginModel][OriginEfficientNet])
 - [EfficientNet (EdgeTPU)][] ([OriginModel][OriginEfficientNet-EdgeTPU])
 - [Nasnet-Large][] ([OriginModel][OriginNasnet-Large])
 - [Nasnet-Mobile][] ([OriginModel][OriginNasnet-Mobile])
 - [Resnet-50][] ([OriginModel][OriginResnet-50])
 - [Resnext-50][] ([OriginModel][OriginResnext-50])
 - [Senet-50][] ([OriginModel][OriginSenet-50])
 - [Squeezenet][] ([OriginModel][OriginSqueezenet])
 - [VGG-16][] ([OriginModel][OriginVGG-16])
 - [Xception][] ([OriginModel][OriginXception])
 - [DenseNet][] ([OriginModel][OriginDenseNet])

### Detection
 - [Faster-RCNN-ZF][] ([OriginModel][OriginFaster-RCNN-ZF])
 - [Mobilenet-SSD][] ([OriginModel][OriginMobilenet-SSD])
 - [Mobilenet-SSD-FPN][] ([OriginModel][OriginMobilenet-SSD-FPN])
 - [MTCNN PNet][]([OriginModel][OriginMTCNN PNet]) [RNet][]([OriginModel][OriginRNet]) [ONet][]([OriginModel][OriginONet]) [LNet][]([OriginModel][OriginLNet])
 - [SSD][] ([OriginModel][OriginSSD])
 - [Tiny-YOLO][] ([OriginModel][OriginTiny-YOLO])
 - [YOLO-v1][] ([OriginModel][OriginYOLO-v1])
 - [YOLO-v2][] ([OriginModel][OriginYOLO-v2])
 - [YOLO-v3][] ([OriginModel][OriginYOLO-v3])
 - [YOLO-v4][] ([OriginModel][OriginYOLO-v4])
 - [YOLO-v5][] ([OriginModel][OriginYOLO-v5])
 - [YOLO-v6][] ([OriginModel][OriginYOLO-v6])
 - [YOLO-v7][] ([OriginModel][OriginYOLO-v7])
 - [YOLO-v8][] ([OriginModel][OriginYOLO-v8])
 - [YOLO-v9][] ([OriginModel][OriginYOLO-v9])
 - [YOLO-v10][] ([OriginModel][OriginYOLO-v10])
 - [YOLO-v11][] ([OriginModel][OriginYOLO-v11])
 - [YOLO-v12][] ([OriginModel][OriginYOLO-v12])
 - [YOLO-v13][] ([OriginModel][OriginYOLO-v13])
 - [FCOS][] ([OriginModel][OriginFCOS])

### Face Recognition
 - [ArcFace][] ([OriginModel][OriginArcFace])
 - [UltraFace][] ([OriginModel][OriginUltraFace])

### Segmentation
 - [ENET][] ([OriginModel][OriginENET])
 - [SegNet][] ([OriginModel][OriginSegNet])
 - [DeepLab-v3][] ([OriginModel][OriginDeepLab-v3])

### Super Resolution
 - [SRCNN][] ([OriginModel][OriginSRCNN])
 - [VDSR][] ([OriginModel][OriginVDSR])
 - [EDSR_x2][] ([OriginModel][OriginEDSR_x2])
 - [EDSR_x3][] ([OriginModel][OriginEDSR_x3])
 - [EDSR_x4][] ([OriginModel][OriginEDSR_x4])
 - [ESRGAN][] ([OriginModel][OriginESRGAN])

### Image Denoising
 - [SID][] ([OriginModel][OriginSID])

### Pixel Processing
 - [Fast Style Transfer][] ([OriginModel][OriginFast Style Transfer])
 - [Pix2Pix][] ([OriginModel][OriginPix2Pix])

### Voice Processing
 - [QuartzNet][] ([OriginModel][OriginQuartzNet])
 - [DPRNN][] ([OriginModel][OriginDPRNN])
 - [RNNOISE][] ([OriginModel][OriginRNNOISE])
 - [Speaker Verification][] ([OriginModel][OriginSpeaker Verification])
 - [DS_CNN][] ([OriginModel][OriginDS_CNN])

### Pose Estimation
 - [Open Pose][] ([OriginModel][OriginOpen Pose])
 - [CPM Mobile][] ([OriginModel][OriginCPM Mobile])
 - [MSPN][] ([OriginModel][OriginMSPN])

### Age Estimation
 - [SSRNet][] ([OriginModel][OriginSSRNet])

### Recurrent Net
 - [LSTM - Command Recognition][]
 - [LSTM - Speech Recognition][]
 - [RNN-T Encoder][] [RNN-T Decoder][]

### Transformer
 - [BERTBase][] ([OriginModel][OriginBERTBase])
 - [ViT][] ([OriginModel][OriginViT])
 - [Swin-Transformer][] ([OriginModel][OriginSwin-Transformer])

### Large Language Model
 - **LLM-Chat**
   - **Qwen**
     - [Qwen2.5-0.5B-Instruct][]
     - [Qwen2.5-1.5B-Instruct][]
     - [Qwen2.5-3B-Instruct][]
   - **MiniCPM**
     - [MiniCPM-1B-sft-bf16][]
   - **LLaMa**
     - [Llama3.2-1B-Instruct][]
     - [Llama-2-7b-chat-hf][]
   - **DeepSeek**
     - [DeepSeek-R1-Distill-Qwen-1.5B][]
     - [DeepSeek-R1-Distill-Llama-8B][]
   - **Gemma**
     - [gemma-2-2b-it][]
     - [gemma-3-1b-it][]
 - **Audio to Text**
   - **Whisper**
     - [whisper-small][]
     - [whisper-tiny][]
     - [whisper-large-v3-turbo][]
 - **VLM**
   - **Llava**
     - [llava-1.5-7b-hf][]
   - **FastVLM**
     - [FastVLM-0.5B][]
   - **Qwen**
     - [Qwen2.5-VL-3B-Instruct][]
 - **TTS**
   - **OuteTTS**
     - [Llama-OuteTTS-1.0-1B][]


## About Acuity

Acuity is a python based neural-network framework built on top of Tensorflow, it provides a set of easy to use high level layer API as well as infrastructure for optimizing neural networks for deployment on [Vivante Neural Network Processor IP](http://www.verisilicon.com/en/IPPortfolio/VivanteNPUIP) powered hardware platforms. Going from a pre-trained model to hardware inferencing can be as simple as 3 automated steps.

![Acuity Workflow](/docs/acuity_123.png)


 - Importing from popular frameworks such as Tensorflow and PyTorch

   > AcuityNet natively supports Caffe, Tensorflow, PyTorch, ONNX, TFLite, DarkNet, and Keras imports, it can also be expanded to support other NN frameworks.

 - Fixed Point Quantization

   > AcuityNet provides accurate Post Training Quantization and produces accuracy numbers before and after quantization for comparison. Advanced techniques are built-in into AcuityNet quantizer, such as KL-Divergence, Weight Equalization, Hybrid Quantization, Per Channel Quantization, etc.

 - Graph Optimization

   > Neural-network graph optimization is performed to reduce graph complexity for inference, such as Layer Fusion, Layer Removal and Layer Swapping

 - Tensor Pruning

   > Pruning neural networks tensors to remove ineffective synapses and neurons to create sparse matrix

 - Training and Validation

   > AcuityNet provides capability to train and validate Neural Networks

 - Inference Code Generator

   > Generates OpenVX Neural Network inference code which can run on any OpenVX enabled platforms

## About Vivante NPUIP

Vivante NPUIP is a highly scalable and programmable neural network processor that supports a wide range of Machine Learning applications. It has been deployed in many fields to accelerate ML algorithms for AI-vision, AI-voice, AI-pixel and other special use cases. Vivante NPUIP offers high performance MAC engine as well as flexible programmable capability to adopt new operations and networks without having to fall back to CPU. Today, over 120 operators are supported and continue to grow.

Mature software stack and complete solutions are provided to customers for easy integration and fast time to market.

Tooling
 - Acuity Toolkits
 - Acuity IDE

Runtime software stack support
 - OpenVX and OpenVX NN Extension
 - OpenCL
 - Android NN API
 - TFLite NPU Delegate
 - ONNX Runtime Execution Provider
 - ARMNN Backend

[Alexnet]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/alexnet/alexnet.json
[Inception-v1]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/inception_v1/inception_v1.json
[Inception-v2]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/inception_v2/inception_v2.json
[Inception-v3]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/inception_v3/inception_v3.json
[Inception-v4]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/inception_v4/inception_v4.json
[Mobilenet-v1]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/mobilenet_v1/mobilenet_v1.json
[Mobilenet-v2]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/mobilenet_v2/mobilenet_v2.json
[Mobilenet-v3]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/mobilenet_v3/mobilenet_v3.json
[EfficientNet]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/efficientnet_b0/efficientnet_b0.json
[EfficientNet (EdgeTPU)]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/efficientnet_edgetpu/efficientnet_edgetpu.json
[Nasnet-Large]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/nasnet_large/nasnet_large.json
[Nasnet-Mobile]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/nasnet_mobile/nasnet_mobile.json
[Resnet-50]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/resnet50/resnet50.json
[Resnext-50]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/resnext50/resnext50.json
[Senet-50]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/senet50/senet50.json
[Squeezenet]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/squeezenet/squeezenet.json
[VGG-16]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/vgg16/vgg16.json
[Xception]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/xception/xception.json
[Faster-RCNN-ZF]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/faster_rcnn_zf/faster_rcnn_zf.json
[Mobilenet-SSD]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/mobilenet_ssd/mobilenet_ssd.json
[Mobilenet-SSD-FPN]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/mobilenet_ssd_fpn/mobilenet_ssd_fpn.json
[MTCNN PNet]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/mtcnn/mtcnn_pnet.json
[RNet]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/mtcnn/mtcnn_rnet.json
[ONet]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/mtcnn/mtcnn_onet.json
[LNet]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/mtcnn/mtcnn_lnet.json
[SSD]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/ssd/ssd.json
[Tiny-YOLO]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/tiny_yolo/tiny_yolo.json
[YOLO-v1]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v1/yolo_v1.json
[YOLO-v2]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v2/yolo_v2.json
[YOLO-v3]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v3/yolo_v3.json
[YOLO-v4]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v4/yolo_v4.json
[YOLO-v5]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v5/yolo_v5.json
[YOLO-v6]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v6/yolo_v6.json
[YOLO-v7]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v7/yolo_v7.json
[YOLO-v8]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v8/yolo_v8.json
[YOLO-v9]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v9/yolo_v9.json
[YOLO-v10]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v10/yolo_v10.json
[YOLO-v11]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v11/yolo_v11.json
[YOLO-v12]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v12/yolo_v12.json
[YOLO-v13]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/yolo_v13/yolo_v13.json
[ENET]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/enet/enet.json
[SegNet]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/segnet/segnet.json
[DeepLab-v3]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/deeplab_v3/deeplab_v3.json
[Fast Style Transfer]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/fast_style_transfer/fast_style_transfer.json
[Pix2Pix]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/pix2pix/pix2pix.json
[Open Pose]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/open_pose/open_pose.json
[CPM Mobile]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/cpm/cpm.json
[LSTM - Command Recognition]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/lstm/lstm.json
[LSTM - Speech Recognition]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/deepspeech2/deepspeech2.json
[QuartzNet]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/quartznet/quartznet.json
[DPRNN]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/dprnn/dprnn.json
[RNNOISE]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/rnnoise/rnnoise.json
[Speaker Verification]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/speaker_verification/speaker_verification.json
[DS_CNN]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/ds_cnn/ds_cnn.json
[MSPN]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/mspn/mspn.json
[FCOS]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/fcos/fcos.json
[SSRNet]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/ssrnet/ssrnet.json
[DenseNet]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/densenet/densenet.json
[ArcFace]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/arcface/arcface.json
[UltraFace]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/ultraface/ultraface.json
[SRCNN]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/srcnn/srcnn.json
[VDSR]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/vdsr/vdsr.json
[EDSR_x2]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/edsr/edsr_x2.json
[EDSR_x3]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/edsr/edsr_x3.json
[EDSR_x4]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/edsr/edsr_x4.json
[ESRGAN]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/esrgan/esrgan.json
[SID]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/sid/sid.json
[RNN-T Encoder]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/rnn-t/encoder_layernorm.json
[RNN-T Decoder]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/rnn-t/decoder_cifg.json
[BERTBase]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/bert_base/bert_base_vsi_frozen.json
[ViT]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/vit/vit.json
[Swin-Transformer]: https://verisilicon.github.io/acuity-models/viewer/?url=../models/swin_transformer/swin_transformer.json

[OriginAlexNet]: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
[OriginInception-v1]: http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
[OriginInception-v2]: http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz
[OriginInception-v3]: http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
[OriginInception-v4]: http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
[OriginMobilenet-v1]: http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
[OriginMobilenet-v2]: https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz
[OriginMobilenet-v3]: https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz
[OriginEfficientNet]: https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b0.tar.gz
[OriginEfficientNet-EdgeTPU]: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu
[OriginNasnet-Large]: https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz
[OriginNasnet-Mobile]: https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz
[OriginResnet-50]: http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
[OriginResnext-50]: https://dl.fbaipublicfiles.com/resnext/imagenet_models/resnext_50_32x4d.t7
[OriginSenet-50]: https://github.com/hujie-frank/SENet
[OriginSqueezenet]: https://github.com/BVLC/caffe/wiki/Model-Zoo#squeezenet-alexnet-level-accuracy-with-50x-fewer-parameters
[OriginVGG-16]: http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
[OriginXception]: https://drive.google.com/file/d/1sJCRDhaNaJAnouKKulB3YO8Hu3q91KjP/view?usp=sharing
[OriginFaster-RCNN-ZF]: https://github.com/rbgirshick/fast-rcnn#extra-downloads
[OriginMobilenet-SSD]: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
[OriginMobilenet-SSD-FPN]: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
[OriginMTCNN PNet]: https://github.com/imistyrain/MTCNN/tree/master/model/caffe
[OriginRNet]: https://github.com/imistyrain/MTCNN/tree/master/model/caffe
[OriginONet]: https://github.com/imistyrain/MTCNN/tree/master/model/caffe
[OriginLNet]: https://github.com/imistyrain/MTCNN/tree/master/model/caffe
[OriginSSD]: https://github.com/weiliu89/caffe/tree/ssd#models
[OriginTiny-YOLO]: https://drive.google.com/file/d/14-5ZojD1HSgMKnv6_E3WUcBPxaVm52X2/view?usp=sharing
[OriginYOLO-v1]: https://pjreddie.com/media/files/yolov1.weights
[OriginYOLO-v2]: https://pjreddie.com/media/files/yolov2.weights
[OriginYOLO-v3]: https://pjreddie.com/media/files/yolov3.weights
[OriginYOLO-v4]: https://github.com/AlexeyAB/darknet/#pre-trained-models
[OriginYOLO-v5]: https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt
[OriginYOLO-v6]: https://github.com/meituan/YOLOv6
[OriginYOLO-v7]: https://github.com/WongKinYiu/yolov7
[OriginYOLO-v8]: https://github.com/ultralytics/ultralytics/releases/tag/v8.0.4
[OriginYOLO-v9]: https://github.com/WongKinYiu/yolov9
[OriginYOLO-v10]: https://github.com/THU-MIG/yolov10
[OriginYOLO-v11]: https://github.com/ultralytics/ultralytics/tree/v8.3.202
[OriginYOLO-v12]: https://github.com/sunsmarterjie/yolov12
[OriginYOLO-v13]: https://github.com/iMoonLab/yolov13
[OriginENET]: https://github.com/TimoSaemann/ENet
[OriginSegNet]: https://github.com/BVLC/caffe/wiki/Model-Zoo#segnet-and-bayesian-segnet
[OriginDeepLab-v3]: https://github.com/tensorflow/models/tree/master/research/deeplab
[OriginFast Style Transfer]: https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ?usp=sharing
[OriginPix2Pix]: https://github.com/affinelayer/pix2pix-tensorflow
[OriginOpen Pose]: https://github.com/CMU-Perceptual-Computing-Lab/openpose
[OriginCPM Mobile]: https://drive.google.com/open?id=1gOwBY5puCusYPCQaPcEUMmQtPnGHCPyl
[OriginLSTM - Speech Recognition]: https://github.com/tensorflow/models/tree/master/research/deep_speech
[OriginQuartzNet]: https://github.com/NVIDIA/NeMo/blob/main/docs/source/asr/configs.rst
[OriginDPRNN]: https://github.com/sp-uhh/dual-path-rnn
[OriginRNNOISE]: https://github.com/xiph/rnnoise
[OriginSpeaker Verification]: https://github.com/HarryVolek/PyTorch_Speaker_Verification
[OriginDS_CNN]: https://github.com/ARM-software/ML-KWS-for-MCU/tree/master/Pretrained_models/DS_CNN
[OriginMSPN]: https://github.com/megvii-detection/MSPN
[OriginFCOS]: https://github.com/tianzhi0549/FCOS
[OriginSSRNet]: https://github.com/shamangary/SSR-Net/tree/master/pre-trained
[OriginDenseNet]: https://github.com/shicai/DenseNet-Caffe
[OriginArcFace]: https://github.com/deepinsight/insightface/wiki/Model-Zoo
[OriginUltraFace]: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
[OriginSRCNN]: https://github.com/tegg89/SRCNN-Tensorflow/tree/master/checkpoint/srcnn_21
[OriginVDSR]: https://cv.snu.ac.kr/research/VDSR/VDSR_code.zip
[OriginEDSR_x2]: https://cv.snu.ac.kr/research/EDSR/model_pytorch.tar
[OriginEDSR_x3]: https://cv.snu.ac.kr/research/EDSR/model_pytorch.tar
[OriginEDSR_x4]: https://cv.snu.ac.kr/research/EDSR/model_pytorch.tar
[OriginESRGAN]: https://github.com/xinntao/ESRGAN
[OriginSID]: https://github.com/cchen156/Learning-to-See-in-the-Dark/blob/master/download_models.py
[OriginBERTBase]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
[OriginViT]: https://github.com/google-research/vision_transformer
[OriginSwin-Transformer]: https://github.com/microsoft/Swin-Transformer
[Qwen2.5-0.5B-Instruct]: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
[Qwen2.5-1.5B-Instruct]: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
[Qwen2.5-3B-Instruct]: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
[MiniCPM-1B-sft-bf16]: https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16
[Llama3.2-1B-Instruct]: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
[Llama-2-7b-chat-hf]: https://huggingface.co/NousResearch/Llama-2-7b-chat-hf
[DeepSeek-R1-Distill-Qwen-1.5B]: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
[DeepSeek-R1-Distill-Llama-8B]: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
[gemma-2-2b-it]: https://huggingface.co/google/gemma-2-2b-it
[gemma-3-1b-it]: https://huggingface.co/google/gemma-3-1b-it
[whisper-small]: https://huggingface.co/openai/whisper-small
[whisper-tiny]: https://huggingface.co/openai/whisper-tiny
[whisper-large-v3-turbo]: https://huggingface.co/openai/whisper-large-v3-turbo
[llava-1.5-7b-hf]: https://huggingface.co/llava-hf/llava-1.5-7b-hf
[FastVLM-0.5B]: https://huggingface.co/apple/FastVLM-0.5B
[Qwen2.5-VL-3B-Instruct]: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
[Llama-OuteTTS-1.0-1B]: https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B
