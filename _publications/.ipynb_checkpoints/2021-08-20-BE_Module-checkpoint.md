---
title: "Boundary Enhancement Semantic Segmentation for Building Extraction from Remote Sensed Image"
permalink: /publication/BE_module/
# venue: 'Accepted'
# paperurl: '/files/pdf/research/Boundary_Enhancement__Accepted_.pdf'
# link: "https://arxiv.org/abs/2011.02972"
# github: https://github.com/hin1115/Boundary_Enhanced_Building_Extraction
# citation: 'Hoin Jung, Han-Soo Choi, Myungjoo Kang, IEEE Transactions on Geoscience and Remote sensing'
tags:
  - references
  - bash
---

### Abstract
 Image processing via convolutional neural network(CNN) has been developed rapidly for remote sensing technology. Moreover, techniques for accurately extracting building footprints from remote sensed images have attracted considerable interest owing to their wide variety of common applications, including monitoring natural disasters and urban development. Extraction of building footprints can be performed easily by semantic segmentation using U-Net-like CNN architectures. However, obtaining precise boundaries of segmentation masks remains challenging due to various impediments surrounding target objects. In this study, we propose a method to elaborate edges of buildings detected in remote sensed images to enhance the boundaries of segmentation masks. The proposed method adopts *holistically-nested edge detection(HED)*, which extracts edge features at an encoder of a given architecture. In the proposed *boundary enhancement(BE)* module, an extracted edge and segmentation mask are combined, sharing mutual information. To enable the proposed method efficiently to adapt to a wide variety of conditions, we design a distinctive approach adopting a HED-unit and BE module, which is applicable to various semantic segmentation networks containing encoder-decoder structures. Experiments were conducted on five different datasets (DeepGlobe, Urban3D, WHU(HR, LR), and Massachusetts). The results demonstrate that our proposed approaches improved on the performance of prior methods for extracting building footprints. Comparative experiments were conducted on various backbone architectures including U-Net, ResUNet++, TernausNet, and USPP to ensure the effectiveness of the proposed method. Based on various evaluation metrics and qualitative analysis, our results show that the proposed method achieved improved performance compared to prior methods for all datasets and backbone networks.