
# GlioAI: Automatic Brain Tumor Detection System for Diffusion-Weighted MRI

Feras Baig  
Vaibhav Shahi  
* fbaig23@students.d125.org
* vshahi23@students.d125.org  
 

  
  


## Contents

* [Abstract](https://github.com/ferasbg/GlioAI/blob/master/docs/Abstract.md#abstract)
* [Introduction](https://github.com/ferasbg/GlioAI/blob/master/docs/Abstract.md#introduction)
* [Method and Design](https://github.com/ferasbg/GlioAI/blob/master/docs/Abstract.md#method-and-design)
* [Results](https://github.com/ferasbg/GlioAI/blob/master/docs/Abstract.md#results)
* [Conclusion and Future Work](https://github.com/ferasbg/GlioAI/blob/master/docs/Abstract.md#conclusion-and-future-work)
* [FAQ](https://github.com/ferasbg/GlioAI/blob/master/docs/Abstract.md#faq)

## Abstract

In the 21st century, healthcare providers have been faced with the major challenge in brain tumor treatment planning and making a qualitative evaluation and determination of the presence and extent of the tumor in the patient. The non-invasive magnetic resonance imaging (MRI) technique has emerged as the primary diagnostic tool for brain tumors without any ionizing radiation. Manual image feature extraction methods of determining brain tumor extent from 3D MRI clusters is a very time-consuming task and is limited to operator experience, and is prone to human error. Despite advancements in chemotherapy methods and treatment execution, there are still areas in the clinical treatment pipeline including tools to build for cancer detection that need to be optimized for better patient results in terms of survivability and transition back to civilian life. The recent progress in computer vision and deep learning may be exploited to achieve automatic detection of tumorous nodules to provide accurate clinical information and reduce cognitive workload for the radiologists. Given this context, we propose a fully-automated method for brain tumor detection which is developed using a VGG-based deep convolutional neural network. Our method was evaluated using crowdsourced data in order to assist physicians around the globe by detecting brain-related cancers via the web, available entirely free for use. Cross-validation has shown that our method can obtain promising image classification efficiently.

## Introduction

Malignant brain tumors are one of the most deadly forms of cancer, partially due to the dreadful diagnosis, but also because of the direct consequences on decreased cognitive function and lasting adverse impact on the quality of life of the patient. The most frequent primary brain tumors in adults are primary central nervous system lymphomas and gliomas, which account for almost 80% of malignant cases. T1 and T2 refer to the difference between the weights of the MR images which highlight different feature regions that the radiologist wants to call attention to. T2 weighted images are optimized to deliver better resolution results for tumors. Our system has been trained with both types of weights in order to accomodate both imaging techniques. Manual MRI image feature extraction methods are very time consuming, limited to operator experience, and are prone to human error. Simply put, radiologists and oncologists cannot control the variable of diagnostic error because of the limited time to make an accurate assessment of the MRI images of the patients. In some cases even for highly trained radiologists, detecting tumorous nodules from MRI scans and formulating a cancer diagnosis become challenging tasks. To add, MRI reports alone take over 2 weeks to be examined with patients, a time duration too long for taking action to eliminate malignant cancers if present.  The focus of the patient can be fully restored without friction, cementing the importance of quality-first healthcare. Given this context, we propose a fully-automated brain tumor detection system that was built using crowdsourced data in order to assist physicians around the globe by detecting brain-related cancers via the web, available entirely free for use.

# Method and Design
* [Finding Optimal Neural Network Configuration](https://github.com/ferasbg/GlioAI/blob/master/docs/Abstract.md#20-optimizing-neural-network-configuration)
* [Data Acquisition and Data Pre-Processing ](https://github.com/ferasbg/GlioAI/blob/master/docs/Abstract.md#21-data-acquisition-and-data-pre-processing)
* [Data Augmentation](https://github.com/ferasbg/GlioAI/blob/master/docs/Abstract.md#22-data-augmentation)
* [Preparing the Data](https://github.com/ferasbg/GlioAI/blob/master/docs/Abstract.md#23-preparing-the-data)
* [Training the Model](https://github.com/ferasbg/GlioAI/blob/master/docs/Abstract.md#24-training-the-model)

### 2.0. Optimizing Neural Network Configuration

In order to build a system that yields high accuracy, the neural network that will be implemented in the back-end must be configured to the specific image data and detect the image at target accuracy of greater than 85-90%. When dealing with an image recognition task involving data in the context of medical imaging, data acquisition is limited to <1000 labeled images, so we have resorted to crowdsourced alternatives. Although, there is a lot that can be done in terms of optimizing the current situation with the limited set of MRI images. Given the context of our problem, we test our hypothesis by training and testing two different neural networks (VGG-Net vs. GlioNet) under the same control variables (same GPU, train/test split ratio, number of epochs, batching data). We wanted to determine whether or not the implementation of transfer learning would yield better tumor detection performance, versus a neural network that is custom-built and optimized for the task but does not yield the same level of training done with different types of image data that was done with the pretrained convolutional neural network.

### 2.1. Data Acquisition and Data Pre-Processing 
 
We acquired a crowdsourced dataset from Google Open-Source Datasets, split the dataset so that 239 images were used in order for training + cross-validation, 14 test images were put in a separate folder and was not touched until the model was finished training. Loaded images into the development environment and set it with a directory. Because the image dimensions were slightly different, I applied a transformation so that all the images in the dataset were 224 by 224, and set the color parameter to RGB.
 
 
### 2.2. Data Augmentation
 
In order to maximize the usage of our low-volume dataset, we will apply data augmentation methods via a number of random transformations, so that the model would never see twice the exact same picture. This helps prevent overfitting and helps the model generalize better. We used the keras.preprocessing.image.ImageDataGenerator class from Keras. The target was to apply random transformations and normalization operations to the image data during training. Because we are dealing with a small dataset, there is a possibility of overfitting because the model can make generalizations on features that are irrelevant and may actually not be helpful. Overfitting happens when a model exposed to too few examples learns patterns that do not generalize to new data, i.e. when the model starts using irrelevant features for making predictions. We wanted to make sure this in order to configure the system to remain highly accurate with new test data (user input).
 
 
Data augmentation is one way to fight overfitting, but it isn't enough since our augmented samples are still highly correlated. The main focus for fighting overfitting should be the entropic capacity of the model. Entropic capacity refers to how much information the model is allowed to store. A model that can store a lot of information has the potential to be more accurate by leveraging more features, but it is also more at risk to start storing irrelevant features. Meanwhile, a model that can only store a few features will have to focus on the most significant features found in the data, and these are more likely to be truly relevant and to generalize better. Because of this, we configured the model to extract only the most important elements during the data pre-processing phase when masking and reshaping the images, so that the model only interprets the transformed image.


A model that can store a lot of information has the potential to be more accurate by leveraging more features, but it is also more at risk to start storing irrelevant features. Meanwhile, a model that can only store a few features will have to focus on the most significant features found in the data, thus the quality of generalization is improved significantly.

In order to modulate entropic capacity, we optimized the number of parameters in the model (the number of layers + size of each layer). To further improve the model in terms of reducing overfitting and inaccurate generalizations, we implemented a dropout layer. Dropout prevents a layer from seeing twice the exact same pattern, working together with data augmentation to account for calling out possible inaccurate feature correlations the model might generate.


### 2.3. Preparing the Data
 
 In order to prepare the data for training, we will use .flow_from_directory() to generate batches of image data. We trained the model with a GPU called the Geforce Tesla K80, and implemented a validation dataset in order to cross-verify the accuracy of the model during training to simultaneously guarantee that the model is not overfitting.

### 2.4 Training the Model

In order to train the model, we implemented the model.compile() class in order to compile all of the input data and pass it through all of the layers of the neural network. The input data passed through the convolutional layer, which apply a filter to the input data by highlighting a feature region within the image and repeat the process for all of the subregions. The feature maps of the processed input image then are reduced to many dimensional vectors and arrays that are evaluated by the artificial neural network, and due to the decreased dimensionality, the neural network can access all of the smaller subregions to then create a response that is recorded between 0 to 1. All of the layers (convolutional (highlight regions of input image), average-pooling (down-sampling), fully-connected layers, act as the activation functions apart from the optimizer algorithm and selected loss function (categorical-crossentropy).

## Results

The optimized convolutional neural network performed at greater than 90% accuracy with thousands of images, while the other neural network with no pretrained nodes performed at a mere 60-70% accuracy.


## Conclusion

Given that we can precisely automate the process of detecting whether a brain tumor is present in a patient or not, while simultaneously accompanying it with an easy-to-use user interface (for the doctor + patient), hospitals and patients will be able to simplify their workflow for detecting anomalies much earlier and are able to capture it with precision without having to sacrifice accuracy.
To further add, healthcare providers will be able to adjacently use applications that are built on top of the rapidly evolving tech infrastructure for care delivery with less friction of accessibility and utilization (via web).There are many improvements to make within the models themselves to account for more diverse and unpredictable anomalies, which can be effectively improved in a cost-effective manner via generating more patient data to train the model using GANs. In this coming decade (2020-2029), the necessity for automation within care delivery will hopefully be deployed at scale, putting the core central focus of the patient back into the hands of the care providers, while lining up monetary incentives for all parties involved via an inverse system between efficiency and cost with automation.

## Improvements

* I: App
* II: Neural Network Architecture
* III: Web Platform Engineering
* IV: Reflection

### App
* Add sign-up page for users
* build API so medical developers can integrate the prognosis tool into their applications
* Add additional action buttons to allow patient to take action on prognosis (via booking appointments, getting directions to local clinical spaces)
* Build out CRUD properties for user profile and action buttons in terms of adding notes feature on the web page, etc.
### Neural Network Architecture
* Build General Adversarial Network in order to compensate for scaling data augmentation methods to generate diverse sets of medical data to train the model
* Build feature that outlines the tumor-infected nodules for the radiologist in order to prevent accidental treatment for healthy tissues in the brain
* Build additional neural network that can allow for more types of patient output (via fetched patient data to construct the final diagnostic assessment in order to prevent narrow-based diagnosis, which is why GlioAI is a diagnostic tool that still needs to be utilized by trained radiologists that can piece together elements of further abstraction along with image-based diagnostic assessment in order to yield optimized results.

### Web Platform Engineering
* Build out crowdsourcing platform so users (certified doctors who are verified via medical school email) can assist with machine-based diagnostic decisions (crowdsourcing platform for brain cancer detection (initial MVP, scale and branch out to other specialties later)
* Incentivize (platform) users with app-specific crypto tokens to reward them in proportion to the amount of value they export to assist in helping other physicians with making a prognosis etc.
### Reflection
* Given the current state that the model itself has been trained on a limited set(s) of patient MRI images with great accuracy, there is alot of area for improvement in terms of deploying extensive data augmentation (diversity of input image data for training), feature design, and overall application engineering and usability.

# Phase I: Build Crowdsourcing Protocols for Doctors in Need of Diagnostic Feedback

* The future of GlioAI lies in the idea of turning into a decentralized and pseudononymous crowdsourcing platform for medical practicioners and verified physicians and healthcare providers within the context of deep-knowledge tasks to further prune outputs from machines & automated systems ranging in disease detection and other areas in health.

* Build a platform that can distribute high-value work (aggregation of feedback from board-certified doctors)

* Game design mechanisms can be built out within the crowdsourcing platform in order to line up incentives for users to offer verified feedback that gets simultaneously ranked.

* Propagandistic behaviors cannot occur because of the account verification process in order to create content or rank/upvote other posts (containment + authentication-based friction).

* Enhancing treatment results via crowdsourcing platform specifically for verified doctors and healthcare providers (verified via school email + State ID)

* Integrate gig-based cryptoeconomic mechanisms in order to incentivize (digital) teledoctors to be able to easily generate income via telemedicine tasks to ensure accuracy of diagnosis within timely conditions via providing direct emotional support + answer questions and make clarifications.

* Crowdsourcing platform + machines = data-driven digital healthcare ecosystem

# Phase II: Working With Tangible Atoms to Deploy Network for Shipping Treatments
* Long term down the road, we can monetize in the future after productionization, and further deploy use of these systems by integrating drones to ship treatment medicine with tutorials on the web platform (goal is to make treatment methods open source and qualitatively aggregated together by verified doctors, also self-improving mechanism in terms of data and understanding)

# Project OKRS
* Refine backend system via building out external GAN via Tensorflow (generator + discriminator) in order to improve data that the neural network is trained on (by generating synthetic images that are able to account for far more anomalies involving Head MRI Scans, thus improving diversity of image data the model has been trained with

* Allow users to book appointments with local doctors and overhead hospital in-patient management in local healthcare facilities with google maps API

* Build platform with verification features built in to allow doctors to recieve feedback on content they post to designated specialties (ex = imaging --> brain tumors, hemorrhage, etc.)

* Build site that can sustain high-traffic load with all features built into platform

## References

* [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)

* [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

* [Deep Radiomics for Brain Tumor Detection and Classification from Multi-Sequence MRI](https://arxiv.org/abs/1903.09240)

* [CrowdBC: A Blockchain-based Decentralized Framework for Crowdsourcing](https://eprint.iacr.org/2017/444.pdf)

* [Handbook of Neuro-Oncology Neuroimaging](https://www.amazon.com/Handbook-Neuro-Oncology-Neuroimaging-Herbert-Newton-ebook/dp/B01DV7SKZA/ref=sr_1_1?keywords=Handbook+of+Neuro-Oncology+Neuroimaging%5D&qid=1577385706&s=books&sr=1-1)

* [Luigi Pirtoli, Giovanni Luca Gravina, Antonio Giordano (eds.) - Radiobiology of Glioblastoma_ Recent Advances and Related Pathobiology](https://www.amazon.com/Radiobiology-Glioblastoma-Advances-Pathobiology-Pathology-ebook/dp/B01GPJPJ2I/ref=sr_1_1?keywords=Radiobiology+of+Glioblastoma_+Recent+Advances+and+Related+Pathobiology&qid=1577385668&s=books&sr=1-1)

* [Advances in Biology and Treatment of Glioblastoma](https://www.amazon.com/Advances-Biology-Treatment-Glioblastoma-Research-ebook/dp/B073LLJJ6B/ref=sr_1_1?keywords=Advances+in+Biology+and+Treatment+of+Glioblastoma&qid=1577385631&s=books&sr=1-1)

* [Glioblastoma_ Molecular Mechanisms of Pathogenesis and Current Therapeutic Strategies](https://www.amazon.com/Glioblastoma-Mechanisms-Pathogenesis-Therapeutic-Strategies-ebook/dp/B008BB7URG/ref=sr_1_1?keywords=Glioblastoma_+Molecular+Mechanisms+of+Pathogenesis+and+Current+Therapeutic+Strategies&qid=1577385586&s=books&sr=1-1)
