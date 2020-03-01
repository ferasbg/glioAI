<h1 align="center"> 
  <img src=https://github.com/ferasbg/GlioAI/blob/master/media/fynlir_logo.png width="25%"><br/>GlioAI: Automatic Brain Tumor Detection System
</h1> 
<h4 align="center">
    Automatic Brain Tumor Detection Using 2D Deep Convolutional Neural Network for Diffusion-Weighted MRI
</h4>

## Contents

**Part I: Summary**

* [Overview](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#overview)
* [Introduction](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#context)
* [Objectives](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#objectives)
* [Workflow](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#workflow)
* [Dataset](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#dataset)

**Part II: Results**

* [Results](https://github.com/ferasbg/GlioAI#experiment-and-results)
* [Model and Training](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#model-and-training)
* [Comparison of Model Performance](https://github.com/ferasbg/GlioAI#comparison-of-the-models)


**Part III: Conclusion and Future Work**
* [Conclusion](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#conclusion)
* [Improvements](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#improvements)
* [Future of GlioAI](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#future-of-glioai)
* [Dependencies](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#dependencies)


**Additional Documentation**

* [References](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#references)
* [Bibliography](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#bibliography)
* [Attribution](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#attribution)
* [Contributing](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#contributing)
* [License](https://github.com/ferasbg/GlioAI/blob/master/docs/README.md#license)



## Overview 

GlioAI is an automatic brain cancer detection system that detects tumors in Head MRI scans.



## Context

* Primary malignant brain tumors are the most deadly forms of cancer, partially due to the dismal prognosis, but also because of the direct consequences on decreased cognitive function and poor quality of life. 

* The issue with building an effective machine learning system for a medical imaging-based task before was that there was not enough data. Using transfer learning and implementing slight architecture modifications to adapt the model to our dataset allowed GlioAI to perform at over 95% accuracy. 

* Given the context of a transfer learning approach to a feature detection problem, it is crucial to safeguard the probability of model overfitting with prevention methods involving data augmentation, applying data normalization and dropout layer. 

* [Magnetic Resonance Imaging](https://www.wikiwand.com/en/Magnetic_resonance_imaging) is a new method that has emerged for improving safety in acquiring patient image data. The utilization of these imaging tools are not yet fully maximized due to the variable of human operation within detecting cancers without enough time to make an accurate prognosis. 

* Because manual image feature extraction methods are very time inefficient, limited to operator experience, and are prone to human error, a reliable and fully automatic classification method using MRI data is necessary for efficient cancer detection.

* To begin solving this issue, we propose a fully automatic method for brain tumor classification, which is developed using [OxfordNet](https://keras.io/applications/#vgg16), a convolutional neural network that has been trained on over a million images from the ImageNet database.
* We can further enhance the usability of this tumor detection system by building a web application that stores the trained model in the back-end.


## Objectives

**Reduce Mortality Rates** 
* Create a model that will remove the variable of prognostic human error to improve patient survivability.

**Controlling Treatment Output** 
* Control the outcome in order to build a system that will mitigate human error and mortality rates.

**Scalability** 
* Accelerate the process of deployment for deep-learning based application.

**Cost-Effective** 
* Build a cost effective solution that reduces treatment costs via automation.

**Usability + Accessibility** 
* Create a user-friendly web app that will allow physicians and patients to easily upload their MRI data and receiving data reports and diagnostic results.

# Workflow

<h1 align="center">
  <img src=https://github.com/ferasbg/GlioAI/blob/master/media/web%20app%20workflow.png width="55%"
</h1>
  
<h1 align="center">
  <img src=https://github.com/ferasbg/GlioAI/blob/master/media/glioai%20app%20workflow.png width="95%"
</h1>
  
  
### Synopsis

* The user will import an image of a brain scan and the image will be sent as an http request through the resource server (REST API)
* The API will deliver the request to the back-end model
* The model will return a diagnosis back through the REST API server and to the front-end UI for the user.


## Back-End Design: Implement Convolutional Neural Network

We will be using a deep convolutional neural network, which is a neural network with a set of layers that will perform convolutions, pooling the set of regions of the image to extract features, along with with a softmax function that translates the last layer into a probability distribution.

<h1 align="center">
  <img src=https://github.com/ferasbg/GlioAI/blob/master/media/cnn%20process.jpg width="85%"
</h1>   

### Training Method

* We are only interested in applying transfer learning, which relies on training based on previously learned knowledge with a corpus of different large-scale datasets. 

* Because we are given a low volume of training data and are working with images, we decided to use [VGG16](https://keras.io/applications/#vgg16), a state-of-the-art convolutional neural network with 16 layers to increase the probability of attaining a greater model accuracy.


## Dataset

The [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) was used to train the model which had 253 brain MRI scans.   

The model was trained on 239 images belonging to two classes, and tested on 14.

# Experiment and Results

* I: Model and Training
* II: Comparative Model Results
* III: Data Evaluation

## Model and Training

The model consists of:

- CNN Layer
- Average Pooling Layer
- Dense Layer
- Fully Connected Layer
- Loss Function: Categorical Cross-Entropy
- Optimization Algorithm: Adam

Model is trained on 25 epochs.



  Models | Accuracy | Loss |	  
  | --- | --- | --- |	
  | Transfer Learning | 97%  |  13%  |	
  | No Transfer Learning | 76% | 49% |

## Transfer Learning: Model Accuracy
<h1 align="center">
  <img src=https://github.com/ferasbg/Fynlir/blob/master/media/TL%20Accuracy.png width="35%"
</h1> 
  
### No Transfer Learning: Model Accuracy

<h1 align="center">
  <img src=https://github.com/ferasbg/GlioAI/blob/master/media/NTL%20Accuracy.png width="35%"
</h1>
  
## Transfer Learning: Loss Curve

<h1 align="center"> 
 <img src=https://github.com/ferasbg/GlioAI/blob/master/media/TL%20Loss%20Curve.png width="35%" 
</h1>
  
  
## No Transfer Learning: Loss Curve

<h1 align="center">
  <img src=https://github.com/ferasbg/GlioAI/blob/master/media/NTL%20Loss%20Curve.png width="35%"
</h1>


## Comparison of the Models

<h1 align="center">
<img src=https://github.com/ferasbg/GlioAI/blob/master/media/comparison%20of%20the%20models.png width"65%"
</h1>    

### Evaluation

When comparing the results of the different models that were trained, it is clear that the transfer-learning based model is the most accurate deep learning model to deploy for the web app.

## Conclusion

* Given that we can precisely automate the process of detecting whether a brain tumor is present in a patient or not, while simultaneously accompanying it with an easy-to-use user interface (for the doctor + patient), hospitals and patients will be able to simplify their workflow for detecting anomalies much earlier and are able to capture it with precision without having to sacrifice accuracy. 

* To further add, healthcare providers will be able to adjacently use applications that are built on top of the rapidly evolving tech infrastructure for care delivery with less friction of accessibility and utilization (via web). 

* There are many improvements to make within the models themselves to account for more diverse and unpredictable anomalies, which can be effectively improved in a cost-effective manner via generating more patient data to train the model using [GANs](https://developers.google.com/machine-learning/gan/gan_structure). 

* After further model retuning and additional training optimization, GlioAI can specifically meet the pain points located within diagnosing brain tumors from MRI head scans, for brain cancer specialists and brain oncologists alike. Heading to a future where knowledge is aggregated and integrated with automated cancer detection systems in order to cut down diagnosis time over 1000-fold, from around 14 days of full reports to nearly 10-15 minutes, given the infrastructure for the crowdsourcing platform is built and incentive structures (via gig-based crypto token) and are aligned with verified physician users  

* In this coming decade (2020-2029), the necessity for automation within care delivery will hopefully be deployed at scale, putting the core central focus of the patient back into the hands of the care providers, while lining up monetary incentives for all parties involved via an inverse system between efficiency and cost with automation.  

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

### Web Platform Engineering
* Build out crowdsourcing platform so users (certified doctors who are verified via medical school email) can assist with machine-based diagnostic decisions (crowdsourcing platform for brain cancer detection (initial MVP, scale and branch out to other specialties later)
* Incentivize (platform) users with app-specific crypto tokens to reward them in proportion to the amount of value they export to assist in helping other physicians with making a prognosis etc.

## Reflection 

Given the current state that the model itself has been trained on a limited set(s) of patient MRI images with great accuracy, there is alot of area for improvement in terms of deploying extensive data augmentation (diversity of input image data for training), feature design, and overall application engineering and usability. 

# Future of GlioAI
* I: Main Focus for the Future
* II: Developmental Scope for 2020s
* III: Reflection on Targets
## Phase I: Integrate Cryptoeconomic Mechanisms within Crowdsourcing Web Platform for Radiologists

* The future of GlioAI lies in the idea of turning into a decentralized and pseudononymous crowdsourcing platform for medical practicioners and verified physicians and healthcare providers within the context of deep-knowledge tasks to further prune outputs from machines & automated systems ranging in disease detection and other areas in health. 

* Game design mechanisms can be built out within the crowdsourcing platform in order to line up incentives for users to offer verified feedback that gets simultaneously ranked. 

* Propagandistic behaviors cannot occur because of the account verification process in order to create content or rank/upvote other posts (containment + authentication-based friction).

* Enhancing treatment results via crowdsourcing platform specifically for verified doctors and healthcare providers (verified via school email + State ID)

* Integrate gig-based cryptoeconomic mechanisms in order to incentivize (digital) teledoctors to be able to easily generate income via telemedicine tasks to ensure accuracy of diagnosis within timely conditions via providing direct emotional support + answer questions and make clarifications.

* Crowdsourcing platform + machines = data-driven digital healthcare ecosystem

## Phase II: Deploy Telemedicine Mechanisms for Digital Doctors for Treating/Diagnosing Brain Cancer 

* We can further deploy use of these systems by integrating drones to ship treatment medicine with tutorials on the web platform (goal is to make treatment methods open source and qualitatively aggregated together by verified doctors, also self-improving mechanism in terms of data and understanding)

## Reflection on Scope/Targets for the Future

* The number of improvements to make are immense, and the roots originate from one single automated brain cancer detection system.
* We can build out models and deploy GANs for virtually all image-based radiology tasks first and then shift the focus from just digital radiology (platform + machines) to streamline processes for other diseases.

# Dependencies 

* Python

### Deep Learning

* Keras
* Tensorflow
* Matplotlib
* NumPy

### Web Application 

* Flask 
* CSS 
* Javascript
* HTML5
            

## References

* [Brain MRI Image Classification for Cancer Detection Using Deep Wavelet Autoencoder-Based Deep Neural Network](https://ieeexplore.ieee.org/document/8667628.)

* [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)

* [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

* [Deep Radiomics for Brain Tumor Detection and Classification from Multi-Sequence MRI](https://arxiv.org/abs/1903.09240)

* [CrowdBC: A Blockchain-based Decentralized Framework for Crowdsourcing](https://eprint.iacr.org/2017/444.pdf)

## Bibliography

* [Handbook of Neuro-Oncology Neuroimaging](https://www.amazon.com/Handbook-Neuro-Oncology-Neuroimaging-Herbert-Newton-ebook/dp/B01DV7SKZA/ref=sr_1_1?keywords=Handbook+of+Neuro-Oncology+Neuroimaging%5D&qid=1577385706&s=books&sr=1-1)

* [Luigi Pirtoli, Giovanni Luca Gravina, Antonio Giordano (eds.) - Radiobiology of Glioblastoma_ Recent Advances and Related Pathobiology](https://www.amazon.com/Radiobiology-Glioblastoma-Advances-Pathobiology-Pathology-ebook/dp/B01GPJPJ2I/ref=sr_1_1?keywords=Radiobiology+of+Glioblastoma_+Recent+Advances+and+Related+Pathobiology&qid=1577385668&s=books&sr=1-1)

* [Advances in Biology and Treatment of Glioblastoma](https://www.amazon.com/Advances-Biology-Treatment-Glioblastoma-Research-ebook/dp/B073LLJJ6B/ref=sr_1_1?keywords=Advances+in+Biology+and+Treatment+of+Glioblastoma&qid=1577385631&s=books&sr=1-1)

* [Glioblastoma_ Molecular Mechanisms of Pathogenesis and Current Therapeutic Strategies](https://www.amazon.com/Glioblastoma-Mechanisms-Pathogenesis-Therapeutic-Strategies-ebook/dp/B008BB7URG/ref=sr_1_1?keywords=Glioblastoma_+Molecular+Mechanisms+of+Pathogenesis+and+Current+Therapeutic+Strategies&qid=1577385586&s=books&sr=1-1)


## Attribution

Icon by [Srinivas Agra](https://thenounproject.com/srinivas.agra) from [thenounproject](https://thenounproject.com/)

## Contributing

Contributions are always welcome! For bug reports or requests please submit an issue.

## License

[MIT](https://github.com/ferasbg/GlioAI/blob/master/docs/LICENSE)
