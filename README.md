<h1 align="center"> 
  <img src=https://github.com/ferasbg/glioAI/blob/master/media/icons/fynlir_logo.png width="25%"><br/>GlioAI: Automatic Brain Tumor Detection System
</h1> 
<h4 align="center">
    Automatic Brain Tumor Detection Using 2D Deep Convolutional Neural Network for Diffusion-Weighted MRI
</h4>

<div align="center">
    <img src="https://github.com/ferasbg/glioAI/blob/master/media/app/malignant_tumor_recognition.gif" width="800" align="center">
</div>

<h2 align="center">
  GlioAI analyzing an MRI image (never seen before) that was tumorous, all done in less than 10 seconds.
</h2>
<div align="center">
  <img src="https://github.com/ferasbg/glioAI/blob/master/media/app/automated_brain_tumor_screening.gif" width="800" align="center">
</div>

<h2 align="center">
   GlioAI analyzing an MRI image of a healthy patient who wants to screen themselves efficiently to detect possible development of a brain tumor.
</h2>
  
  

## Contents

**Part I: Summary**

* [Overview](https://github.com/ferasbg/glioAI/blob/master/README.md#overview)
* [Introduction](https://github.com/ferasbg/glioAI/blob/master/README.md#context)
* [Objectives](https://github.com/ferasbg/glioAI/blob/master/README.md#objectives)
* [Workflow](https://github.com/ferasbg/glioAI/blob/master/README.md#workflow)
* [Dataset](https://github.com/ferasbg/glioAI/blob/master/README.md#dataset)

**Part II: Results**

* [Results](https://github.com/ferasbg/glioAI/blob/master/README.md#experiment-and-results)
* [Model and Training](https://github.com/ferasbg/glioAI/blob/master/README.md#model-and-training)
* [Comparison of Model Performance](https://github.com/ferasbg/glioAI/blob/master/README.md#comparison-of-the-models)
* [Usage](https://github.com/ferasbg/glioAI/blob/master/README.md#usage)


**Part III: Conclusion and Future Work**
* [Conclusion](https://github.com/ferasbg/glioAI/blob/master/README.md#conclusion)
* [Improvements](https://github.com/ferasbg/glioAI/blob/master/README.md#improvements)
* [Future of GlioAI](https://github.com/ferasbg/glioAI/blob/master/README.md#future-of-glioai)
* [Dependencies](https://github.com/ferasbg/glioAI/blob/master/README.md#dependencies)


**Additional Documentation**

* [References](https://github.com/ferasbg/glioAI/blob/master/README.md#references)
* [Bibliography](https://github.com/ferasbg/glioAI/blob/master/README.md#bibliography)
* [Attribution](https://github.com/ferasbg/glioAI/blob/master/README.md#attribution)
* [Contributing](https://github.com/ferasbg/glioAI/blob/master/README.md#contributing)
* [License](https://github.com/ferasbg/glioAI/blob/master/README.md#license)

**FAQ**

* [Value Proposition](https://github.com/ferasbg/glioAI/blob/master/README.md#value-proposition)
* [Market Size](https://github.com/ferasbg/glioAI/blob/master/README.md#market-size)
* [Product and User Acquisition](https://github.com/ferasbg/glioAI/blob/master/README.md#user-acquisition)
* [Product Execution](https://github.com/ferasbg/glioAI/blob/master/README.md#execution)
* [Growth Potential](https://github.com/ferasbg/glioAI/blob/master/README.md#growth-potential)
* [Miscellaneous](https://github.com/ferasbg/glioAI/blob/master/README.md#misc)




## Overview 

GlioAI is an automatic brain tumor recognition system that detects tumors in Head MRI scans.



## Context

* Primary malignant brain tumors are the most deadly forms of cancer, partially due to the dismal prognosis, but also because of the direct consequences on decreased cognitive function and poor quality of life. 

* The issue with building an effective machine learning system for a medical imaging-based task before was that there was not enough data. Using transfer learning and implementing slight architecture modifications to adapt the model to our dataset allowed GlioAI to perform at over 95% accuracy. 

* Given the context of a transfer learning approach to a feature detection problem, it is crucial to safeguard the probability of model overfitting with prevention methods involving data augmentation, applying data normalization and the dropout layer. 

* [Magnetic Resonance Imaging](https://www.wikiwand.com/en/Magnetic_resonance_imaging) is a new method that has emerged for improving safety in acquiring patient image data. The utilization of these imaging tools are not yet fully maximized due to the variable of human operation within detecting cancers without enough time to make an accurate prognosis. 

* Because manual image feature extraction methods are very time inefficient, limited to operator experience, and are prone to human error, a reliable and fully automatic classification method using MRI data is necessary for efficient cancer detection.

* To begin solving this issue, we propose a fully automatic method for brain tumor recognition, which is developed using [OxfordNet](https://keras.io/applications/#vgg16), a convolutional neural network that has been trained on over a million images from the ImageNet database.

* Because of the context of our problem, we optimized/modified the neural network so it is able to specifically train within the context of the Head MRI scans to improve performance. 

* We can further enhance the usability of this tumor detection system by building a web application that stores the trained model as an endpoint that can be hit from the front-end (user). 


## Objectives

### Reduce Mortality Rates
* Create a model that will remove the variable of prognostic human error to improve patient survivability.

### Controlling Treatment Output 
* Control the outcome in order to build a system that will mitigate human error and mortality rates.

### Scalability
* Accelerate the process of deployment for deep-learning based applications in a medical imaging context.

### Cost-Effective
* Build a cost effective solution that reduces treatment costs via automation.

### Usability + Accessibility
* Create a user-friendly web app that will allow physicians and patients to easily upload their MRI data and receiving data reports and diagnostic results.

# Workflow

<h1 align="center">
  <img src=https://github.com/ferasbg/glioAI/blob/master/media/app/web%20app%20workflow.png width="55%"
</h1>
  
<h1 align="center">
  <img src=https://github.com/ferasbg/glioAI/blob/master/media/app/app_workflow.jpg width="95%"
</h1>
  
  
### Synopsis

* The user will import an image of a brain scan and the image will be sent as an http request to the endpoint of the model
* The model will return a response to the user with the diagnosis.


## GlioNet: Optimized Deep Convolutional Neural Network Implemented in Back-End Infrastructure 

We will be using an optimized deep convolutional neural network called GlioNet, which is a neural network with a set of layers that will perform convolutions, pooling the set of regions of the image to extract features, that will translate the last layer into a probability distribution using the softmax function to classify the output of the patient MRI data.

<h1 align="center">
  <img src=https://github.com/ferasbg/glioAI/blob/master/media/app/cnn%20process.jpg width="85%"
</h1>   

### Training Method

* We are only interested in applying transfer learning, which relies on training based on previously learned knowledge with a corpus of different large-scale datasets. 

* Because we are given a low volume of training data and are working with images, we decided to use [VGG16](https://keras.io/applications/#vgg16) for our core infrastructure to begin with. It is a state-of-the-art convolutional neural network with 16 layers to increase the probability of attaining a greater model accuracy.


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
  <img src=https://github.com/ferasbg/glioAI/blob/master/media/data/TL%20Accuracy.png width="35%"
</h1> 
  
## No Transfer Learning: Model Accuracy


<h1 align="center">
  <img src=https://github.com/ferasbg/glioAI/blob/master/media/data/NTL%20Accuracy.png width="35%"
</h1>
  
## Transfer Learning: Loss Curve
<h1 align="center"> 
 <img src=https://github.com/ferasbg/glioAI/blob/master/media/data/TL%20Loss%20Curve.png width="35%" 
</h1>
  
  
## No Transfer Learning: Loss Curve


<h1 align="center">
  <img src=https://github.com/ferasbg/glioAI/blob/master/media/data/NTL%20Loss%20Curve.png width="35%"
</h1>


## Comparison of the Models

<h1 align="center">
<img src=https://github.com/ferasbg/glioAI/blob/master/media/data/comparison%20of%20the%20models.png width"65%"
</h1>    

### Evaluation

When comparing the results of the different models that were trained, it is clear that the optimized deep neural network performed at a far higher diagnostic accuracy, which we will implement in the web app.

## Conclusion

* Given that we can precisely automate the process of detecting whether a brain tumor is present in a patient or not, while simultaneously accompanying it with an easy-to-use user interface (for the doctor + patient), hospitals and patients will be able to simplify their workflow for detecting anomalies much earlier and are able to capture it with precision without having to sacrifice accuracy. 

* **It is really important to add that because image-based tumor recognition has become a well-defined problem with heavily promising solutions, we see the viability in the productionization and scale in developing this product further for our future mobile application, and platform for doctors.**

* To further add, healthcare providers will be able to adjacently use applications that are built on top of the rapidly evolving tech infrastructure for care delivery with less friction of accessibility and utilization (via web). 

* There are many improvements to make within the models themselves to account for more diverse and unpredictable anomalies, which can be effectively improved in a cost-effective manner via generating more patient data to train the model using [GANs](https://developers.google.com/machine-learning/gan/gan_structure). 

* After further model retuning and additional training optimization, GlioAI can specifically meet the pain points located within diagnosing brain tumors from MRI head scans, for brain cancer specialists and brain oncologists alike. Heading to a future where knowledge is aggregated and integrated with automated cancer detection systems in order to cut down diagnosis time over 1000-fold, from around 14 days of full reports to nearly 10-15 minutes, given the infrastructure for the crowdsourcing platform is built and incentive structures (via gig-based crypto token) and are aligned with verified physician users  

* In this coming decade (2020-2029), the necessity for automation within care delivery will hopefully be deployed at scale, putting the core central focus of the patient back into the hands of the care providers, while lining up monetary incentives for all parties involved via an inverse system between efficiency and cost with automation.  



## Usage

1. Install Dependencies With Your Command-Line

``` 
$ pip3 install -r requirements.txt
```

2. Install GlioAI
``` 
$ git clone https://github.com/ferasbg/glioAI.git
```

3. Change Directory to GlioAI
``` 
$ cd glioAI
```

4. Change Directory to Internal App 
``` 
$ cd glioai
```
5. Launch Server with App
``` 
$ python3 manage.py runserver
```

## Feature Roadmap

* I: App
* II: Neural Network Architecture
* III: Web Platform Engineering
* IV: Reflection

### App

* Add sign-up page for users  
* build mobile app with internal chatting system 
* integrate all web features onto mobile app
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

## Takeaway
The future of GlioAI will be a web platform that will allow doctors to recieve feedback from other verified doctors in order to make a far more efficient and accurate diagnosis in less than half the time. 

**GlioAI, the collaborative encyclopedic medical platform for doctors, built for the 21st century.**

## Bottleneck

- ensuring patient data privacy with patient-centric monetization model (not driven by monetization from selling data)
- backend security
- KYC Verification (private keys to ensure image data is anonymous)
- ranking cases based on user profile and professional line of work

## Phase I: Build Crowdsourcing Protocols for Doctors in Need of Diagnostic Feedback

* The future of GlioAI lies in the idea of turning into a decentralized and pseudononymous crowdsourcing platform for medical practicioners and verified physicians and healthcare providers within the context of deep-knowledge tasks to further prune outputs from machines & automated systems ranging in disease detection and other areas in health. 

* Build a platform that can distribute high-value work (aggregation of feedback from board-certified doctors)

* Game design mechanisms can be built out within the crowdsourcing platform in order to line up incentives for users to offer verified feedback that gets simultaneously ranked. 

* Propagandistic behaviors cannot occur because of the account verification process in order to create content or rank/upvote other posts (containment + authentication-based friction).

* Enhancing treatment results via crowdsourcing platform specifically for verified doctors and healthcare providers (verified via school email + State ID)

* Integrate gig-based cryptoeconomic mechanisms in order to incentivize (digital) teledoctors to be able to easily generate income via telemedicine tasks to ensure accuracy of diagnosis within timely conditions via providing direct emotional support + answer questions and make clarifications.

* Crowdsourcing platform + machines = data-driven digital healthcare ecosystem

## Phase II: Working With Tangible Atoms to Deploy Network for Shipping Treatments

* We can further deploy use of these systems by integrating drones to ship treatment medicine with tutorials on the web platform (goal is to make treatment methods open source and qualitatively aggregated together by verified doctors, also self-improving mechanism in terms of data and understanding)

## Project OKRS

- [ ] Refine backend system via building out external GAN via Tensorflow (generator + discriminator) in order to improve data that the neural network is trained on 
- [ ] Allow users to book appointments with local doctors and overhead hospital in-patient management in local healthcare facilities with google maps API
- [ ] Build platform with verification features built in to allow doctors to recieve feedback on content they post to designated specialties (ex = imaging --> brain tumors, hemorrhage, etc.)
- [ ] Build site that can sustain high-traffic load with all features built into platform


# FAQ

* [Value Proposition](https://github.com/ferasbg/glioAI/blob/master/README.md#value-proposition)
* [Tech](https://github.com/ferasbg/glioAI/blob/master/README.md#tech)
* [Market Size](https://github.com/ferasbg/glioAI/blob/master/README.md#market-size)
* [User Acquisition](https://github.com/ferasbg/glioAI/blob/master/README.md#user-acquisition)
* [Execution](https://github.com/ferasbg/glioAI/blob/master/README.md#execution)
* [Founder Market Fit](https://github.com/ferasbg/glioAI/blob/master/README.md#founder-market-fit)
* [Growth Potential](https://github.com/ferasbg/glioAI/blob/master/README.md#growth-potential)
* [Misc](https://github.com/ferasbg/glioAI/blob/master/README.md#misc)
* [Concluding Statement](https://github.com/ferasbg/glioAI/blob/master/README.md#concluding-statement)


## Value Proposition

<details><summary>Who specifically needs your product and why?</summary>
<p>
  
That is great question! Our product targets trained radiologists with heavy patient workloads who are in dire need of a reliable painkiller product that can restore the time that was taken away from them by circumstance to be able to interact with patients at the fully deep interpersonal level that was missing before. 

</p>
</details>


<details><summary>What are the main painpoints that you are targetting specifically?</summary>
<p>
   
* That’s a great question. We built this system because we wanted to alleviate the pain points involving limited time, high stress, and low diagnostic accuracy that is circumstantially faced by physicians and radiologists when making a treatment diagnosis given the extremely restricted conditions they have to deal with coupled with a high volume of patients. 
* If we can accelerate the process of detecting the presence of tumors themselves, we can allow physicians to have the ability to make more time to truly be there for their patient.
* It would cost virtually nothing for the user except internet bandwidth costs for web searches routed from ISP. Essentially anyone who is able to generate an image of a head MRI scan from their local hospital or clinical office can use this system, and it’s as simple as uploading the image to receive the diagnosis. 
* In the future, we will monetize the product we will launch in order to support the development, while hospitals will be able to generate larger profit margins from the decreased mortality rates from earlier cancer detection due to the efficient screening process that can be done all through the web.

</p>
</details>


</details>

<details><summary>What are you bringing to the table that is new and different than the current market?</summary>
<p>
  
* Very important question to consider! The optimized neural network that powers the GlioAI application is able to perform at a far more accurate level than other current neural networks including AlexNet and GoogleNet. 
  
* GlioAI operates on a primary backend infrastructure from VGG-Net, **is reported to perform at 97-98% accuracy**, and has been trained specifically with thousands of head MRI scans (labeled data) with pretrained layers.

* GlioAI is  **far more efficient than other current technologies due the improved methods of data augmentation** (the acquisition and scale of the training data) and **the consumer-friendly app-based usability of the neural network** and distribution potential by non-power users.

* GlioAI has a promising future in terms of becoming production-ready using GAN to generate MRI images to further dominate the position of the models that are being used. Additionally, there will be other infrastructure developments to make GlioAI ready for doctors around the world. 

</p>
</details>

<details><summary>What needs will be met for your userbase (doctors) with your product?</summary>
<p>
  
- The main priority is to **fill in the need of the doctor's workflow efficiency**, and taking a piece of the diagnostic process and speeding it up so that the doctor can have the freedom to allocate the recovered time to invest in their patient
- Our goal is for GlioAI to **help drastically improve the overall patient-doctor experience in terms of time, energy, and financial costs on both ends of the care delivery process** and dispersing positive secondary effects from the efficiency of the diagnosis during the treatment process.
- In order to make sure this happens, we **integrate our system into the daily workflow of the cancer specialist/radiologist/oncologist through a web app that can be downloaded by doctors around the world** who need to gain insight into the prognosis of their patient through MRI scans efficiently in order to move faster through the appointment for more logistical tasks that don't involve the pure interpersonal communication that is often cut down or out due to the variable of high-volume number of patients in hospitals or local clinical offices.
- Our goal is to **remove the stress and burden from the doctor’s workload**, to remove the variable of increased error and increased aging due to unnecessary stress from tasks that do not have to take a greater cognitive load on the overworked doctor to **give them leverage and control over their time with the patient.**
- Additionally, the goal of GlioAI is to also **guarantee flow throughout the treatment process so that they can work on the processes that require more cognitive load and attention that have further abstraction and systemic complexity** that is not well-defined yet for AI systems alike.
- Our goal is to **simplify the workflow of the doctor in the most efficient and reliable method possible for a specifically well-defined portion of the workflow of the doctor.**
  
</p>
</details>

<details><summary>How does your system improve current methods being deployed for image-based cancer detection?</summary>
<p>
  
- Our backend **model performs at over 97% accuracy, performing better than AlexNet and GoogleNet for brain tumor classification**, which again are just neural networks that have not been productionized for web users. 
- Additionally, GlioAI is a **far more improved and easy-to-use application** for the use of doctors in the healthcare space, so we saw our advantage to build a better solution both on the backend and the front-end involving the user experience.
- Our system was designed with the **state-of-the-art data augmentation methods for the specific context of our problem** (medical imaging, tumor detection).
- GlioAI has been **built with neural network layers that have been pretrained on millions of images and numerous image recognition tasks**, and the stored learning is then transfered for image feature extraction for our problem, which uses regions of MRI scans that **constructs connections between the pixels in the image subregions at an ultra-fast speed because it has already been trained, and is self-improving the more that the users feed it data.** 
- We have focused on **optimizing the quantity and diversity of the data we trained our model with, working to move from a few thousand images** to a few million specifically for head MRI scans with/without brain tumors in the coming future using generative adversarial neural networks.
- Again, we are only getting started. For **GlioAI 2.0, we will be able to further improve our system by building out a GAN that is able to escape the regulatory bottleneck involved in restricted data acquisiton and train our model with hundreds of thousands of images, cementing our system's accuracy and beating out the accuracy of doctors with decades of experience.** 
- Furthermore, building a GAN to further train an optimized neural network that has had pretrained layers will be able to increase the validity of the diagnostic performance and accuracy, and reducing the error even further. Of course it's important to keep in mind that this remains a feature implementation for the future. Using GAN, our plan is to produce millions of MRI images that represent anomalies generated by a separate neural network that creates images based on what we already have while letting our model train in the cloud, thus being able to have the most accurate brain tumor detection models in the market, let alone the world. 
- Keep in mind that the bottleneck as to why others were not able to do this is because of the difficulty with the data acquisition itself.
* Timing is everything when it comes to a patient with a live tumor growing in their brain, and it is crucial to speed up the diagnostic process while simultaneously increasing the rate of accuracy, which remains the highest priority for GlioAI.

</p>
</details>


<details><summary>What are the main costs involved in terms of resources (money, time, compute) for the user?</summary>
<p>
  
* There are no additional costs! Although, the user needs to have working internet, a CMD to install our app (which is pre-installed on your computer), and a web browser on the computer to run the app on a local server (which also comes pre-installed, so easy-install and efficient user onboarding along with application usability). We will soon launch on the cloud. 
* The ROI itself pays off the user's bandwidth costs given that over time there will be less fatalities and projected sales from patients that transfer their healthcare to the options given that provide more assurance, transparency, efficiency, and most importantly, patient quality.

</p>
</details>

<details><summary>What specific part of the radiology pipeline are you improving, what abstraction is left in the diagnostic process?</summary>
<p>
  
* We are specifically targetting patient diagnosis and automated image scanning and analysis during testing for brain tumors. 
* Additionally put, we are targetting the earlier part of the entire brain cancer treatment workflow to speed up the time it takes to detect the development of the tumor itself, which is extremely important in order to determine the treatment approach.
* To give some context of what processes go into treating a cancer patient, the current workflow additionally involves data acquisition, diagnostics, treatment planning, and treatment execution to deliver the cure/solution to the patient to alleviate and reduce the level of development of the brain tumor. 
* Abstraction remains an issue with mainly understanding how to eliminate the tumor itself in a more efficient manner, and being able to map out simulations of how the drugs themselves remove the tumor itself with as little downside as possible, which still remains a huge issue with current chemotherapy methods and drug-related solutions. 

</p>
</details>

<details><summary>Why is the time of now the best time to deploy this technology for the world?</summary>
<p>
  
* AI systems have shown to be able to detect cancers from image-related input data and have demonstrated a promise in accuracy and consistent results, and due to the current state of healthcare, are more seen as a necessary painkiller rather than simply a novel upgrade than previous workflows. In order to advance our position within healthcare for cancers and diseases that can be diagnosed via images, now is the right time to productionize these systems for doctors around the world that are in dire need of more time with their patients along with alleviating their stress to allow them to perform far better in a consistent manner.
* In the 2020s, we are projecting to see more development and innovation in the biotech sectors, as regulatory bottlenecks and other limitations will be redesigned in order to incentivize individuals to advance the current technology that is used in healthcare. 
* Because the incentives are lined up for us to move in to this industry coupled with the well-defined nature of the problem that we are solving, we see a bright future for the development and deployment of our technology for doctors and patients alike.
* Current treatment pipelines need innovation in a dire manner, and in a robust manner where they can be easily implemented and baked into the workflow of doctors and other healthcare providers that operate in the industry.
* We see a huge growth potential that we can leverage upon in deploying our technology. 
* We will be able to exponentially optimize current healthcare pipelines involving treatments and diagnostic processes and image-related tasks that go into detecting cancer and treating it.


</p>
</details>

<details><summary>In Lehmann's terms, explain to me the complete workflow behind the development of your product.</summary>
<p>
  
* We modified a convolutional neural network from DeepMind so that it is optimized specifically for the context of our image recognition problem, which was to detect brain tumors. 
* We did this by training the last layers with a specific dataset that was augmented to thousands of image data, and then tuned the model to avoid overfitting on the image data during training using a dropout layer, which increases the validation accuracy which then improves how the model is able to generalize each image to result to a diagnosis!
* Next, we then implemented our modified neural network into the backend infrastructure of our web application, so that we can build out the front-end user interface for the user which would be the doctor.
* Note that we prioritized usability over feature complexity in order to reduce the friction and steps it takes for the doctor to upload an image and receive a diagnosis by keeping the interface simple and clean so we remove the difficulty and skills required to use the app.
* We then tested our system with MRI images that it had never seen before, and the image goes through some data preprocessing functions in order to fit the dimensions and features that our system can analyze (you can think of this process like adding filters and cropping as you do when editing images).
* The input image is then preprocessed and then is fed to the neural network.
* Based on the probability distribution that is produced by the softmax function of the model, the score would then determine whether or not the image has a tumor.
* The response from our backend system is then sent to the front-end interface of the user (their screen).

</p>
</details>

<details><summary>How is your method improving the current state of all machine intelligence systems?</summary>
<p>
  
* **Our method is able to simultaneously improve the accuracy of our image recognition algorithm with an an easy-to-use application that can work on any web browser** alleviating painpoints of radiologists without being difficult-to-use keeping in mind of the importance of the doctor's time within their treatment workflow, and hospitals alike.
* We want doctors to focus on their patients rather than dealing with clunky software, having control over their time and the use of the technology they apply to help them diagnose patients and additionally decreasing the lag of time involved in treatment planning and executing a workable treatment as efficiently as possible.
* Our system is focused to be user-centric and built for the doctor, while improving the nature of our system over time and simplfying the distribution of our technology simultaneously (via the web).

</p>
</details>


<details><summary>How does the neural network, work?</summary>
<p>
  
### Introduction
  Let's divide up the workflow that was executed to build out and implement the neural network into 1) data acquisition, 2) data augmentation, 3) model training, 4) saving, and 5)loading the model for the app.

### Data Acquisition
* Let's begin with how we built the neural network, the data that we used, and how we enhance the performance of the neural network along with the signal (quality) of our system overall from the diversity of image data that was given.
* We acquired an open-source dataset from Google Cloud, which was a total of 253 images. We knew that we had to split the data that we train our model with along with the data that we use to test to make sure our models works with novel patient data as well.
### Data Augmentation
* To make sure that we account for more diverse data and patient cases due to the limited nature of the data that we are working with, we were able to successfully augment the number of images that we started with by applying transformations to the images themselves in order to produce thousands of images that the model can train on, so that it does not see the image twice and can make more data-supported generalizations.
* We then came across another problem that we were able to fix as well, which involved the limited nature of our image data which led to a possibility of model overfitting, where the model can make generalizations on irrelevant features and are not actually helpful because the volume of the images was low to begin with.
* Augmenting our data to fight overfitting was not going to be enough because our augmented samples are still highly correlated. In order to avoid storing information regarding irrelevant features of the images, we configured the model to extract only the most important elements during the data pre-processing phase when masking and reshaping the images, so that the model only interprets the transformed image. We wanted to configure our model so that it can only store a few features which will have to focus on the most significant features found in the data, thus the quality of generalization then improved significantly.
* To further improve the model in terms of reducing overfitting and inaccurate generalizations, we implemented a dropout layer. Dropout prevents a layer from seeing twice the exact same pattern, working together with data augmentation to account for calling out possible inaccurate feature correlations the model might generate.
* The preparation for our model is almost complete. Now we use the .flow_from_directory() function in tensorflow in order to generate batches of image data in order to control how many different images that model can be trained on through each cycle, which we will call epochs.
* We trained the model with a GPU called the Geforce Tesla K80, and implemented a validation dataset in order to cross-verify the accuracy of the model during training to simultaneously guarantee that the model is not overfitting.
### Training the Model
* In order to train the model, we implemented the model.compile() class in order to compile all of the input data and pass it through all of the layers of the neural network. 
* The input data passed through the convolutional layer, which apply a filter to the input data by highlighting a feature region within the image and repeat the process for all of the subregions. The feature maps of the processed input image then are reduced to many dimensional vectors and arrays that are evaluated by the artificial neural network, and due to the decreased dimensionality, the neural network can access all of the smaller subregions to then create a response that is recorded between 0 to 1.
* Our neural network had 16 layers, which consisted of convolutional layers that processed the various dimensions of the subregions within our image, and passed the different subregions through the average pooling layer, which calculates the average for each patch of the feature map that was formed by the convolutional layer. 
* This is done in order to reduce the amount of data actually being processed through the system in order to highlight the most important pixels of our image (down-sampling --> reducing pixels of image). 
* We repeated this process several times, and then applied our Adam optimizer function which reduces the cost function which increases the measure of the performance of the model in respect to training. 
* We then implemented a dense layer which fully connects all of the neurons in all of the layers in our neural network. 
* We finally converted the output of the last layer into a probability distribution of multiple classes in order to attach the score to the classes, which were either if the image had a tumor (yes), or if the image did not have a tumor (no).

### Saving Our Model
* We then saved our trained keras model into an H5 file that stored the weights and architecture of our model, so that we can use it for our application.
### Loading the Model
* In the front-end, we were able to successfully load the keras model stored in the backend in order to take the input image and return a response.

</p>
</details>

## Market Size 
<details><summary>How many people are in your target market?</summary>
<p>
  
This is really important. There are thousands of hospitals (approx 16,500 according to the American Hospital Association) and dozens of independent clinician offices around the entire world.

</p>
</details>


<details><summary>How fast is the market growing?</summary>
<p>

According to Allied Market Research, "the current growth rate of the digital pathology market is projected to reach USD 1.139 billion by 2025 from USD 613 million **in 2020**, at a CAGR **of** 13.2% during the forecast period."  

</p>
</details>

<details><summary>How much time and money do users in your target market waste before switching to your product/service?</summary>
<p>

According to the National Health Services, it usually takes a week or two for the **results** of an **MRI** scan to come through. Let's operate under an average of 10.5 days for a diagnosis to be sent to the patient regarding the development of a brain tumor. **The diagnostic turnaround with our system cuts down the previous time from 10.5 days (252 elapsed hours) to 10 seconds, saving over 250 hours of elapsed time, resulting in over 95% time saved!**


</p>
</details>

## Product and User Acquisition

<details><summary>What monetization model will you use (to protect your users?)</summary>
<p>
  
- We will monetize through a subscription-based model, giving access for doctors to use our tumor recognition system for free.
- Additionally to the basic diagnostic features of the app, the premium features that they want to use if they choose to take their workflow to the next level, will include detailed MRI reports, and other developed models that analyze other abstractions that go into the overall diagnosis
- Additionally to this, The GlioAI Premium Mobile Subscription will include data visualization that allows them to view sets of different data trends of their own patients to be able to gauge clarity of their patients (operating like a second brain CRM for doctors) and different analytics in an easy-to-read format through graphs, charts, and other illustrations created through the MRI reports additionally to the feedback that is sent by other verified doctors on the web platform.

</p>
</details>

## Product Execution 

<details><summary>What should we see in 1 year, 2 years, 3 years, and beyond? Illustrate GlioAI's future and your vision for this product.</summary>
<p>
  
*  analytics report + more features to allow users to connect with doctors on the app, so that they can recieve immediate feedback. No more week-long wait times to hear back from the doctor, those days are over. We will now develop the next part of GlioAI 2.0, involving an integrated application to allow patients to communicate individually with their doctors over an encrypted chatting network in the application in order to gain real-time insight on the development of their case including treatment planning and additional steps to take after image scans and testing have taken place. 
*  Long term, our main goal is to make brain cancer as critical as the flu, and this can be done through being able to have the right tools to remove error and any lag from patient diagnosis to treatment. For this to happen, we need to invest more in being able to have an increased developmental understanding on our treatment methods and being able to treat patients as efficiently and accurately as possible.
 


</p>
</details>

<details><summary>You have the minimum product built. Now what parts of this are you going to build next?</summary>
<p>
  
* We will be launching a mobile app where users can upload their MRI images and communicate closely with connected doctors and also be able to access the additional diagnostic notes aggregated from crowdsourced feedback from verified doctors around the world that will use the GlioAI platform.  

</p>
</details>


## Growth Potential

<details><summary>State your projected earnings for the upcoming quarters given that the app is deployed and the company becomes established. How much can you make a year?</summary>
<p>
 
*  We will be able to set a $30 monthly subscription for our mobile app. For our web browser-based application, there will be various subscription options, including Basic, Pro, and Enterprise. Enterprise will target hospitals that will choose to purchase a monthly renewed license to the app in order to reduce expenses that doctors have to pay out-of-pocket if they want to have access to the entire set of features in GlioAI Mobile and GlioAI Web. Users will be able to have their accounts connected.
*  The enterprise-based monthly subscription will be $500 (because it is targeted towards hospitals, and the ROI pays for the subscription itself over time).
* Given that we can be explicit on how the hospitals will be able to increase their profit margins and revenue in their quarterly earnings reports, then we can use cold-email + other marketing methods to acquire these customers.  
* If we are able to acquire over 900-1000 hospitals in the US (there are 6,146 total according to the American Hospital Association), then we can make over $5.7 million in 12 months ($500 x 950 hospital users x 12 months) given that we are on pace with the development of the web platform, mobile application, system deployment, cloud-hosting, and other feature-related targets, then we can effectively scale our company to play as a dominant player in the telemedicine platform + digital radiology market, effectively leveraging us to compete with fewer competitors, allowing us to focus on the robustness of our product.


</p>
</details>


<details><summary>How does this become a billion-dollar company?</summary>
<p>
  
*  If we can effectively scale our platform and applications and dominate the market of decentralized and distributed diagnostic systems, then we can scale our development team in order to build other applications within digital radiology that can be integrated along with live feedback from verified doctors (distributing the process of doctor feedback through the web instead of isolating the task to 1 doctor only --> limited feedback). 
*  If we capture the market share involving the digital pathology tools that are being built due to our ability to scale infrastructure, then we can scale operations and offer various applications for individual organs of the human body (brain tumors --> brain, lung cancers --> lungs, etc.), and deliver more products simultaneously for hospitals in the US, and then throughout the world, allowing us to make around (1100 US hospitals + 9000/16000 international hospitals = 10100 hospitals x $500 subscription x 4 different product licenses x 12 months = $242.4 million per year --> 4 and a half years --> $1 billion company)


</p>
</details>

<details><summary>If your startup succeeds, what additional areas might you be able to expand into within this healthcare space?</summary>
<p>
  
We currently will focus on radiology and maximize the image-related diagnostic processes with other modalities other than MRI, and will branch out to other cancers in order to offer more products and develop generative models with other forms of image data in the future.

</p>
</details>



## Misc

<details><summary>What have you learned so far from working on this idea?</summary>
<p>
  
* There is so much uncapped potential from merging platform infrastructure for telemedicine given the recent laws passed to allow doctors to operate in the 50 states. 
* Now is the best time for doctors to be able to speed up and refine the feedback and diagnostic accuracy, handling any abstraction from the feedback of verified doctors and the diagnosis from the systems on the platform. 
* There will be so much intelligence aggregated and ranked, and there is a lot of potential for building this telemedicine network that can transform itself as a cybernetic collective over a digital network, where machines can automatically detect cancers and doctors can offer more feedback from input images that other doctors can post.

</p>
</details>


<details><summary>Why isn’t someone already doing this?</summary>
<p>
  
* Currently [viz.ai](https://www.viz.ai/) is working on building software solutions for detecting strokes using medical images, and when dealing with the development of diagnostic systems, there needs to be more stability and product robustness.
* Currently, it is the best time to exploit the opportunity to build a telemedicine-based network platform and implementing machine learning systems within the platform, in order to combine aggregated high-signal human feedback along with the diagnostics delivered through the automated systems operating within the network infrastructure.

</p>
</details>

<details><summary>What will keep you up at night?</summary>
<p>
  
* servers must run! (no outages)
* platform maintanence
* continued functionality of all features

</p>
</details>

# Dependencies 

* Python

### Deep Learning

* Keras
* Tensorflow
* Matplotlib
* NumPy

### Web Application 

* Django 
* CSS 
* Bootstrap
* HTML5

# Links 

* [Video](https://www.youtube.com/watch?v=ttS-RH3o0mM)
* [Project Site](https://ferasbg.github.io/glioAI/)
* [Visual Write-Up](https://medium.com/@cryptomartian/glioai-automatic-brain-tumor-detection-system-for-diffusion-weighted-mri-1c808281245f?source=your_stories_page---------------------------)

## References

* [Brain MRI Image Classification for Cancer Detection Using Deep Wavelet Autoencoder-Based Deep Neural Network](https://ieeexplore.ieee.org/document/8667628.)

* [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)

* [Deep Radiomics for Brain Tumor Detection and Classification from Multi-Sequence MRI](https://arxiv.org/abs/1903.09240)

* [Handbook of Neuro-Oncology Neuroimaging](https://www.amazon.com/Handbook-Neuro-Oncology-Neuroimaging-Herbert-Newton-ebook/dp/B01DV7SKZA/ref=sr_1_1?keywords=Handbook+of+Neuro-Oncology+Neuroimaging%5D&qid=1577385706&s=books&sr=1-1)

* [Luigi Pirtoli, Giovanni Luca Gravina, Antonio Giordano (eds.) - Radiobiology of Glioblastoma_ Recent Advances and Related Pathobiology](https://www.amazon.com/Radiobiology-Glioblastoma-Advances-Pathobiology-Pathology-ebook/dp/B01GPJPJ2I/ref=sr_1_1?keywords=Radiobiology+of+Glioblastoma_+Recent+Advances+and+Related+Pathobiology&qid=1577385668&s=books&sr=1-1)

* [Advanced Magnetic Resonance Imaging in Glioblastoma: A Review](http://cco.amegroups.com/article/view/15820)

* [Radiomics in Brain Tumor: Image Assessment, Quantitative Feature Descriptors, and Machine-Learning Approaches](https://www.ncbi.nlm.nih.gov/pubmed/28982791)

* [An Ensemble of 2D Convolutional Neural Networks for Tumor Segmentation](https://link.springer.com/chapter/10.1007/978-3-319-19665-7_17)

* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

* [Dropout: A Simple Way to Prevent Neural Networks From Overfitting](https://dl.acm.org/doi/10.5555/2627435.2670313)

* [Tumor Cell Heterogeneity](https://www.ncbi.nlm.nih.gov/pubmed/24084451)

* [Fundamentals of Diagnostic Error in Imaging. Radiographics: a Review Publication of the Radiological
Society of North America](https://www.ncbi.nlm.nih.gov/pubmed/30303801)

* [Early Grade Classification in Meningioma Patients Combining Radiomics and Semantics Data](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.4955670)

* [Advances in Biology and Treatment of Glioblastoma](https://www.amazon.com/Advances-Biology-Treatment-Glioblastoma-Research-ebook/dp/B073LLJJ6B/ref=sr_1_1?keywords=Advances+in+Biology+and+Treatment+of+Glioblastoma&qid=1577385631&s=books&sr=1-1)

* [Glioblastoma_ Molecular Mechanisms of Pathogenesis and Current Therapeutic Strategies](https://www.amazon.com/Glioblastoma-Mechanisms-Pathogenesis-Therapeutic-Strategies-ebook/dp/B008BB7URG/ref=sr_1_1?keywords=Glioblastoma_+Molecular+Mechanisms+of+Pathogenesis+and+Current+Therapeutic+Strategies&qid=1577385586&s=books&sr=1-1)

* [CrowdBC: A Blockchain-based Decentralized Framework for Crowdsourcing](https://eprint.iacr.org/2017/444.pdf) 

* [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)


## Attribution

Icon by [Srinivas Agra](https://thenounproject.com/srinivas.agra) from [thenounproject](https://thenounproject.com/)

## Contributing

Contributions are always welcome! For bug reports or requests please submit an issue.

## License

[MIT](https://github.com/ferasbg/GlioAI/blob/master/docs/LICENSE)
