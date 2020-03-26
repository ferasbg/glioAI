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
* [Tech](https://github.com/ferasbg/glioAI/blob/master/README.md#tech)
* [Market Size](https://github.com/ferasbg/glioAI/blob/master/README.md#market-size)
* [User Acquisition](https://github.com/ferasbg/glioAI/blob/master/README.md#user-acquisition)
* [Execution](https://github.com/ferasbg/glioAI/blob/master/README.md#execution)
* [Founder Market Fit](https://github.com/ferasbg/glioAI/blob/master/README.md#founder-market-fit)
* [Growth Potential](https://github.com/ferasbg/glioAI/blob/master/README.md#growth-potential)
* [Misc](https://github.com/ferasbg/glioAI/blob/master/README.md#misc)
* [Concluding Statement](https://github.com/ferasbg/glioAI/blob/master/README.md#concluding-statement)



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
  
</p>
</details>


</details>

<details><summary>What are you bringing to the table that is new and different than the current market?</summary>
<p>
  
</p>

</details>


</details>

<details><summary>What needs will be met for your userbase (doctors) with your product?</summary>
<p>
  
</p>
</details>


</details>

<details><summary>What are your users doing now?</summary>
<p>
  [insert statement involving the current methods they are doing and the downside effects from the ineffectiveness]
</p>
</details>

</details>

<details><summary>How does your product work in more detail?</summary>
<p>
  
</p>
</details>


</details>

<details><summary>How does your system improve current methods being deployed for image-based cancer detection?</summary>
<p>
  [insert statement involving model accuracy and it's range of usage and future features that can be implemented and the scope to use GANs with modified Oxfordnet (GlioNet) in order to account for millions of images (given there is GPU/compute access, monetization must be implemented, thus be specific in the costs involved on the userside and how the monetization will help you scale the product]
</p>
</details>


</details>

<details><summary>What are the main costs involved in terms of resources (money, time, compute) for the user?</summary>
<p>
  [specify what "burden" cost that user has to take (i.e. internet usage costs, CPU usage/efficiency when rendering page, what we are alleviating in terms of cost]
</p>
</details>


</details>

<details><summary>What specific part of the radiology pipeline are you improving, and what should we keep in mind of in terms of the abstraction that is left in the full process of the diagnosis and treatment of a patient that we can further improve the efficiency for?</summary>
<p>
  
</p>
</details>


<details><summary>Why is the time of now the best time to deploy this technology for the world?</summary>
<p>
 [AI promises and results that are being achieved currently in the 2020s, how it can truly revolutionize healthcare space, along with the promise of more investment within optimizing current healthcare pipelines involving treatments and diagnostic processes and image-related tasks that go into detecting cancer and treating it, how we can leverage current market growth]

</p>
</details>


<details><summary>In Lehmann's terms, explain to me the complete workflow behind the development of your product.</summary>
<p>
 [Sure, go into neural net dev --> neural net modification --> testing different neural nets --> implementation of trained model and prediction function in backend --> connecting backend infrastructure with frontend to render web app]

</p>
</details>


<details><summary>How is your method improving the current state of all machine intelligence systems?</summary>
<p>
 [Sure, go into neural net dev --> neural net modification --> testing different neural nets --> comparable results to AlexNet, ImageNet, etc. for specifically brain tumor imaging]

</p>
</details>


<details><summary>How does the neural network, work?</summary>
<p>
 [Sure, go into neural net dev --> datapreprocessing --> conv --> average pooling --> relu --> conv --> fully-connected --> optimizer --> dense layer --> softmax function for probability distribution for last layer]

</p>
</details>


<details><summary> How did you build out an app with the trained neural network that you built?</summary>
<p>
 [Sure, go into neural net dev --> datapreprocessing --> conv --> average pooling --> relu --> conv --> fully-connected --> optimizer --> dense layer --> softmax function for probability distribution for last layer]

</p>
</details>


## Market Size 
<details><summary>How many people are in your target market?</summary>
<p>

[important to state that its better to ride the wave of current trends and specializing in a specific painpoint and maximizing the efficiency of the product --> productionization + launch with baked in features for users with access to MRI images to solve one part of the problem]

</p>
</details>


<details><summary>How fast is the market growing?</summary>
<p>

[important to state that its better to ride the wave of current trends and specializing in a specific painpoint and maximizing the efficiency of the product --> current growth rate, and number of users and people using computers for EHR (electronic health records, image-related tools within MRIs, etc.]

</p>
</details>


<details><summary>How fast is the market growing?</summary>
<p>

[important to state that its better to ride the wave of current trends and specializing in a specific painpoint and maximizing the efficiency of the product --> current growth rate, and number of users and people using computers for EHR (electronic health records, image-related tools within MRIs, etc.]

</p>
</details>


<details><summary>How much time and money do users in your target market waste before switching to your product/service?</summary>
<p>

[time wasted = x amount of time for MRI diagnosis, time to recieve MRI report, time for accurate diagnosis, implications of wrong diagnosis on the patient's end, hospital, and/or doctor operating as an LLC]


</p>
</details>


<details><summary>What are some trends in your target market, and related markets?</summary>
<p>
  
[SECRET_KEY=%395830582544w345w5fe55asf5efefefferfapgxyuhefo32]

</p>
</details>

## Product and User Acquisition

<details><summary>What monetization model will you use (to protect your users?)</summary>
<p>
  
[SECRET_KEY=%395830582544w345w5fe55asf5efefefferfapgxyuhefo32]

</p>
</details>


<details><summary>How is this system feasible for production?</summary>
<p>
  
(go into workflow specifics and the breakdown of the "complexity" of launching to the cloud and productionizing the software)


</p>
</details>


<details><summary>What will you do to prepare for the nuance and hidden variables in deploying medical software? What black swan event do you think you should and will prepare for moving forward?</summary>
<p>
  
[SECRET_KEY=%395830582544w345w5fe55asf5efefefferfapgxyuhefo32]


</p>
</details>


<details><summary>Why might doctors be reluctant to try this tool, and how can you ensure them that this will help them more than any downside relating to the use of the technology itself?</summary>
<p>
  
[again, this only automates 1 part of the entire digital radiology pipeline (show img with arrow here) and there are always downsides with every novel product that users may acquire, but we ensure that the main intent is to make sure that the doctor as the user is in mind throughout the development of the product, to make sure that it is easy to use, not clunky software, and is able to cut down times that they previously spent due to the intense schedule with heavy load of patients, the entire end-to-end process of patient care (cancer patients) is not solved yet, but our technology is able to solve a piece of the puzzle. If you feel hesitant to use our product, use your local image database of head MRI scans of brain tumor patients along with our test dataset in order to verify that our system, works! Also refer to the video demo where we show this as well :) --> revise pls]


</p>
</details>

<details><summary> Ok, so your system works. But how will you solve the bottleneck of regulations with the FDA in order to deploy it for doctors worldwide?</summary>
<p>
  
[SECRET_KEY=%395830582544w345w5fe55asf5efefefferfapgxyuhefo32]

</p>
</details>

<details><summary>How will you get users?</summary>
<p>
  
[SECRET_KEY=%395830582544w345w5fe55asf5efefefferfapgxyuhefo32]

</p>
</details>

<details><summary>What exactly makes you different from existing options?</summary>
<p>
  
[SECRET_KEY=%395830582544w345w5fe55asf5efefefferfapgxyuhefo32]


</p>
</details>


<details><summary>How will users find out about you?</summary>
<p>
  
[SECRET_KEY=%395830582544w345w5fe55asf5efefefferfapgxyuhefo32]


</p>
</details>



## Product Execution 

<details><summary>Can you be clear on your execution targets and your strategy in terms of the phases you would go through? What are your main targets going into the launch and deployment of this product, and how will you take this forward afterwards?</summary>
<p>
  
[SECRET_KEY=%395830582544w345w5fe55asf5efefefferfapgxyuhefo32]

</p>
</details>


<details><summary>Why should we think that you can revolutionize the state of healthcare, and what will you bring to the table long term with your product and the scope of it's future?</summary>
<p>
  
[SECRET_KEY=%395830582544w345w5fe55asf5efefefferfapgxyuhefo32]

</p>
</details>


<details><summary>What should we see in 1 year, 2 years, 3 years, and beyond? Illustrate GlioAI's future and your vision for this product.</summary>
<p>
  
[SECRET_KEY=%395830582544w345w5fe55asf5efefefferfapgxyuhefo32]

</p>
</details>


<details><summary>You have the minimum product built. Now what parts of this are you going to build next?</summary>
<p>
  
jekgnenelkgfnleka

</p>
</details>



## Founder Market Fit
<details><summary>Why did you choose this idea?</summary>
<p>
  
[SECRET_KEY=%395830582544w345w5fe55asf5efefefferfapgxyuhefo32]

</p>
</details>


<details><summary>Why are you uniquely qualified to work on this?</summary>
<p>
  
[SECRET_KEY=%395830582544w345w5fe55asf5efefefferfapgxyuhefo32]

</p>
</details>


<details><summary>Why do you want to dedicate your time to working on this? Why does this problem matter to you at all?</summary>
<p>
  
[explain philosophical, objective, and emotional connection to this product, and to the fine detail why it matters whatsoever to even exist]

</p>
</details>




## Growth Potential
<details><summary>Alright, so we got ourselves an automated tumor recognition system that doctors can use. How will you monetize this? How will you make money? </summary>
<p>
  
efnefnjaefnelafoea

</p>
</details>


<details><summary>State your projected earnings for the upcoming quarters given that the app is deployed and the company becomes established. How much can you make a year?</summary>
<p>
  
efnefnjaefnelafoea

</p>
</details>


<details><summary>How does this become a billion-dollar company?</summary>
<p>
  
jekgnenelkgfnleka

</p>
</details>


<details><summary>If your startup succeeds, what additional areas might you be able to expand into within this healthcare space?</summary>
<p>
  
jekgnenelkgfnleka

</p>
</details>



## Misc
<details><summary>What have you learned so far from working on this idea?</summary>
<p>
  
jekgnenelkgfnleka

</p>
</details>


<details><summary>Why isn’t someone already doing this?</summary>
<p>
  
(best time to exploit opportunity to build telemedicine platform and products in machine learning/AI)

</p>
</details>


<details><summary>What are the key things about your field that outsiders don’t understand?</summary>
<p>
  
jekgnenelkgfnleka

</p>
</details>


<details><summary>What will keep you up at night?</summary>
<p>
  
(making sure that servers are running and that the platform is maintained and functioning so that doctors can have continual access to feedback from other verified doctors along with access to digital pathology tools)

</p>
</details>


<details><summary>What obstacles will you face and how will you overcome them?</summary>
<p>
  
jekgnenelkgfnleka

</p>
</details>


<details><summary>What is your main takeaway and take on the current state of the healthcare system and how patients and care providers alike can distribute care at a far more efficient cost? </summary>
<p>
  
(referring to entire pipeline and processes within the space and industries alike)

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

# Links for Other Viewing Formats

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
