# Module 01 : Get started with AI on Azure #

## 1. Introduction to AI ##

Key Workloads :

1. **Machine Learning** : We teach a computer model to make predictions and draw conclusions from data.
2. **Anomaly Detection** : Automatically detect errors or unusual activity in a system
3. **Computer Vision** : Capability of software to interpret the world visually through cameras, video and images
4. **Natural Language Processing** :Interpretation of written or spoken language and respond
5. **Knowledge Mining** : Extract information from large volumes of unstructured data to create a searchable knowledge store

## 2. Understand Machine Learning ##

Real world Example :

*The Yield* is a agricultural technology company based in Australia, using sensors, data from the weather and machine learning to help farmers make informed decisions related to weather, soil, and plant conditions.

**How does it work?**

There is a large amount of data available everywhere, then we use this data to train a machine learning model that can make **predictions** and **inferences**
based on the relationships they find.

### Example : Identify and catalog different species of wildflower ###

1. A team of botanists and scientists collect data from wildflower samples
2. The team labels the samples with the correct species
3. The labeled data is processed using an algorithm that finds relationships between between the features of the samples and the labeled species
4. The results of the algorithm are encapsulated in a model
5. When new samples are found by volunteers, the model can identify the correct species label.

### ML in Microsoft Azure ###

The Azure ML service is a cloud based platform for creating, managing and publishing machine learning models. It provides the following :

1. *Automated machine Learning* : Enables non-experts to quickly create an effective machine learning model from data.
2. *Azure ML Designer* : a GUI for no-code development of ML solutions.
3. *Data and Compute management* : Cloud-based data storage and compute resources taht professional data scientist can use to run data experiments code at scale.
4. *Pipelines* : Can define pielines to orchestrate model training, deployment and management tasks.

## 3. Understand Anomaly Detection ##

A ML based technique that analyzes data over time and identifies unusual changes.

Example of a racing car :

1. Sensors in the car collect telemetry such as engine revolutions, brake temperature, and so on.
2. An anomaly detection model is trained to understand expected fluctuations in the telemetry measurements over time.
3. If a measurement occurs outside of the normal expected range, the model reports an anomaly that can be used to alert the race engineer to call the driver in for a pit stop to fix the issue before it forces retirement from the race.

In MAzure, the Anomaly Detector provides an API to create anomaly detection solutions : <https://azure.microsoft.com/fr-fr/products/cognitive-services/anomaly-detector/>

## 4. Understand computer vision ##

This is an area of AI that deals with visual processing.
For example, the **Seeing AI**

Computer vision models and capabilities :

1. Image classification : for example, in a traffic, classify taxis, buses, cyclists ...
2. Object Detection : Identify individual objects within an image and identify their location with a bounding box.
3. Semantic Segmantation : Advanced, it  classify each individual pixels in the image to the object they belong.
4. Image Analysis : Combine ML with advanced image analysis to extract information. Example : A man with a Dog in a picture => Person with a Dog is return, not Person then Dog.
5. Face detection, analysis, and recognition : Locates human faces in an image. It can be combined to recognize individuals based on their facial features.
6. Optical character recognition : Detect and read text in images.

### CVS in MAz ##

1. Computer Vision : Images, video to extract descriptions, tags, objects and text.
2. Custom vision : Use this service to train custom image classification and object detection models using your own images.
3. Face : Build face detection and facial recognition solutions
4. Form Recognizer : Extract information from scanned forms and invoices.

## 5. Understand NLP ##

Area of AI that deals with creating software taht understands written and spoken language.

What is included in Maz :

- Language :understanding and analyzing text, training language models that can understand spoken or text-based commands, and building intelligent applications.
- Translator : translate text between more than 60 languages.
- Speech : recognize and synthesize speech, and to translate spoken languages.
- Azure Bot : conversational AI, the capability of a software "agent" to participate in a conversation. Developers can use the Bot Framework to create a bot and manage it with Azure Bot Service - integrating back-end services like Language, and connecting to channels for web chat, email, Microsoft Teams, and others

## 6. Understand Knowledge Mining ##

It involve extracting information from large volumes of often unstructured data to create a searchable knowledge store.

In Maz, you have access to Azure Cognitive Search, a private, enterprise, search solution that has tools for building indexes. The indexes can then be used for internal only use, or to enable searchable content on public facing internet assets.

Azure Cognitive Search can utilize the built-in AI capabilities of Azure Cognitive Services such as image processing, content extraction, and natural language processing to perform knowledge mining of documents. The product's AI capabilities makes it possible to index previously unsearchable documents and to extract and surface insights from large amounts of data quickly.

## 7. Challenge and risks with AI ##

1. Bias can affect results : A loan-approval model discriminates by gender due to bias in the data with which it was trained
2. Errors may cause harm : An autonomous vehicle experiences a system failure and causes a collision
3. Data could be exposed : A medical diagnostic bot is trained using sensitive patient data, which is stored insecurely
4. Solutions may not work for everyone : A home automation assistant provides no audio output for visually impaired users
5. Users must trust a complex system : An AI-based financial tool makes investment recommendations - what are they based on?
6. Who's liable for AI-driven decisions ? : An innocent person is convicted of a crime based on evidence from facial recognition – who's responsible?

## 8. Understanding Responsible AI ##

At Microsoft, AI software development is guided by a set of six principles, designed to ensure that AI applications provide amazing solutions to difficult problems without any unintended negative consequences.

### Fairness ###

AI systems should treat all people fairly. For example, suppose you create a machine learning model to support a loan approval application for a bank. The model should predict whether the loan should be approved or denied without bias. This bias could be based on gender, ethnicity, or other factors that result in an unfair advantage or disadvantage to specific groups of applicants.

Azure Machine Learning includes the capability to interpret models and quantify the extent to which each feature of the data influences the model's prediction. This capability helps data scientists and developers identify and mitigate bias in the model.

Another example is Microsoft's implementation of Responsible AI with the "Face service", which retires facial recognition capabilities that can be used to try to infer emotional states and identity attributes. These capabilities, if misused, can subject people to stereotyping, discrimination or unfair denial of services.

### Reliability and safety ###

AI systems should perform reliably and safely. For example, consider an AI-based software system for an autonomous vehicle; or a machine learning model that diagnoses patient symptoms and recommends prescriptions. Unreliability in these kinds of systems can result in substantial risk to human life.

AI-based software application development must be subjected to rigorous testing and deployment management processes to ensure that they work as expected before release.

### Privacy and Security ###

AI systems should be secure and respect privacy. The machine learning models on which AI systems are based rely on large volumes of data, which may contain personal details that must be kept private. Even after the models are trained and the system is in production, privacy and security need to be considered. As the system uses new data to make predictions or take action, both the data and decisions made from the data may be subject to privacy or security concerns.

### Inclusiveness ###

AI systems should empower everyone and engage people. AI should bring benefits to all parts of society, regardless of physical ability, gender, sexual orientation, ethnicity, or other factors.

### Transparency ###

AI systems should be understandable. Users should be made fully aware of the purpose of the system, how it works, and what limitations may be expected.

### Accountability ###

People should be accountable for AI systems. Designers and developers of AI-based solutions should work within a framework of governance and organizational principles that ensure the solution meets ethical and legal standards that are clearly defined.
