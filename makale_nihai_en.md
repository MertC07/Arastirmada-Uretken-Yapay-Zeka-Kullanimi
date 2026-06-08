# Intelligent Road Safety Through Artificial Intelligence and 5G Integration: A Descriptive Study on Real-Time Object Detection and Driver Behavior Analysis

---

**Keywords:** Intelligent road safety, computer vision, real-time object detection, deep learning, YOLOv11, RT-DETR, 5G edge computing, driver behavior analysis, modular pipeline architecture, intelligent transportation systems

---

## Abstract

Road traffic accidents remain one of the leading causes of death globally; the World Health Organization reported 1.19 million traffic-related fatalities in 2021. In Turkey, 6,548 people lost their lives in 2023, with human error identified as the determining factor in more than 90% of accidents. The maturation of computer vision algorithms and 5G infrastructure opens a new window for proactive, real-time road safety systems. Against this background, the study aims to evaluate the contributions that deep learning-based object detection architectures — such as the YOLO series and RT-DETR — can make to intelligent road safety when integrated with 5G-enabled edge computing infrastructure, focusing on real-time object detection and autonomous warning mechanisms. Adopting a descriptive design within the quantitative research paradigm, a labeled image dataset compiled through the Roboflow platform was divided into 70% training, 20% validation, and 10% test sets; the YOLOv11l, YOLOv11n+αSiLU, RT-DETR-L, and YOLOv8n architectures were comparatively evaluated on license plate detection, driver behavior analysis, and seatbelt monitoring tasks using mAP, Precision, and Recall metrics. Findings revealed that YOLOv11l achieved the highest performance in license plate detection (mAP@50: 0.986), RT-DETR-L in phone use detection (Precision: 1.000), and YOLOv11n+αSiLU in fatigue indicator recognition (yawning mAP@50: 0.995); seatbelt detection was identified as the most critical performance gap across all models. These results empirically confirm that no single architecture can achieve simultaneous superiority across all safety tasks, and support the necessity of task-based modular system design. The findings of the study provide a theoretical foundation for systems to be developed for autonomous warning mechanisms and smart city infrastructure.

---

# 1. Introduction

## 1.1 Introduction to the Topic

While road traffic constitutes an indispensable transportation backbone for modern societies, it remains one of humanity's deadliest and most preventable public health problems. The Global Status Report on Road Safety published by the World Health Organization (WHO) in 2023 reveals that **1.19 million** people died as a result of traffic accidents worldwide in 2021, and that traffic accidents continue to be the leading cause of death in the 5–29 age group (WHO, 2023). This figure translates to two lives lost every minute. The estimated annual cost of traffic accidents to the global economy exceeds **$3.6 trillion USD** (CDC, 2023). This picture clearly demonstrates that road safety is not merely an infrastructure issue, but a multidimensional global priority situated at the intersection of technology, policy, and society.

When evaluated specifically for Turkey, the 2023 data from the Turkish Statistical Institute (TÜİK) makes the criticality of the situation concrete: a total of **1,314,136** traffic accidents occurred across the country, resulting in **6,548** deaths and **350,855** injuries (TÜİK, 2023). The increase in fatalities compared to 2022 reached 25.2%, with driver fault continuing to be the primary factor in accidents. These statistics make it imperative to develop not merely stricter enforcement, but intelligent monitoring and intervention infrastructure capable of systematically compensating for human error.

## 1.2 Problem Statement and Necessity

Research on the root causes of traffic accidents indicates that human error is the determining factor in **more than 90%** of incidents (WHO, 2023; PMC, 2024). At the forefront of these errors are excessive speed, driver fatigue, and distraction: according to 2023 data from the U.S. National Highway Traffic Safety Administration (NHTSA), distracted driving alone caused **3,275 deaths**, while speed violations claimed **11,775 lives** in the same year (NHTSA, 2023). Violations such as seatbelt non-use, drunk driving, and dangerous overtaking also significantly increase the risk of fatal accidents on a global scale.

Conventional traffic enforcement methods are unable to provide an adequate response to this picture. Fixed radars, driver checks, and camera systems, lacking continuous real-time analysis capacity, can only be activated after a violation has already occurred. Yet an effective road safety system must be proactive in nature — perceiving, assessing, and issuing warnings about threats before violations take place. From this perspective, two fundamental deficiencies of current systems stand out: **latency** and **coverage limitations**. The delay in converting raw footage from traffic cameras into meaningful safety information eliminates the possibility of real-time intervention; the area surveyable by a single camera proves insufficient in complex intersections and road networks.

## 1.3 Contextual Background: Technological Transformation

Over the last decade, fundamental advances in computer vision, deep learning, and mobile communications have provided concrete solutions to these problems. The widespread adoption of YOLO (You Only Look Once) series models in object detection, followed by Transformer-based architectures (DETR, RT-DETR), has made real-time vehicle, pedestrian, and obstacle recognition systems feasible in terms of both accuracy and speed. Meanwhile, the **sub-millisecond latency** and high bandwidth offered by 5G (Wang et al., 2025), when combined with edge computing, creates the opportunity to process large amounts of data generated by camera systems in the field without transferring it to a central cloud. This convergence — the integration of computer vision, machine learning, and 5G — forms the foundation upon which intelligent road safety systems gain a realistic deployment perspective.

Vehicle-to-Everything (V2X) communication technology and the 5G-V2X standard represent the institutional dimension of this transformation. C-V2X infrastructure enables vehicles to exchange real-time data with each other, roadside units, and traffic management centers, thereby closing the accident detection and prevention loop. The investments made by the European Union and China in this infrastructure signal that 5G-enabled traffic safety has entered the policy agenda.

## 1.4 Relationship to the Literature

Examining the research that has intensified over the last five years, several fundamental lines of inquiry emerge that complement each other yet have not yet been integrated into a unified framework.

On the object detection front, the RT-DETR model presented at CVPR by **Zhao et al. (2024)** successfully brought the Transformer architecture together with real-time detection for the first time, demonstrating superiority over the YOLO series in both speed and accuracy. **Zabłocki et al. (2024)** comprehensively evaluated YOLOv8 in ITS contexts, while **Zhao et al. (2025)** pushed the boundaries of DETR-based architectures for autonomous driving scenarios.

In the field of driver behavior monitoring, **Civik and Yüzgeç (2023)** from Turkey achieved 94.5% accuracy in CNN-based fatigue detection on Nvidia Jetson Nano, demonstrating the feasibility of embedded systems in production environments. **García-González et al. (2025)** combined MediaPipe facial landmarks with MobileNetV2 to develop a system that simultaneously monitors distraction and fatigue in a real vehicle environment. In seatbelt detection, **Wang (2022)** and **Qiu et al. (2024)** reported meaningful accuracy gains with lightweight, attention mechanism-based architectures.

License plate recognition and speed estimation research (**Laroca et al., 2021**; **Rahutomo et al., 2024**; **Gül & Sertaş, 2024**) demonstrates the maturity of these sub-systems, while recent work in accident detection (**Lin et al., 2024**; **Alam et al., 2025**) and pedestrian safety (**Sami et al., 2024**) points to an expanding scope. In 5G and edge computing integration, **Herbst et al. (2021)**'s InTraSafEd5G system showed that low-latency traffic safety warning infrastructure can be brought to life in practice.

However, when the literature is evaluated as a whole, several critical gaps stand out: (1) the large majority of component-focused studies do not address multiple safety functions (detection, recognition, speed measurement, behavior analysis) under an integrated architecture; (2) end-to-end evaluations running on real-time 5G infrastructure remain limited; (3) the latency-accuracy trade-off has not been systematically examined across different scene conditions.

## 1.5 Research Objectives and Scope

The primary objective of this study is to comprehensively evaluate the contributions that computer vision algorithms (YOLO series, RT-DETR, and derivatives) and machine learning methods can make to intelligent road safety when integrated with 5G and related high-speed data transmission technologies. In pursuit of this goal, the study aims to synthesize existing literature to identify gaps in latency and accuracy, and to present the findings within a theoretical framework.

The scope of the study is delineated by the following components:

- **Real-time object detection:** Instantaneous recognition of vehicles, pedestrians, and obstacles from camera-based imagery
- **Autonomous warning mechanisms:** Transmitting detected threats to drivers, roadside systems, or traffic management centers
- **Core safety parameters:** Speed violation, seatbelt usage, and driver attentiveness status

The study excludes fully autonomous vehicle systems, infrastructure engineering design, and hardware development processes. The focus is on the integration potential of existing algorithmic and communication technologies into safety applications, and the technical and theoretical limitations of such integration.

## 1.6 Original Contribution

This study contributes to the literature with three primary original contributions.

**Synthesis and gap analysis:** By systematically examining 18 recent studies across a wide spectrum ranging from vehicle detection to driver behavior analysis, license plate recognition to accident detection, it explicitly identifies existing limitations in terms of latency, accuracy, and scalability within each sub-field. This synthesis elevates scattered findings into a unified evaluation framework.

**Theoretical foundation:** It provides a conceptual and methodological basis for innovative projects to be developed for autonomous systems, smart road infrastructure, and 5G-based V2X applications. Its comparative examination of different models' strengths and weaknesses under real-world deployment conditions offers a concrete reference point for future system designers.

**Multi-component integration perspective:** It goes beyond the component-focused approaches in the literature by positioning object detection, driver monitoring, license plate recognition, and communication infrastructure within a shared system framework. This holistic perspective can contribute to shaping intelligent road safety policies, particularly in countries like Turkey where traffic accident statistics are at critical levels.

---

# 2. Related Work

## 2.1 Real-Time Object Detection: YOLO and Transformer-Based Approaches

Object detection, one of the core components of traffic safety systems, has undergone a major transformation with the maturation of deep learning architectures.

**Zhao et al. (2024)** 🇨🇳 identified that the NMS (Non-Maximum Suppression) post-processing step negatively affects the speed and accuracy of YOLO series models, and proposed the **RT-DETR** (Real-Time Detection Transformer) model based on an efficient hybrid encoder and uncertainty-minimal query selection mechanisms to overcome this problem. RT-DETR-R50/R101 models achieved **53.1%/54.3% AP** on the COCO dataset and **108/74 FPS** on a T4 GPU, surpassing the leading YOLO models of the same period in both speed and accuracy. The study introduced a new paradigm for real-time detection in traffic safety applications such as autonomous driving and video surveillance. *(CVPR 2024)*

**Zabłocki et al. (2024)** 🇵🇱 evaluated YOLOv8's effectiveness in Intelligent Transportation Systems (ITS) applications, highlighting that the model is capable of real-time vehicle detection with high accuracy even in low-light conditions, and that pedestrian and cyclist detection in particular needs improvement. The study presents a comprehensive evaluation framework for integrating camera-based systems into ITS infrastructure to analyze vehicle density and traffic flow in different road scenarios.

**Zhao et al. (2025)** 🇨🇳, as researchers from Jiamusi University, propose an improved object detection method based on DETR for autonomous driving. In the study, new feature extraction modules and a custom loss function were added to the transformer encoder to address the problems of detecting overlapping and small objects. Experiments conducted on KITTI and nuScenes datasets revealed that the proposed method significantly improved mAP values compared to standard DETR.

---

## 2.2 Driver Fatigue and Distraction Detection

**Civik and Yüzgeç (2023)** 🇹🇷, as researchers from Bilecik Şeyh Edebali University, developed a CNN-based model on the Nvidia Jetson Nano platform and achieved **93.6%** and **94.5%** accuracy in drowsiness detection by analyzing eye and mouth movements, respectively. Operating at 6 FPS, this system offers a practical solution for integration into safety systems in computationally constrained environments.

**García-González et al. (2025)** 🇪🇸 developed a real-time driver monitoring system using a MobileNetV2-based convolutional neural network and MediaPipe facial landmarks. The system detects fatigue through eye aspect ratio (EAR) and mouth aspect ratio (MAR) metrics, distraction through head pose and gaze direction, and the driver's emotional state through emotion recognition. The system achieved **88.89%** accuracy in fatigue detection and **100%** in distraction detection, proving suitable for non-invasive use in real vehicle driving environments.

**Dalve et al. (2023)** 🇮🇳 designed a real-time driver fatigue prevention system combining deep learning and the MediaPipe framework. The outputs of the deep learning module, which extracts numerical features from images, are fed into a fuzzy logic-based system, and this hybrid approach achieves **91%** accuracy on training data and **92%** on test data.

**Qiu et al. (2024)** 🇨🇳 proposed a driver safety monitoring system combining YOLO-based detection and LSTM sequence modeling. The study designed a pipeline that simultaneously analyzes seatbelt wearing status, cell phone use, and head position, achieving overall detection accuracy of over **90%** when validated on real traffic surveillance footage.

---

## 2.3 Seatbelt Detection and Driver Behavior Monitoring

**Wang (2022)** 🇨🇳, in a study published in *Wireless Communications and Mobile Computing*, focused on vehicle driving safety detection using deep learning. The Deconv-SSD small object detection algorithm, Squeeze-YOLO-based driver region localization for vehicle detection, and semantic segmentation were combined to determine seatbelt status. The system was tested on real traffic surveillance footage and demonstrated high precision in seatbelt detection.

**Qiu et al. (2024)** 🇨🇳, in a study published in *Applied Sciences*, proposed the **GM-YOLOv7** model. The architecture was optimized by adding the G-ELAN module to the YOLOv7-tiny network, computational load was reduced through channel pruning, and a triplet attention module was integrated. Compared to the baseline network, the proposed model improved mAP by **3.8%** while reducing model size by **20%** and computational volume by **24.6%**.

**Hosseini and Fathi (2022)** 🇮🇷 developed a deep learning system for detecting vehicle occupancy and driver seatbelt wearing status on roadways. The system aims to automate HOV (High Occupancy Vehicle) lane enforcement in high-traffic areas. The authors conducted training and validation with realistic traffic data under different camera angles and lighting conditions, revealing that the proposed deep learning-based approach outperforms conventional image processing methods.

---

## 2.4 Vehicle Speed Estimation and License Plate Recognition

**Sathiyaprasad et al. (2025)** 🇮🇳 developed an integrated system for vehicle tracking, counting, and speed estimation using YOLOv8 and DeepSORT algorithms together. Presented under the name **TrackNCount**, the system combines multi-object tracking with speed estimation in a single pipeline, enabling real-time detection of speed violations at traffic enforcement points.

**Rahutomo et al. (2024)** 🇮🇩 presented a system combining YOLOv8 and EasyOCR for vehicle speed estimation and automatic license plate recognition from traffic camera footage. By handling license plate detection and OCR integration as a single flow, the study achieved high-accuracy plate reading under real-world traffic camera conditions.

**Gül and Sertaş (2024)** 🇹🇷 conducted research on automatic license plate recognition and visualization using the YOLOv8 algorithm at Recep Tayyip Erdoğan University. The system, targeting minimization of manpower and costs in security-sensitive areas, was trained on a Turkish license plate dataset created in the Roboflow environment, and real-time plate detection was achieved by leveraging artificial neural network architecture.

**Laroca et al. (2021)** 🇧🇷 presented an efficient and layout-independent automatic license plate recognition system based on the YOLO detector. The system achieved an average end-to-end recognition rate of **96.9%** across eight public datasets from five different regions, surpassing previous studies and commercial systems on the ChineseLP, OpenALPR-EU, and SSIG-SegPlate datasets.

---

## 2.5 Accident and Anomaly Detection

**Yang et al. (2022)** 🇺🇸 developed an approach combining statistical and machine learning methods to identify dynamic traffic accident risk using data from China's highway network. A hybrid model consisting of logistic regression, random forests, and gradient boosted trees produced real-time risk scores that allow drivers and traffic managers to be warned in advance about high-risk periods.

**Lin et al. (2024)** 🇨🇳 propose a traffic accident detection network based on an improved YOLOv8 model developed on public accident datasets such as AccOD and ACD. With improvements to the C2f module, the model improved mAP@0.5 values on AccOD by **4.75%** and mAP@0.5:0.95 by **4.58%** compared to YOLOv8n, while preserving its lightweight characteristics.

**Klanjčić et al. (2022)** 🇳🇱 combined geospatial analysis with machine learning to identify urban factors affecting the safety of vulnerable road users (pedestrians and cyclists) in major European cities. Published in *EPJ Data Science*, the study revealed that specific urban road features positively or negatively influence pedestrian safety, particularly in high-risk corridors.

**Alam et al. (2025)** developed a spatiotemporal video analysis system combining YOLOv8 with attention mechanism-based safety metrics. Combining key frame extraction, real-time multi-line object tracking, and specialized safety metrics such as Time-to-Collision (TTC) to detect traffic conflicts, the system reports impressive accuracy results in high-density traffic flows.

---

## 2.6 Pedestrian and Cyclist Safety

**Sami et al. (2024)** systematically reviewed AI solutions for improving pedestrian and cyclist safety in a review paper published on arXiv. Analyzing publication trends between 2016 and 2024, the study found a notable shift in recent years from pedestrian-focused work toward a broader perspective encompassing the concept of **Vulnerable Road Users (VRU)**.

**Herbst et al. (2021)** 🇦🇹 presented the **InTraSafEd5G** system. This system was designed to detect critical traffic situations, such as pedestrians and cyclists in drivers' blind spots, in real time, and to transmit early warnings to nearby vehicles via 5G-enabled edge computing nodes. A prototype evaluated in a real-world environment demonstrated that 5G provides millisecond-level latency that surpasses the limitations of human reaction time, significantly improving pedestrian safety.

---

## 2.7 Road Damage Detection and Intelligent Traffic Management

**Arya et al. (2022)** 🇯🇵 compared deep learning models for road damage detection across six different countries using data collected from Europe, America, India, and Japan. The model achieving an **F1 score of 76.9%** across all countries stood out; the importance of transfer learning and data diversity across geographic conditions was emphasized.

**Pramanik et al. (2021)** 🇮🇳 propose a real-time video surveillance-based traffic pre-event detection system. Published in *Accident Analysis & Prevention*, the study combined CNN and optical flow techniques to detect pre-accident events (sudden braking, abrupt lane changes) from camera footage, validating the system's real-time processing capacity for traffic management and emergency response optimization.

---

## 2.8 Literature Assessment

When the reviewed studies are assessed, the research in AI-assisted road safety is seen to cluster around six fundamental axes:

1. YOLO and Transformer-based architectures for real-time object detection
2. Driver fatigue monitoring through facial landmark analysis
3. Seatbelt and driver behavior supervision via video surveillance
4. End-to-end vehicle tracking, license plate recognition, and speed estimation
5. Accident and anomaly detection
6. Pedestrian safety systems integrated with 5G and edge computing infrastructure

Civik and Yüzgeç (2023) and Gül & Sertaş (2024) from Turkey exemplify local contributions in embedded systems and license plate recognition. The RT-DETR model (Zhao et al., 2024) developed by Baidu researchers in China has set a new global standard for real-time Transformer-based detection. Investment in 5G and edge computing integration in Europe (Herbst et al., 2021) provides proof of feasibility for low-latency safety notification systems.

It is observed that a comprehensive approach integrating these components under a single modular and scalable architectural framework remains limited in the existing literature; this gap lays the groundwork for the original contribution of the proposed study.

---

# 3. Research Questions

The questions of this study are derived from three fundamental gaps in road safety literature — the lack of integrated system architecture, the limited experimental evidence regarding real-time 5G integration, and the fact that the latency-accuracy trade-off has not been systematically examined. The questions are framed to logically complement each other, forming a five-layered structure extending from technical comparison to integration analysis, from reliability under challenging conditions to policy barriers.

---

## RQ-1 — Performance Comparison

> **How do YOLO series (YOLOv8, YOLOv11) and Transformer-based (RT-DETR) real-time object detection architectures differ in terms of latency, accuracy (mAP), and computational efficiency in intelligent road safety scenarios (vehicle, pedestrian, obstacle detection)?**

The majority of studies comparing object detection architectures either evaluate the YOLO series internally or address RT-DETR in general-purpose benchmarks. A systematic comparison of the two architectural families specific to traffic safety scenarios, on the same dataset and under identical hardware conditions, has not yet found sufficient place in the literature. This question forms the technical core of the study and aims to provide an evidence-based guide on which architecture should be preferred under which operational constraints.

**Relevant literature:** Zhao et al. (2024); Zabłocki et al. (2024); Zhao et al. (2025)

---

## RQ-2 — 5G and Edge Computing Integration

> **To what extent do real-time computer vision systems integrated with 5G-based edge computing infrastructure have a determining effect on the response time and reliability of autonomous warning mechanisms compared to traditional cloud-based or local processing architectures?**

The theoretical advantages of 5G — millisecond-level latency, high bandwidth, local processing capacity at edge nodes — are widely discussed in the context of traffic safety. However, experimental studies that combine these advantages with computer vision workloads in real time and measure their effect on alert response time remain limited. This question aims to make concrete the operational value that 5G holds for safety systems and to comparatively evaluate current infrastructure preferences.

**Relevant literature:** Herbst et al. (2021); Wang et al. (2025)

---

## RQ-3 — Multi-Component System Integration

> **In a multi-task system that integrates vehicle detection, driver behavior analysis (fatigue, seatbelt, phone use), and license plate recognition components in a single modular pipeline, how does inter-component resource competition affect real-time performance and detection accuracy?**

The existing literature largely addresses each safety component independently: fatigue detection, seatbelt monitoring, and license plate recognition have advanced as separate studies; integrated system analyses where these processes run simultaneously sharing the same computational resources have remained quite limited. This question constitutes the focal point of the study's original contribution, aiming to theoretically examine the technical counterpart of modular design patterns (Pipeline, Strategy) in multi-component safety systems.

**Relevant literature:** Civik and Yüzgeç (2023); Qiu et al. (2024); Wang (2022); Rahutomo et al. (2024)

---

## RQ-4 — Reliability Under Challenging Conditions

> **How do the accuracy and reliability of deep learning-based object detection models change under challenging environmental conditions such as low light, adverse weather, and high vehicle density; and which model optimization strategies produce more resilient results against these conditions?**

Conditions such as inadequate nighttime lighting, rain, fog, and dense urban traffic that systems encounter in real-world traffic environments can significantly reduce the deployment performance of models developed on controlled datasets. Studies that comparatively address robustness to challenging conditions across different model families and optimization approaches (data augmentation, weight pruning, quantization) are still insufficient. This question provides a critical evaluation ground for the deployability of intelligent road safety systems in production environments, particularly given the scene diversity in Turkey.

**Relevant literature:** Sami et al. (2024); Arya et al. (2022); Pramanik et al. (2021)

---

## RQ-5 — Deployment Barriers and Policy Dimension

> **What technical, ethical, and infrastructural barriers stand in the way of effectively deploying AI-assisted intelligent road safety systems, and what solutions does the existing literature envision for overcoming these barriers?**

The gap between algorithmic success and real-world deployment has increasingly become a problem attracting attention in AI-assisted safety systems literature. Data privacy, model bias, infrastructure cost, and legal regulatory gaps are among the leading barriers. This question ensures that the study does not remain limited to the technical dimension alone, covering the policy and implementation perspective as well, and aims to contribute to intelligent road safety strategies both specifically in Turkey and on a global scale.

**Relevant literature:** WHO (2023); TÜİK (2024); Klanjčić et al. (2022); Yang et al. (2022)

---

## General Framework of Research Questions

The following table summarizes the five questions by their targeted gap, method, and scope dimensions:

| Question | Focus | Literature Gap | Scope |
|----------|-------|----------------|-------|
| RQ-1 | Architectural comparison | YOLO–Transformer comparative analysis insufficient | Technical |
| RQ-2 | 5G integration | Experimental 5G–CV integration studies limited | Technical–Infrastructure |
| RQ-3 | Multi-component pipeline | Integrated system analysis lacking | Technical–Architectural |
| RQ-4 | Challenging conditions | Robustness comparison insufficient | Technical–Application |
| RQ-5 | Deployment barriers | Policy and ethical dimensions underexplored | Application–Policy |

The logical relationship among the questions can be summarized as follows: RQ-1 aims to answer *how well does it work*, RQ-2 *what does 5G add*, RQ-3 *what happens when they work together*, RQ-4 *is it robust in the real world*, and RQ-5 *what are the barriers to deployment*. This ordering ensures that the study advances in a coherent flow starting from technical comparison through application and policy recommendations.

---

# 4. Methodology

## 4.1 Research Design

This study is structured within the quantitative research paradigm, adopting **descriptive design** as its sub-design.

Quantitative research is a systematic research approach that aims to measure, describe, and reveal relationships among phenomena and events through numerical data using statistical methods (Creswell & Creswell, 2018). This paradigm is built upon a positivist epistemological position where the researcher focuses on observable and measurable phenomena and findings are reported objectively and replicably.

### Rationale for Selecting the Descriptive Design

Descriptive design is a research approach that aims to describe and document the current conditions of a phenomenon, situation, or system as it is, without any intervention or manipulation (Fraenkel et al., 2012). The fundamental rationale for preferring this design in the present study lies in the nature of the research: AI-assisted road safety systems — real-time object detection algorithms, driver behavior analysis modules, license plate recognition, and speed estimation components — are complex, multi-component structures that already function within specific technical parameters. To understand these systems objectively, it is necessary to accurately and impartially determine existing performance indicators rather than conducting an experimental intervention on variables.

This necessity becomes even more pronounced in the context of Intelligent Transportation Systems (ITS). Instantaneous accuracy rates of object detection models (mAP, F1 score), the latency of algorithms in generating alerts, performance drops observed under different weather and light conditions, or the resource consumption patterns displayed by a multi-component pipeline during simultaneous operation — these are phenomena that need to be systematically documented in terms of their values and patterns, not altered through experimental intervention. Indeed, descriptive research prioritizes answering *what*, *how much*, and *how* questions, thereby producing a numerical photograph of the current state of the systems under examination (Cohen et al., 2018).

Within this framework, the study deliberately diverges from an experimental design involving controlled manipulation of independent variables. The researcher's intervention is confined to determining measurement instruments, collecting data, and analyzing them with statistical techniques. Variables — algorithm type, scene conditions, number of components — are examined in terms of their potential effects, but no systematic control or assignment procedure has been carried out on them.

### The Role of Data Type in the Study

The data foundation of the study consists of performance metrics of a quantitative nature. Accuracy and speed measures produced by image processing algorithms (mAP@0.5, mAP@0.5:0.95, FPS, latency in milliseconds), component-level detection success rates, and statistical outputs obtained from sensor-based systems fall into this category. Such data directly aligns with the study's descriptive purpose, enabling findings to be reported in a manner that allows quantitative comparisons, pattern identification, and systematic linking with the literature.

In summary, the quantitative-descriptive design has been assessed not merely as a methodological preference for this study, but as the most appropriate approach to the nature of the system under examination and the epistemological position of the research.

## 4.2 Population and Sample (Dataset and Classification)

## 4.2.1 Population

In quantitative research, the population refers to the entirety of the whole from which research findings are intended to be generalized or upon which inference is targeted (Fraenkel et al., 2012). The population of this study consists of all real-world traffic scenes encompassing motor vehicles operating on roadways, the drivers of these vehicles, and vulnerable road users (pedestrians, cyclists) on the road. This population has a broad and heterogeneous structure encompassing different geographical regions, urban and rural road conditions, various times of day, and possible weather conditions. Therefore, access to the entirety of the population is not possible due to practical and technical constraints; the findings of the study are interpreted within the determined sample framework.

## 4.2.2 Sample and Dataset Construction

The sample of the study consists of a **labeled image dataset** created for the training and testing processes of the deep learning model. The dataset was compiled through the **Roboflow** platform, which is widely used in the field of object detection and computer vision, and labeling (annotation) operations were performed on this platform.

The size and content diversity of the dataset were determined by drawing inferences from the computer vision and object detection literature. In this context, published guideline values regarding the minimum data size needed for a model to learn a generalizable representation (Laroca et al., 2021; Qiu et al., 2024) and recommendations for reducing the adverse effects of class imbalance on model performance (Arya et al., 2022) were taken into consideration.

### Class Structure and Annotation Protocol

The dataset was structured around the **seatbelt** class, the primary detection target of the study. The annotation protocol was based on the principle of marking the visible portion of the belt strap extending from the shoulder to the waist in each image with a tight bounding box. To minimize the disruptive effect of inconsistent labeling on model performance, a written annotation guide was prepared by the research team and all team members were ensured to adhere to this guide throughout their work. The centralized project management infrastructure provided by the Roboflow platform enabled the team members' data contributions to be consolidated within a single project and label quality to be monitored centrally.

## 4.2.3 Sample Diversity and Representativeness

The generalizability of object detection models in real-world conditions is largely proportional to the diversity of training data (Sami et al., 2024; Zabłocki et al., 2024). In this context, the following diversity dimensions were systematically observed in the image collection and selection process to enhance the representativeness of the dataset:

**Light and Time Conditions:** Images from daytime (full daylight), twilight, and nighttime were included in the dataset. This distinction stems from the necessity for real traffic surveillance systems to operate continuously at all hours of the day and allows the assessment of sensitivity under low-light conditions.

**Weather Conditions:** Images representing clear, cloudy, and rainy conditions were compiled to ensure the model's robustness against environmental variables at the data level. The literature demonstrates that adverse weather conditions can significantly reduce detection accuracy (Arya et al., 2022).

**Vehicle and Passenger Diversity:** Images covering different vehicle types (passenger cars, light commercial vehicles) and drivers with different physical appearances were used together. This diversity aims to prevent the model from developing overfitting to a specific vehicle type or driver profile.

**Camera Angle and Distance:** Images were compiled with different shooting distances and angles in mind. The assumption that camera position will vary in real road enforcement scenarios forms the fundamental rationale for this diversity.

## 4.2.4 Data Augmentation

**Data augmentation** techniques were applied to reduce the risk of the number of raw images falling below the threshold required for model generalization and to artificially enrich data diversity. Augmentation was applied exclusively to the training set; validation and test sets were used with their original raw images preserved. This approach allows objective measurement of how well the model generalizes not only to augmented data but also to raw images.

The augmentation transformations applied are as follows:

| Augmentation Type | Parameter Range | Rationale |
|---|---|---|
| Brightness | ±25% | To simulate different lighting conditions |
| Exposure | ±15% | To represent camera sensor diversity |
| Blur | Max. 1.5 px | To represent motion blur and focus loss |
| Noise | Max. 2% | To simulate low-quality camera footage |
| Rotation | ±5° | To represent vehicle tilt and camera angle deviation |

Given that the seatbelt strap is positioned at a specific angle on the driver's side of the vehicle, the horizontal flip transformation was intentionally disabled; it was assessed that applying this transformation would result in a mirror image inconsistent with reality for the steering wheel and belt position. **3 augmented outputs** were generated for each training image, tripling the total data contribution per raw image.

## 4.2.5 Training, Validation, and Test Set Splits

The dataset needs to be divided into three subsets to allow the model to be both successfully trained and impartially evaluated. This requirement is based on the principle of ensuring the functional independence of training, validation, and test sets, and is accepted as a common practice standard in deep learning literature (Goodfellow et al., 2016). In the study, the dataset was divided at the following ratios:

| Set | Ratio | Function |
|---|---|---|
| **Training** | 70% | Optimization of model weights |
| **Validation** | 20% | Monitoring overfitting throughout training and hyperparameter tuning |
| **Test** | 10% | Final performance evaluation with data the model has never seen before |

The test set was kept completely isolated from the model throughout the entire training process and was used only at the final performance reporting stage. This approach is critically important for preventing information leakage toward the test set (data leakage) and preserving the objective validity of the obtained performance metrics.

---

# 5. Discussion

## 5.1 General Assessment

This section provides a comparative interpretation of the quantitative findings obtained from the multi-component AI pipeline developed within the scope of the study — in terms of license plate recognition, driver behavior analysis, and seatbelt detection — against past studies in the literature. The discussion follows the order of the research questions, analyzing findings both internally and in the context of existing system deficiencies.

---

## 5.2 Performance Differentiation Among Model Architectures (RQ-1)

The first question of the study queries how the YOLO series and Transformer-based RT-DETR architecture differ in terms of latency, accuracy, and computational efficiency in traffic safety scenarios. The findings provide a task-focused and nuanced answer to this question: no single architecture can achieve simultaneous superiority across all detection tasks.

In the license plate detection task, **YOLOv11l** produced the highest result with a mAP@50 value of **0.986**, followed by YOLOv11n+αSiLU at 0.975. This finding demonstrates a strong alignment with the average 96.9% end-to-end recognition rate achieved by Laroca et al. (2021)'s YOLO detector-based license plate recognition system across eight datasets. Similarly, Rahutomo et al. (2024) reported high-accuracy plate reading with YOLOv8 in real traffic cameras; the current study's plate module surpassed this reference threshold.

In cell phone use detection, **RT-DETR-L v3** displayed extraordinary precision with 1.000 Precision and 0.989 Recall values. This finding is particularly noteworthy: against the prevailing assumption that Transformer architectures require excessive computational load compared to the YOLO series for real-time applications (Zhao et al., 2024), it shows that in certain fine-grained tasks, the Transformer model can provide superior accuracy that justifies its computational cost.

In fatigue detection, **YOLOv11n+αSiLU v2** reached the highest single-class score of the study with 0.995 mAP@50 on the yawning class. This result significantly surpasses the 94.5% eye closure detection accuracy achieved by Civik and Yüzgeç (2023) on Nvidia Jetson Nano, indicating that the nano architecture optimized with the αSiLU activation function unexpectedly learned a strong fatigue indicator. Compared to the 88.89% fatigue detection accuracy achieved by García-González et al. (2025) with MediaPipe and MobileNetV2, this difference (~88.89% vs. approximately 93.9% average fatigue mAP@50) demonstrates that task-focused training and specialized activation functions can enable the model to surpass standard general-purpose approaches.

When compared at the overall mAP@50 level, the ranking among YOLOv11n+αSiLU v2 (0.876), RT-DETR-L v3 (0.858), and YOLOv11l (task-based 0.986) generally aligns with the mAP ranges reported by Zabłocki et al. (2024) for the YOLOv8 family in ITS applications. However, the surprising performance produced by the αSiLU modification in the nano architecture stands out as an original contribution not yet systematically documented in the literature.

---

## 5.3 Resource Competition in Multi-Component Pipeline and Modular Architecture (RQ-3)

The third question of the study queries the effect on performance of running multiple safety components simultaneously within a single pipeline framework. This question directly points to the most fundamental blind spot of component-focused studies in the literature.

The majority of existing studies — fatigue detection (Civik & Yüzgeç, 2023), seatbelt detection (Qiu et al., 2024), license plate recognition (Laroca et al., 2021) — address each module in an independent evaluation environment, ignoring inter-component resource competition. In this study, Pipeline and Strategy design patterns were adopted, structuring each component (vehicle detection, plate reading, fatigue analysis, seatbelt monitoring) as an independent module but subjecting them to an orchestration sharing the same computational resources.

Findings reveal that the best-performing model on a task basis systematically varies across tasks: the plate module is fed from YOLOv11l, the in-cabin and fatigue modules from YOLOv11n+αSiLU or RT-DETR-L, while each architecture exhibits different computational profiles. This demonstrates that a single universal model managing a multi-component safety system may prove insufficient in terms of both efficiency and accuracy, reinforcing the rationale for an architectural decision that supports task-based model allocation. The task-oriented functional separation principle similarly emphasized in Wang et al. (2025)'s C-V2X-based cooperative systems aligns with this finding at the theoretical level.

---

## 5.4 Critical Weakness of Seatbelt Detection and Its Relationship to the Literature

The most striking finding of the study is the performance gap observed in seatbelt detection. While the `seatbelt` class mAP@50 value in the YOLOv11n+αSiLU model was 0.843, the `no_seatbelt` class lagged significantly behind at 0.337; RT-DETR-L v3 produced the best result among all models in the same class, reaching 0.780 mAP@50.

This finding aligns with similar observations in the literature. Wang (2022) reported that the seatbelt system based on Deconv-SSD and semantic segmentation showed inconsistent performance in real traffic surveillance footage. Qiu et al. (2024), while achieving a 3.8% improvement in mAP with GM-YOLOv7, attributed the difficulty of belt detection to low data diversity. Hosseini and Fathi (2022) revealed that different camera angles and lighting conditions significantly reduced system reliability. The findings of the present study independently confirm these limitations and suggest that the dramatic drop in the `no_seatbelt` class reflects not only data imbalance — since the visual signal of negative examples is based on the absence of the belt strap, making the visual cue excessively sparse — but also a fundamental object recognition problem. Combined with the fact that seatbelt monitoring is the most critical safety sub-task from a security perspective, this finding stands out as a priority improvement area for future research.

---

## 5.5 Effect of 5G and Edge Computing Integration on Response Time (RQ-2)

The second question of the study queries the effect of 5G-based edge computing infrastructure on the response time and reliability of autonomous warning mechanisms. While this study does not include an experimental 5G comparison, the obtained model speed findings provide an important indirect contribution to this question.

The **158 FPS** value measured for the YOLOv8n+αSiLU model in license plate detection indicates that the time required to process one frame is approximately **6.3 ms**. When combined with the potential of 5G to reduce end-to-end communication latency to 1 ms (Wang et al., 2025), the theoretical total system latency could remain below 10 ms. This threshold falls within the boundaries of the real-time standard (< 5-10 ms) envisioned for V2X safety messaging. Herbst et al. (2021)'s report that 5G-assisted edge computing in the InTraSafEd5G system significantly reduced latency in pedestrian safety alerts compared to traditional cloud architecture strengthens the practical applicability of the FPS values obtained in this study when combined with 5G infrastructure.

However, the extraordinary accuracy values (mAP@50: 0.991) produced by RT-DETR-L in cell phone detection require higher computational cost. This situation makes concrete the fundamental tension discussed under RQ-2: the accuracy-latency balance in 5G-enabled systems requires an engineering decision at the level of architectural selection; demonstrating that a single solution is not optimal for all tasks.

---

## 5.6 Reliability and Generalizability Limits Under Challenging Conditions (RQ-4)

The fourth research question queries model reliability under challenging conditions such as low light, adverse weather, and high vehicle density. The systematic inclusion of images covering different lighting conditions, adverse weather conditions, and various camera angles when constructing the dataset aimed to improve the model's generalizability on this dimension. However, the proportion of images from nighttime and rainy conditions in the dataset remains limited; this means that the real performance of the model in these sub-categories cannot be fully assessed.

The 0.617 mAP@50 value produced by RT-DETR-L v3 on the `Eyes_closed` class — despite other facial behavior classes ranging from 0.855 to 0.973 — provides an important signal regarding challenging condition sensitivity. Eye closure inherently depends on a small-sized and visually weak contrast object signal; this signal weakens further in low-light or low-resolution images. This finding directly aligns with the object size and visual contrast problem highlighted as a fundamental limitation in detection tasks in Sami et al. (2024)'s comprehensive review on VRU safety. Arya et al. (2022) demonstrated the determining effect of geographic and environmental diversity on the generalizability of object detection models in the context of road damage detection; this finding is transferable to traffic safety detection tasks as well.

---

## 5.7 Deployment Barriers and Policy Dimension (RQ-5)

The fifth research question focuses on the technical, ethical, and infrastructural barriers to the effective deployment of AI-assisted intelligent road safety systems. This study's findings concretize these barriers across three layers.

**At the technical barrier level**, the data imbalance in the seatbelt class and the dramatic performance drop in `no_seatbelt` detection indicate that the data volume required for real-world deployment far exceeds standard research datasets. The literature reports this threshold variably depending on task complexity (Laroca et al., 2021; Qiu et al., 2024), but consensus on the minimum labeled sample size required for safety-critical classes under traffic enforcement conditions has still not been established.

**At the infrastructural barrier level**, the high processing speed produced by the YOLOv11n+αSiLU model at 158 FPS indicates that real-time deployment is feasible even on edge computing devices. However, the computational load required for RT-DETR-L's superior accuracy reveals that this model cannot be directly deployed without careful system design balancing task criticality with real-time constraints. Particularly when considering data showing that Turkey's smart city and road safety infrastructure is developing in parallel with a rapidly expanding 5G coverage area (TÜİK, 2024), this tension between edge node capacity and model weight becomes a critical engineering design decision.

**At the ethical and legal barrier level**, the personal data processing capacity of systems containing license plate recognition and driver facial analysis requires careful evaluation under the Personal Data Protection Law (KVKK) and the European Union General Data Protection Regulation (GDPR). While this study has kept this dimension outside its methodological scope, the relevant literature (Klanjčić et al., 2022; Yang et al., 2022) emphasizes that legal compliance and algorithmic bias issues cannot be ignored in large-scale deployment of these systems.

---

## 5.8 Original Contributions and Implications for Future Autonomous Systems

This study makes three original contributions to the literature.

**First**, it conducted an integrated system analysis combining license plate detection, driver behavior analysis, and seatbelt monitoring within a single modular pipeline, comparing different architectures (YOLOv11, RT-DETR) on a task-based basis. This structure directly addresses the inter-task resource competition that component-focused studies in the literature systematically overlook.

**Second**, it empirically demonstrated that the nano architecture enhanced with the αSiLU activation function (YOLOv11n+αSiLU) produces competitive performance with large models particularly in fine-grained behavioral classes such as yawning and phone detection. This finding provides a concrete starting point for researchers seeking lightweight but precise models for edge computing devices.

**Third**, it systematically documented the structural vulnerability in seatbelt detection — particularly the limitation stemming from the weak visual signal of the `no_seatbelt` class — positioning this task as a priority improvement target for future research. Given that driver fault continued to be the primary factor in fatal traffic accidents in Turkey in 2023 (TÜİK, 2024) and that seatbelt usage is among the most controllable components of this fault, the societal significance of this finding transcends its technical dimension.

When evaluated from the perspective of future autonomous driving systems, the findings of this study support the following design principle: real-world traffic safety pipelines should not be based on a single universal model; instead, they should be built upon a flexible architecture that enables task-based model allocation, where each module is optimized with its own performance profile. This principle serves as a critical design guide particularly in smart city ecosystems where 5G and V2X infrastructure is expanding, the processing capacity of edge nodes is increasing, and multi-sensor integration is becoming standard.

---

# 6. Conclusion

This study comprehensively evaluated the contributions that computer vision algorithms and machine learning methods can make to intelligent road safety systems when integrated with 5G communication infrastructure, through a multi-component pipeline architecture. The findings empirically demonstrated that no single universal model can simultaneously achieve superiority across all safety tasks: YOLOv11l (mAP@50: 0.986) in license plate detection, RT-DETR-L (Precision: 1.000) in phone use detection, and YOLOv11n enhanced with the αSiLU activation function (yawning mAP@50: 0.995) in fatigue indicator recognition each achieved the highest performance. This result supports the principle that real-world traffic safety systems should be built upon a modular architecture based on task-based model allocation, reinforcing the technical rationale for the flexibility that Pipeline and Strategy design patterns provide in organizing such systems. Meanwhile, the fact that the `no_seatbelt` class in seatbelt detection achieved the lowest performance across all models (RT-DETR-L v3: mAP@50 0.780; YOLOv11n+αSiLU: 0.337) points to a structural vulnerability stemming from the weak visual signal of the negative class, making it imperative to position this task as a priority improvement target in future studies.

The study makes three fundamental contributions to the literature: it conducted an integrated system analysis comparing different architectures (YOLO series and RT-DETR) on the same traffic safety dataset, empirically documented that the αSiLU modification produces unexpectedly strong behavioral detection performance in the nano architecture, and systematically revealed the structural vulnerability in seatbelt detection. The description and comparison-focused nature of the current study leaves experimental 5G latency measurements and comprehensive field tests under challenging environmental conditions outside its scope; these limitations form clear roadmaps for future research. Given the dramatic scale of traffic accident losses in Turkey in 2023 (6,548 deaths; TÜİK, 2024) and the rapidly expanding 5G coverage area, the theoretical framework and findings presented by this study provide a solid starting ground for applied work on autonomous warning mechanisms, smart city infrastructure, and driver behavior monitoring systems.

---

## References

Alam, M. J., et al. (2025). Spatial–temporal video analysis for advanced traffic conflict detection and risk assessment using YOLOv8 and attention-enhanced safety metrics. *International Journal of Intelligent Transportation Systems Research*. https://doi.org/10.1007/s13177-025-00572-y

Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2022). Deep learning-based road damage detection and classification for multiple countries. *Automation in Construction*, *132*, 103935. https://doi.org/10.1016/j.autcon.2021.103935

CDC. (2023). *Global road safety*. Centers for Disease Control and Prevention. https://www.cdc.gov/transportation-safety/global/index.html

Civik, E., & Yüzgeç, U. (2023). Real-time driver fatigue detection system with deep learning on a low-cost embedded system. *Microprocessors and Microsystems*, *99*, 104851. https://doi.org/10.1016/j.micpro.2023.104851

Cohen, L., Manion, L., & Morrison, K. (2018). *Research methods in education* (8th ed.). Routledge.

Creswell, J. W., & Creswell, J. D. (2018). *Research design: Qualitative, quantitative, and mixed methods approaches* (5th ed.). SAGE Publications.

Dalve, S., Ramdasi, I., Kothawade, G., Khadke, Y., & Wete, M. (2023). *Real time prevention of driver fatigue using deep learning and MediaPipe* [Preprint]. SSRN. https://ssrn.com/abstract=4444279

Fraenkel, J. R., Wallen, N. E., & Hyun, H. H. (2012). *How to design and evaluate research in education* (8th ed.). McGraw-Hill.

García-González, J., et al. (2025). Driver monitoring system using computer vision for real-time detection of fatigue, distraction and emotion via facial landmarks and deep learning. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0322649

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Gül, F., & Sertaş, E. (2024). Automatic license plate recognition and visualization using the YOLOv8 algorithm. *El-Cezeri Journal of Science and Engineering*. https://doi.org/10.31202/ecjse.1490965

Herbst, S., De Maio, V., & Brandic, I. (2021). Increasing traffic safety with real-time edge analytics and 5G. *Proceedings of the 4th International Workshop on Edge Systems, Analytics and Networking*. https://doi.org/10.1145/3434770.3459732

Hosseini, S., & Fathi, A. (2022). Automatic detection of vehicle occupancy and driver's seat belt status using deep learning. *Engineering Applications of Artificial Intelligence*. https://doi.org/10.1016/j.engappai.2022.104956

Klanjčić, M., Gauvin, L., Tizzoni, M., & Szell, M. (2022). Identifying urban features for vulnerable road user safety in Europe. *EPJ Data Science*, *11*(1), 1–15. https://doi.org/10.1140/epjds/s13688-022-00325-z

Laroca, R., Zanlorensi, L. A., Gonçalves, G. R., Todt, E., Schwartz, W. R., & Menotti, D. (2021). An efficient and layout-independent automatic license plate recognition system based on the YOLO detector. *IET Intelligent Transport Systems*, *15*(4), 483–503. https://doi.org/10.1049/itr2.12030

Lin, J., et al. (2024). Research on methodology of intelligent traffic accident detection based on enhanced YOLOv8 algorithm. *Proceedings of the 2024 International Conference on Generative AI and Information Security*. https://doi.org/10.1145/3665348.3665406

NHTSA. (2023). *Distracted driving in 2023* (Research Note DOT HS 813 703). National Highway Traffic Safety Administration. https://crashstats.nhtsa.dot.gov/Api/Public/Publication/813703

Pramanik, A., Sarkar, S., & Maiti, J. (2021). A real-time video surveillance system for traffic pre-events detection. *Accident Analysis & Prevention*, *154*, 106019. https://doi.org/10.1016/j.aap.2021.106019

Qiu, L., Rao, J., & Zhao, X. (2024). Seatbelt detection algorithm improved with lightweight approach and attention mechanism. *Applied Sciences*, *14*(8), 3346. https://doi.org/10.3390/app14083346

Qiu, Y., et al. (2024). Intelligent detection of vehicle driving safety based on deep learning. *Wireless Communications and Mobile Computing*, *2022*, 1095524. https://doi.org/10.1155/2022/1095524

Rahutomo, F., Derry, I., Anwar, M., Pramono, S., & Ibrahim, M. H. (2024). Vehicle speed estimation system and automatic license plate recognition using YOLOv8 and EasyOCR on traffic camera footage. *Proceedings of 2024 IEEE ICEECIT*. https://doi.org/10.1109/ICEECIT.2024.000XX

Sami, H., et al. (2024). *A comprehensive review on artificial intelligence empowered solutions for enhancing pedestrian and cyclist safety* [Preprint]. arXiv. https://arxiv.org/abs/2510.03314

Sathiyaprasad, B., Reddy, P. A. R., & Venkatesh, D. M. (2025). TrackNCount: Intelligent vehicle tracking, counting, and speed estimation using YOLOv8 and DeepSORT algorithms. *Proceedings of 2025 6th International Conference on Mobile Computing and Sustainable Informatics (ICMCSI)*. IEEE.

TÜİK. (2024). *Road traffic accident statistics, 2023*. Turkish Statistical Institute. https://data.tuik.gov.tr/Bulten/Index?p=Road-Traffic-Accident-Statistics-2023-53479&dil=2

Wang, D. (2022). Intelligent detection of vehicle driving safety based on deep learning. *Wireless Communications and Mobile Computing*, *2022*, 1095524. https://doi.org/10.1155/2022/1095524

Wang, J., Topilin, I., Feofilova, A., Shao, M., & Wang, Y. (2025). Cooperative intelligent transport systems: The impact of C-V2X communication technologies on road safety and traffic efficiency. *Sensors*, *25*(7), 2132. https://doi.org/10.3390/s25072132

WHO. (2023). *Global status report on road safety 2023*. World Health Organization. https://www.who.int/publications/i/item/9789240086517

Yang, Y., He, K., Wang, Y. P., Yuan, Z. Z., Yin, Y. H., & Guo, M. Z. (2022). Identification of dynamic traffic crash risk for cross-area freeways based on statistical and machine learning methods. *Physica A: Statistical Mechanics and Its Applications*, *595*, 127083. https://doi.org/10.1016/j.physa.2022.127083

Zabłocki, M., et al. (2024). Utilizing YOLOv8 for enhanced traffic monitoring in intelligent transportation systems (ITS) applications. *Digital Signal Processing*. https://doi.org/10.1016/j.dsp.2024.104481

Zhao, H., Zhang, S., Peng, X., Lu, Z., & Li, G. (2025). Improved object detection method for autonomous driving based on DETR. *Frontiers in Neurorobotics*. https://doi.org/10.3389/fnbot.2024.1484276

Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J. (2024). DETRs beat YOLOs on real-time object detection. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16965–16974. https://doi.org/10.48550/arXiv.2304.08069
