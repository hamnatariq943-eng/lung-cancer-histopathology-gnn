# lung-cancer-histopathology-gnn
 
A GRAPH NEURAL NETWORK FRAMEWORK FOR  REPRESENTATION LEARNING IN WHOLESLIDE  
HISTOPATHOLOGY IMAGES  
   
SUBMITTED BY  
Rubab 
2022-ag-7628 
ADVISED BY  
Dr.  Muhammad Zeeshan Asaf  
  
A FINAL YEAR PROJECT SUBMITTED FOR THE COURSE  
  
BINFO-606 FINAL YEAR PROJECT-I 6(0-6)  
  
DEPARTMENT OF COMPUTER SCIENCE  
FACULTY OF SCIENCES  
UNIVERSITY OF AGRICULTURE,  
FAISALABAD  
2026  
DECLARATION  
I hereby declare that the Final Year Project entitled “A Graph Neural Network Framework for Representation Learning in Whole-Slide Histopathology Images” is the result of my own research work carried out under the guidance and supervision of my supervisor Dr. Muhmmad Zeeshan Asaf. To the best of my knowledge and belief, this project contains no material previously published or written by another person except where proper acknowledgment and references have been provided.  
I further declare that all data, analyses, results, figures, and interpretations presented in this report are original and have been developed solely for the purpose of this study. This work has not been submitted, either in whole or in part, to any other university or institution for the award of any degree, diploma, or academic qualification.  
I take full responsibility for the authenticity and accuracy of the information presented in this project. I understand that any form of plagiarism, fabrication, falsification of data, or misrepresentation of facts is a serious academic offense. If any part of this declaration is found to be incorrect or misleading at any stage, I shall be liable to disciplinary action in accordance with the rules, regulations, and policies of the University of Agriculture Faisalabad (UAF).  
I also acknowledge that the university reserves the right to verify the originality of this work through any appropriate means and to take necessary action in case of violation of academic integrity standards.  
___________________________ 
	 	 Student Name: Rubab  
  	  
 	 
Project Evaluation Matrix (Total: 80 Marks)  
  
Category  	Obtained Marks  	Max Marks  
Project Documentation  	  	15  
Originality & Code Quality  	  	10  
Features Developed  	  	20  
User Interface (UI/UX)  	  	10  
Portability & Deployment  	  	10  
Technical Knowledge (Viva)  	  	15  
  
   	  
 	 
CERTIFICATE  
  
To,  
The Controller of Examinations,  
University of Agriculture,  
Faisalabad.  
The committee certify that Rubab(2022-ag-7628) has successfully completed his/her project in partial fulfillment of the requirements for the course BINFO-606 Final Year Project 6(0-6) Computer Science under our guidance and supervision.  
  
  
Advisor:   	  	  	  	  	  	________________________________  
  
  
  
Evaluation Committee Member:   	  	  	________________________________  
  
  
  
Chairman, Department of Computer Science:  	________________________________  
  
  	  
 	 
ACKNOWLEDGEMENT  
In the name of Allah, the Most Gracious, the Most Merciful.  
First and foremost, I would like to express my deepest gratitude to Almighty Allah for granting me strength, knowledge, patience, and guidance throughout the completion of this research work. His countless blessings, mercy, and wisdom enabled me to overcome challenges and continue this journey with determination and hope.  
As Allah says in the Holy Quran:  
"And my success is not but through Allah. Upon Him I have relied, and to Him I return."  
(Surah Hud 11:88)  
I would like to express my sincere gratitude to my respected supervisor, Dr. Muhammad Zeeshan Asaf, for his valuable guidance, continuous encouragement, constructive suggestions, and unwavering support throughout the course of this study. His expertise and insightful feedback played a vital role in shaping this research and bringing it to completion.  
My sincere appreciation is extended to all individuals who directly or indirectly assisted me during this research. Their cooperation, valuable discussions, and constructive feedback helped improve the quality of this work.  
I am especially grateful to my fellow students for their encouragement, motivation, and helpful suggestions throughout this journey. Their support made the research process more productive and enjoyable.  
Finally, I owe my deepest gratitude to my parents and family for their unconditional love, patience, prayers, and constant encouragement. Their sacrifices, trust, and support have always been a source of strength and inspiration for me.  
I dedicate this achievement to all those who contributed, in one way or another, to the successful completion of this work.  
  	  
 	 
ABSTRACT  
Lung cancer is one of the leading causes of cancer-related mortality worldwide, with Lung Adenocarcinoma (LUAD) and Lung Squamous Cell Carcinoma (LUSC) being the two most clinically significant subtypes. Accurate differentiation between these subtypes is essential for determining appropriate treatment strategies and improving patient survival outcomes. Manual histopathological examination of Whole Slide Images (WSIs) is time-consuming and highly dependent on pathologist expertise, necessitating automated computational solutions.  
This study proposes a graph-based deep learning framework for automated LUAD and LUSC classification using 42 WSIs obtained from The Cancer Genome Atlas (TCGA) repository. Coordinate-based patch extraction combined with tissue masking was applied to select informative tissue-rich regions, yielding approximately 2,770 patches. LAB-based gentle stain normalization was performed to reduce inter-slide color variability while preserving cellular morphology and tissue structures. Each WSI was represented as a graph where extracted patches serve as nodes and spatial adjacency relationships define edges, effectively capturing the structural organization of histopathological tissue. Graph Neural Networks (GNNs) combined with self-supervised representation learning were employed to capture meaningful tissue-level patterns without heavy dependence on manually labeled data.  
Experimental evaluation on a held-out test set achieved perfect classification performance with Accuracy, Precision, Recall, F1-Score, and AUC-ROC all equal to 1.0. These results demonstrate that combining graph-based learning, stain normalization, and informative patch selection significantly improves automated lung cancer subtype classification in computational pathology, offering a promising direction for clinical decision support systems.  
Keywords: Lung cancer, LUAD, LUSC, whole slide images, graph neural network, stain normalization, self-supervised learning, computational pathology.  
 	 
  	  
CONTENTS  
Chapter 1 – INTRODUCTION	1 
1.1   Background	1 
1.2   Description	1 
1.3   Problem Statement	2 
1.4   Scope	2 
1.4.1 Domain Scope	2 
1.4.2 Data Scope	3 
1.4.3 Methodological Scope	3 
1.4.4 Task Scope	3 
1.5  Objectives	3 
Chapter 2 – MATERIALS & METHODS	4 
2.1   Histopathology image analysis	4 
2.2   Data Description	4 
2.2.1 Cancer Genome Atlas	4 
2.2.2 Dataset Overview	5 
2.2.3 LUAD and LUSC	6 
2.3   Informative patch selection strategy and stain normalization	7 
2.3.1 Loading images of the whole slide	8 
2.3.2 Open Slide was used to read the slides at SVS and Thumbnail generation	8 
2.3.3 Generate Tissue mask	8 
2.3.4 Padding	8 
2.3.5 Patch Extraction	9 
2.3.6 Tissue patch filtering	9 
2.3.7 Balanced patch selection	9 
2.3.8 LAB-Based Gentle Stain Normalization and Reconstruction	9 
2.3.9 Tissue Mask Generation on Reconstructed Normalized Tissue Regions	9 
2.3.10 Before and After Comparison	9 
2.4.   Graph construction and Representation Learning	10 
2.4.1 CNN feature extraction	10 
2.4.2 Graph construction KNN connectivity	10 
2.4.3 Self-supervised graph representation learning	11 
2.4.4 GCN classification (fine tuning)	11 
2.4.5 Analysis of visualization and Representation	11 
Chapter 3 - RESULTS & DISCUSSION	13 
3.1   Informative Patch Selection	13 
3.1.1 Whole Slide Thumbnail Generation	13 
3.1.2 Tissue mask generation	14 
3.1.3 Patch Extraction Results	15 
3.1.4 Informative Patch Filtering	16 
3.1.5 Balanced Patch Selection	16 
3.2   Stain Normalization and color Correction	17 
3.2.1 LAB-Based Gentle Stain Normalization	17 
3.2.2 Reconstruction of Normalized LUAD and LUSC patches	19 
3.3  Graph Construction and Representation Learning	22 
3.3.1 ResNet5o Feature Extraction	22 
3.3.2 Graph Construction results	22 
3.3.3 Self supervised Graph Auto encoder pretraining	24 
3.3.3 GCN Fine Tuning Results	24 
3.4.6 Confusion Matrix Analysis	26 
3.4.7 PCA Graph Embedding Visualization	27 
3.4.8 t-SNE Graph Embedding Visualization	28 
3.4.9 Graph Representation Quality Analysis	28 
Chapter 4 - Conclusion	29 
REFERENCES	31 

 
 
  
  
  
  
  	  
 	 
 
LIST OF FIGURES  
Figure 2. 1: Whole slide image with regions at different magnifications ............................................................... 5 
Figure 2. 2: Workflow of whole slide image processing showing and multi-level patch representation, tissue    
selection, stain normalization, patch generation. .................................................................................................... 6 
Figure 2. 3: Overview of the proposed preprocessing framework showing tissue mask generation, informative patch extraction, balanced patch selection, LAB-based gentle stain normalization, coordinate-based  
reconstruction, and visual validation for LUAD and LUSC histopathology images .............................................. 7 
Figure 2. 4: Patch extraction and tissue patch filtering from a whole slide image. ................................................ 8  
Figure 2. 5: Graph construction and self-supervised graph representation learning framework for LUAD and  
LUSC classification. ............................................................................................................................................. 10 
Figure 3. 1: Generated thumbnails of LUAD whole slide histopathology images ............................................... 13  
Figure 3. 2: Generated thumbnails of LUAD whole slide histopathology images ............................................... 14  
Figure 3. 3: LUAD Thumbnail Generation........................................................................................................... 14 
Figure 3. 4: LUSC Thumbnail Generation ........................................................................................................... 15  
Figure 3. 5: Representative LUAD informative patch selected after tissue filtering ............................................ 15  
Figure 3. 6: Representative LUSC informative patch selected after tissue filtering ............................................. 16  
Figure 3. 7: Representative LUAD tissue patches before and after LAB-based gentle stain normalization ........ 18  Figure 3. 8: Representative LUSC tissue patches before and after LAB-based gentle stain normalization ......... 19  
Figure 3. 9: Comparison between the original LUAD whole slide images region and the reconstructed LAB- 
normalized tissue patch grid. ................................................................................................................................ 20  
Figure 3. 10: Comparison between the original LUSC whole slide image thumbnail and the coordinate-based  
reconstructed LUSC tissue regions. ...................................................................................................................... 21  
Figure 3. 11: Comparison of the original LUAD tissue region and initial tissue mask with the reconstructed  
normalized tissue patches and tissue mask after normalization. ........................................................................... 21 
Figure 3. 12: Comparison of the original LUSC whole slide image thumbnail and initial tissue mask with the coordinate-based reconstructed tissue regions and tissue mask after reconstruction. ........................................... 22 
Figure 3. 13: LUAD patch graph visualizations showing 24 nodes and 120 edges .............................................. 23  
Figure 3. 14: LUSC patch graph visualizations showing 100 nodes and 500 edges. ............................................ 24  
Figure 3. 15: Self-supervised graph auto encoder pretraining loss curve. ............................................................ 24  Figure 3. 16: GCN fine-tuning loss curve for LUAD and LUSC classification ................................................... 25  
Figure 3. 17: Training loss curve of the proposed GCN classifier........................................................................ 25 
Figure 3. 18: Training and validation accuracy curve of the proposed graph learning framework ...................... 26  
Figure 3. 19: Confusion matrix of the proposed GCN model for LUAD and LUSC    classification .................. 27  Figure 3. 20: PCA visualization of graph embeddings for LUAD and LUSC sample ......................................... 30 
Figure 3. 21: t-SNE visualization of graph embedding’s for LUAD and LUSC samples .................................... 30  
  
  	  
 	 
LIST OF TABLE  
Table 3. 1: Evaluation ........................................................................................................................................... 26  
  
  
  
  
 
Chapter 1 – INTRODUCTION  
1.1  	Background  
Lung cancer is considered one of the most dangerous and life-threatening diseases worldwide and remains a major cause of cancer-related deaths. Early and accurate diagnosis of lung cancer is extremely important because it directly affects treatment planning and patient survival. Histopathology examination of tissue slides is commonly used by pathologists for identifying different lung cancer subtypes such as Lung Adenocarcinoma (LUAD) and Lung Squamous Cell Carcinoma (LUSC). However, manual analysis of whole slide images (WSIs) is a difficult and time-consuming task because these images are extremely large and contain complex tissue structures. In addition, diagnosis based only on human observation may lead to inter-observer variability and inconsistencies in interpretation.  
Recent advancements in artificial intelligence and deep learning have significantly improved digital pathology and automated histopathology image analysis. Traditional patch-based deep learning methods often process tissue patches independently and ignore important spatial relationships between neighboring tissue regions. Moreover, histopathology slides usually suffer from stain variability caused by different laboratory preparation and scanning conditions, which affects the performance and generalization capability of deep learning models. Therefore, there is a need for an efficient framework that can select informative tissue regions, reduce stain variation, and learn meaningful tissue representations while preserving spatial tissue relationships.  
This research proposes a Graph Neural Network (GNN) based representation learning framework for LUAD and LUSC histopathology image analysis using TCGA whole slide images. The framework integrates tissue mask generation, informative patch extraction, stain normalization, graph construction, and self-supervised graph learning techniques. ResNet50 is used for deep feature extraction, while Graph Convolutional Networks (GCN) and graph autoencoders are utilized for graph representation learning and classification. The proposed approach aims to improve tissue representation learning, reduce dependence on large annotated datasets, and enhance classification performance for lung cancer subtype analysis.  
1.2  	Description  
This research focuses on developing a Graph Neural Network (GNN) based representation learning framework for histopathology whole slide image analysis of lung cancer. The proposed study aims to improve automated classification of Lung Adenocarcinoma (LUAD) and Lung Squamous Cell Carcinoma (LUSC) using deep learning and graph-based representation learning techniques. Histopathology whole slide images are extremely large and contain complex tissue structures, making manual diagnosis difficult and time-consuming for pathologists. Traditional deep learning approaches often process image patches independently and fail to preserve important spatial relationships between tissue regions. Furthermore, stain variations across histopathology slides can negatively affect classification performance and model generalization.  
To address these challenges, this research proposes a multi-stage framework consisting of tissue mask generation, informative patch extraction, stain normalization, graph construction, selfsupervised graph learning, and graph classification. Whole slide histopathology images are obtained from The Cancer Genome Atlas (TCGA) dataset, specifically TCGA-LUAD and TCGA-LUSC datasets in SVS format. Open Slide and OpenCV are used for reading and preprocessing whole slide images, while informative tissue-rich patches are extracted by applying thresholding and morphological operations. The extracted patches are resized to 256×256 pixels and filtered using tissue ratio and variance thresholds to remove noninformative background regions.  
To reduce stain inconsistencies among slides, LAB-based gentle stain normalization techniques are applied while preserving important tissue morphology. The normalized tissue patches are reconstructed using their original spatial coordinates to maintain tissue arrangement consistency. ResNet50 is used as a pretrained feature extractor for generating deep feature embeddings from tissue patches. Graph structures are then constructed where tissue patches act as graph nodes and spatial similarity relationships act as graph edges using K-nearest neighbor (KNN) connectivity.  
The proposed framework further utilizes self-supervised graph autoencoder pretraining and Graph Convolutional Networks (GCN) for graph representation learning and classification.  
The performance of the framework is evaluated using accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix analysis, PCA visualization, and t-SNE visualization. The expected outcome of this research is to develop an efficient and reliable computational pathology framework capable of improving lung cancer subtype classification while reducing dependence on large manually annotated datasets.  
1.3  	Problem Statement  
The extensive diagnostic information in whole slide histopathology images is challenging to analyze automatically because they are gigapixels, heterogeneous tissues, variable stains, and limited annotations. Current patch based deep learning methods tend to overlook significant spatial  connections between tissue regions, and many graph based methods lack adequate consideration of patch choice, stain variation, or label paucity. This has led to the necessity of creating an effective and powerful graph neural network architecture, capable of learning meaningful whole slide representations by retaining diagnostically relevant regions, reducing technical variability, and using the data more efficiently with little need to rely on large scale manual labeling.  
1.4  	Scope  
Every research project requires clear boundaries to maintain focus and feasibility. The scope of this thesis is defined as follows:  
1.4.1 Domain Scope  
The study is limited to computational pathology and focuses specifically on digital histopathology images. It does not cover radiology, genomics, or non image clinical records except where such information may be mentioned for background discussion  
1.4.2 Data Scope  
The experiments will use publicly available whole slide image datasets, primarily The Cancer Genome Atlas (TCGA), with emphasis on relevant cancer types such as lung adenocarcinoma (LUAD) and lung squamous cell carcinoma (LUSC), depending on data availability and experimental suitability  
1.4.3 Methodological Scope  
The proposed research mainly focus on  
•	Patch extraction and optimized patch selection.  
•	Stain normalization.  
•	Graph construction.  
•	Graph Neural Network modeling.  
•	Self-supervised or unsupervised representation learning.  
•	Experimental evaluation using standard metrics.  
•	The study does not aim to develop a complete hospital deployment system or commercial software product.  
1.4.4 Task Scope  
The primary focus of this research is graph-based representation learning for histopathology image analysis. Downstream evaluation tasks may include classification, clustering, retrieval, and related predictive analysis depending on dataset labels, implementation feasibility, and experimental suitability. The study particularly emphasizes LUAD and LUSC subtype classification using graph learning technique  
1.5  	Objectives  
Goal: The goal of this research is to develop an efficient Graph Neural Network (GNN) based representation learning framework for histopathology whole slide image analysis to improve automated classification of Lung Adenocarcinoma (LUAD) and Lung Squamous Cell Carcinoma (LUSC)  
Objectives: The main objective of this study are:  
•	To improve the selection of image patches by keeping the tissue regions that are important for diagnosis and reducing patches that do not add useful information. Generate tissue mask and extract useful informative tissue patches from whole slide histopathology images while removing unnecessary background region.  
  
•	To apply stain normalization and color correction so that differences in slide colors are reduced, allowing histopathology images to be compared and analyzed more reliably and preserving important tissue morphology and nuclei structure.  
•	To use self-supervised or unsupervised learning methods to learn meaningful features from histopathology images without depending heavily on expert labeled data  
  	  
 
Chapter 2 – MATERIALS & METHODS  
The methodology chapter describes the process of doing the research step by step. It gives a vivid explanation of the methods, instruments and the processes through which the research objectives are attained. The methodology described in this paper aims at creating a graph neural network (GNN)-based model of learning meaningful representations of whole slide histopathology images (WSIs).  
2.1  	Histopathology image analysis  
Histopathology is a microscopic study of tissue samples in order to diagnose diseases like cancer. Conventionally it is done manually by professional pathologists that may be timeconsuming and subjective. As digital pathology evolves, such tissue slides are now scan able to create high resolution images called WSIs, and automated analysis methods of artificial intelligence can be applied to them.  
Recent studies indicates that deep learning and machine learning prototypes have enhanced the precision and effectiveness of histopathological image examination, particularly in the areas of tumor identification, classification and prediction of the prognosis of tumors, among others.  
Nevertheless, there are   
•	Very large image sizes (WSIs are extremely high resolution)   
•	Presence of irrelevant background regions   
•	Variability in staining and color   
•	Limited availability of labeled data set  
2.2  	Data Description  
The data utilized in this analysis is found in The Cancer Genome Atlas, a publically accessible and well-known source of cancer research. TCGA is an extensive source of clinical, genomic and histopathological data on various types of cancer, which is why it is very useful in developing and evaluating deep learning models in digital pathology.  
The dataset is in the form of whole slide images (WSIs)which are digital scans of histopathology slides in high resolution . They are detailed images of tissue structures in a microscopic perspective and are used in the diagnosis of diseases, especially in cancer detection and classification. WSIs are very large and ,therefore, cannot be analyzed directly and are usually broken down into smaller image patches to be analyzed.  
2.2.1 Cancer Genome Atlas  
The data set utilized in this research was taken as The Cancer Genome Atlas (TCGA),which is a well-established and publicly available research initiative that provide comprehensive data on the topic of cancer. TCGA was created to assist scientists by providing high quality genomic and clinical data obtained on a large sample of cancer patients. TCGA was the main data source of histopathology images of lung cancer in this study. It contains whole slide images(WSIs) digitized and accompanied by molecular and clinical data making it a reliable and widely accepted resource to study computational pathology. This is because of the consistency and reproducibility of research results due to the availability of standardized and well annotated data.  
In particular, the subtypes of lung cancer were chosen based on the TCGA repository to justify the aims of this study. The datasets presents in depth visualization of tissue samples and can be analyzed through a sophisticated image processing and feature detection. The applicability of using the TCGA data increases the validity of the proposed methodology as it has been widely utilized in the past medical imaging and machine learning studies. Moreover, in the current context, TCGA is an open access platform, which provides transparency and enables other researcher to repeat or expand the results of the current research. This choice of the dataset is in line with the objective of coming up with strong and generalizable models to detect and classify cancers.  
2.2.2 Dataset Overview  
TCGA data is chosen in this study due to its variety and size since it contains images of histopathology of a variety of cancer types, including lung, breast, and colorectal cancer. This variability facilitates the acquisition of strong and generalized feature representations, which is crucial to the suggested graph neural network framework.  
There are however various challenges in TCGA images such as variability in staining, the existence of non-informative background areas as well as the lack of detailed annotation. To overcome these problems. preprocessing methods, including patch extraction, informative patch selection, and stain normalization are included in the proposed methodology.  
Additionally, given that labeled data in TCGA in scarce, at the patch level, this research uses self-supervised and unsupervised learning methods to learn meaningful representations using the data without extensively depending on expert annotations. This renders TCGA especially appropriate to the representation learning tasks in histopathology.  
All in all, using TCGA guarantees that the suggested framework will be tested on a large scale real-world dataset, which will boost the reliability and applicability of the research results  
   
Figure 2. 1: Whole slide image with regions at different magnifications  
Figure 2.1 illustrates a multi-scale representation of a whole-slide histopathology image (WSI). The complete tissue slide is first viewed at a low magnification level and then progressively examined at higher magnifications to reveal detailed cellular and tissue structures. The image demonstrates a hierarchical learning approach, where information is extracted from different magnification levels to capture both global tissue patterns and finegrained pathological features. This helps improve the accuracy of computer-aided cancer diagnosis and classification from histopathology images.  
   
Figure 2. 2: Workflow of whole slide image processing showing and multi-level patch representation, tissue   selection, stain normalization, patch generation.  
Figure 2.2 presents the preprocessing workflow for whole-slide histopathology images. First, tissue regions are identified and non-informative areas are removed through ROI extraction. The selected tissue is then stain-normalized, divided into image patches, and processed at multiple magnification levels to capture both global tissue architecture and detailed cellular features for subsequent analysis.  
2.2.3 LUAD and LUSC  
The Lung Cancer Histopathology images used in this study were acquired from the TCGA; two primary lung adenocarcinoma (LUAD) and lung squamous cell carcinoma (LUSC) were analyzed. In medical image analysis research, these are the two most prevalent forms of Nsclc (Non-Small Cell Lung Cancer) and are widely used.  
The TCGA LUAD are whole slide images (WSIs) that have been digitized from lung adenocarcinoma patients. This type is typically found in the periphery of lungs and is seen in smokers and non smokers.The images within this datasets include high resolution views of tissue structures such as tumor areas, cellular arrangement and morphological differences essential for diagnosis.  
The TCGA LUSC datasets also has whole slide images of patients with lung squamous cell carcinoma. Typically found in the middle of the lungs, this subtypes is related to smoking heavily. High resolution histopathological images with distinct features like keratinization and abnormalities in cell growth patterns.  
The images are stored in both datasets in the SVS file which preserve the high-resolution quality required for computational analysis. These whole slide images are extremely large in size and contain rich visual information at multiple magnification levels , making them suitable for patch based processing and deep learning applications.  
For the purpose of this study , images from both LUAD and LUSC datasets were used to ensure a balanced and comparative analysis. The inclusion of these two subtypes supports the development of robust models capable of distinguishing between different lung cancer patterns. Moreover, the use of TCGA datasets ensures data reliability, as they are wellcurated, standardized, and widely accepted in the research community.  
2.3  	Informative patch selection strategy and stain normalization  
   
Figure 2. 3: Overview of the proposed preprocessing framework showing tissue mask generation, informative patch extraction, balanced patch selection, LAB-based gentle stain normalization, coordinate-based reconstruction, and visual validation for LUAD and LUSC histopathology images  
Figure 2.3 illustrates the complete preprocessing pipeline for LUAD and LUSC whole-slide images. The process begins with WSI acquisition, slide reading, tissue mask generation, and image padding, followed by patch extraction and filtering to retain only informative tissue regions. Selected patches are stain-normalized, reconstructed, and visually compared with the original slides to ensure tissue preservation and staining consistency before further analysis.  
   
Figure 2. 4: Patch extraction and tissue patch filtering from a whole slide image.  
Figure 2.4 shows the patch extraction process from a whole-slide histopathology image. The WSI is divided into small image tiles, and patches containing less than 30% tissue area are removed. Only tissue-rich and diagnostically relevant patches are retained for further analysis and model training.  
2.3.1 Loading images of the whole slide  
Whole slide histopathology images of TCGA-LUAD and TCGA-LUSC were obtained  from the TCGA/GDC portal in the SVS format. The image slides are high resolution digital images of lung cancer tissue specimens. The data set was chosen for graph-based histopathology analysis and representation learning.  
2.3.2 Open Slide was used to read the slides at SVS and Thumbnail generation  
SVS slides were loaded in python jupyter notebook with the Open Slide library to be processed. At this stage , the original dimensions of the slides such as width, height, channels of the image were extracted and recorded. This data was employed in the reconstruction of patches and images. A 256×256 thumbnail is made for each slide because whole slide images are too big to be processed directly at full resolution during early exploration. These Thumbnail give a quick visual summary of the slide’s content without having to load the entire giga pixel image into memory. This step is helpful for looking at the slides, checking their quality, and getting a general idea of how the tissues are spread out before doing more detailed work.  
2.3.3 Generate Tissue mask  
The white background of the image was replaced with tissue regions using thresholding and morphological operations to create a tissue mask. Small noise regions were removed and the quality of the mask was enhanced by morphological opening, closing. This step resulted in the selection of the only meaningful tissue areas for patch extraction.  
2.3.4 Padding  
The dimensions of the whole slide images were not exactly divisible by the patch size which is why there is black padding at the bottom and right hand edge of the image. The slide was padded so the width and height were all even multiples of 256.The original dimensions were retained, so the added padding could be done later after reconstruction.  
2.3.5 Patch Extraction  
The padded whole slide images were split into non-overlapping patches of size 256×256 with a stride of 256.One patch was saved at a time using coordinate based file names so that the spatial information is retained. Later, this coordinate mapping assisted in the reconstruction of the normalized patches to their original slide arrangement.  
2.3.6 Tissue patch filtering  
Automatically removed non-informative patches with too much white background or black padded areas, low saturation or low texture variance. Only diagnostically useful tissue regions were retained using a threshold for tissue ratio> 0.2 and for variance > 300. This filtering cut down on the extraneous data and retained the useful histopathological information.  
2.3.7 Balanced patch selection  
A predetermined number of useful tissue patches was chosen from each slide to ensure the computational efficiency and balanced of the classes. Around 35 useable patches were obtained on each LUAD slide and 100 useable patches were obtained on each LUSC slide. 
The final balanced data set prepared for stain normalization and graph-based learning.  
2.3.8 LAB-Based Gentle Stain Normalization and Reconstruction  
To reduce stain variability among histopathology slides, LAB-based gentle stain normalization was applied to the selected tissue patches. The normalization process standardized the color appearance of Hematoxylin and Eosin (H&E) stained tissue patches while preserving important tissue morphology, nuclei structures, and cellular details. Unlike aggressive stain normalization approaches, gentle LAB normalization maintained the natural visual appearance of tissue regions and avoided excessive alteration of histopathological characteristics.  
After normalization, only informative tissue-rich patches were reconstructed according to their original spatial coordinates in order to preserve the spatial arrangement of tissue regions. Noninformative background patches and unnecessary regions were excluded from reconstruction. This coordinate-based reconstruction strategy helped maintain tissue consistency and improved the visual comparison between original tissue regions and normalized reconstructed tissue regions.  
2.3.9 Tissue Mask Generation on Reconstructed Normalized Tissue Regions  
After reconstruction of the normalized tissue patches, tissue masks were generated again on the reconstructed tissue regions to evaluate whether important tissue structures and spatial morphology were preserved after stain normalization. The reconstructed normalized tissue regions were processed using image thresholding and morphological operations to separate tissue areas from the background regions.  
2.3.10 Before and After Comparison  
The effectiveness of informative patch selection, stain normalization, and reconstruction was visually evaluated using before and after comparison figures. The comparison visualizations included original whole slide image thumbnails, initial tissue masks, reconstructed normalized tissue regions, and post-normalization tissue masks. These comparisons demonstrated that informative tissue-rich regions were successfully selected while unnecessary background regions were removed during preprocessing.  
2.4.  Graph construction and Representation Learning  
   
Figure 2. 5: Graph construction and self-supervised graph representation learning framework for LUAD and LUSC classification.  
The third objective focuses on representing each whole slide image as a graph and applying a Graph Neural Network (GNN) to classify histopathology slides into LUAD (lung Adenocarcinoma) and LUSC (lung squamous cell carcinoma). It follows different stages of pipeline that combines self-supervised pretraining with supervised fine tuning to learn robust and discriminative graph-based representations from normalized tissue patches. Figure 2.5 presents the graph-based learning framework for LUAD and LUSC classification from histopathology images. First, normalized tissue patches are passed through a pretrained ResNet50 model to extract high-dimensional feature vectors. These features are then used to construct patch graphs, where each node represents a tissue patch and edges are formed using K-nearest neighbor (KNN) connectivity. A Graph Autoencoder (GAE) is applied in a selfsupervised manner to learn meaningful graph representations without requiring class labels. The learned graph embeddings are subsequently fine-tuned using a Graph Convolutional Network (GCN) classifier. Finally, the GCN utilizes both local patch information and global tissue structure to accurately distinguish between LUAD and LUSC whole-slide images.  
2.4.1 CNN feature extraction  
Normalized tissue patches are fed to the pre-trained ResNet50 network to get numerical feature representations. The features that have been extracted are important features of tissue, such as appearance of the nuclei, texture of cells , and are arrangement of tissue. The CNN transforms the patterns in the histopathology into meaningful deep features rather than analyzing the raw pixels directly. These are sequently used as feature vectors for graph construction and graph learning.  
2.4.2 Graph construction KNN connectivity  
K-nearest neighbor connectivity with 5 neighbors is used on each whole slide image to create a graph structure. Each of the tissue patches is a node in this graph; the edge are similarity relationship between neighboring patches. This method maintains the spatial and contextual relationships of the tissue regions. This means that the framework more accurately represents tissue interactions as compared to patch independent methods.  
2.4.3 Self-supervised graph representation learning  
A graph Auto encoder (GAE) is used to model the graph representations in a self-supervised way, relying less than heavily on labeled data. At the training time, the encoder learns the latent representations of the graph by reconstructing the relationships and neighborhood structures of graphs. The reconstruction loss is slowly decreasing, suggesting that something useful is being learned about the graph. This process aids the model to automatically learn tissue structures and connectivity pattern.  
2.4.4 GCN classification (fine tuning)  
The pretrained graph embedding’s are fine tuned with a Graph Convolutional Network (GCN) classifier for the classification of LUAD and LUSC. The GCN is able to incorporate morphology data from the local areas along with the graph relationships from the rest of the graph to boost the prediction accuracy. The model learns more discriminative tissue patterns by aggregation of information from adjacent neighbor nodes that are connected. This graphbased method can achieve more stable classification than conventional CNN based methods. The fine tuned classifier is tested on the held out test set of  unseen graphs .The performance is evaluated by the means of standard classification metrics:  
•	Accuracy - To ensure the overall correctness of the classification model, the accuracy can be calculated according to the number of the samples predicted correctly and the total number of samples. The larger the accuracy value, the better the model is able to classify the tissue images of LUAD and LUSC.  
•	Precision measures how many samples predicted as a class are correct.It plays an important role in the classification of histopathology as it lowers the number of false positive predictions.  
•	Recall is the ability of the model to correctly classify all the positive samples from the data .The higher the recall the fewer cancer cases will be missed during classification.  
•	F-1 score is the harmonic means of the precision and recall scores, and it is used to balance the two scores. It is helpful for assessing the performance of classification on a data set where a balanced prediction is crucial.  
•	AUC–ROC is used to evaluate the performance of the model over various classification thresholds against separating LUAD and LUSC classes. The larger the AUC, the more discriminatory it will be and the better its overall classification performance.  
2.4.5 Analysis of visualization and Representation  
To better study the behavior of the models .some visualization methods are used.  
•	Training Curves  
•	Curves of loss and accuracy are observed over the epochs to track convergence and identify over fitting or under fitting.  
•	Embedding Visualization  
The trained graph level embedding’s are reduced to two -dimensions with:  
•	t-SNE(t-Distributed Stochastic Neighbor Embedding)  
•	PCA(Principal Component Analysis)  
These plots show the degree of separation between LUAD and LUSC samples in the learned features space.  
•	Graph Visualization:  
•	Graph constructed are all plotted   Nodes are patches of tissue.  
•	Edges denote similarity on KNN.  
This gives an understandable representation of the graph forms utilized by the model.  
  
  
  
  
  	  
 
Chapter 3 - RESULTS & DISCUSSION  
3.1  	Informative Patch Selection  
The informative patch selection process was performed to retain tissue-rich regions that contain diagnostically relevant information while removing non-informative areas such as excessive background, empty regions, and low-quality tissue patches. Tissue masks generated in the previous step were used to identify candidate tissue regions, and only patches satisfying predefined tissue content and quality thresholds were selected. This process reduced computational complexity and ensured that subsequent stain normalization, graph construction, and representation learning were performed using meaningful histopathological information. The selected patches preserved important cellular and tissue structures required for accurate LUAD and LUSC analysis.  
3.1.1 Whole Slide Thumbnail Generation  
Figure 3.1 and figure 3.2 shows Whole Slide images of LUAD and LUSC were preprocessed in the first stage by generating thumbnails of each slide. Thumbnail generation is used to get a quick look at the whole slide and gave a low-resolution view of the tissue structure prior to tissue segmentation and patch extraction. The generated thumbnails showed that besides diagnostically relevant tissue areas, there are large nontissue areas in the images of a whole slide histopathology image. Thus, background regions had to be removed prior to the extraction of the patches. The LUAD and LUSC thumbnail generation results showed that the proposed preprocessing pipeline was able to load SVS image files using Open Slide, and to produce meaningful slide representations to analyze tissue for future research.   
   
Figure 3. 1: Generated thumbnails of LUAD whole slide histopathology images  
   
Figure 3. 2: Generated thumbnails of LUAD whole slide histopathology images  
3.1.2 Tissue mask generation  
Representative LUAD and LUSC histopathology image regions together with their corresponding tissue masks are presented in Figure 3.3 and in Figure 3.4 .The tissue masks were generated using image thresholding and morphological operations to distinguish tissue regions from background areas. The visual results indicate that the preprocessing framework successfully identified the tissue-rich regions while excluding non-informative background areas. The generated masks preserved important histopathological structures and provided a reliable basis for informative patch extraction. These findings demonstrate the effectiveness of the tissue masking process in reducing unnecessary image content and improving the quality of subsequent preprocessing stages.  
Tissue masks were created using grayscale thresholding and morphological operations. The masks produced correctly distinguished areas of tissues from background areas. The generated tissue masks highlighted informative tissue regions while removing unnecessary background regions from the generated tissue masks.  
Tissue masks were able to identify tissue rich structures and remove the empty white space in the slide in LUAD-WSI  . Likewise, in the case of LUSC slides, the tissue masking process retained the key pathological features and removed the non-essential areas. The tissue mask generation results showed that the proposed mask was effective in isolating tissue areas of interest before patch extraction. This pre-processing step enhanced the quality of the patches and minimized the presence of non-informative regions of the image.   
   
Figure 3. 3: LUAD Thumbnail Generation  
   
Figure 3. 4: LUSC Thumbnail Generation  
3.1.3 Patch Extraction Results  
In the proposed framework, once the tissue mask generation, the framework extracted the non overlapping tissue patches from the whole slide image with a patch size 256×256 pixel. The extraction of patches was done independently for LUAD and LUSC with  optimized tissue thresholds and variance filtering parameters  
LUAD preprocessing was performed by informative tissue patches with tissue ratio filtering and variance thresholding. The removed patches had a useful tissue morphology and avoided the slide background.  
In the case of LUSC pre processing,more stringent thresholds were taken to eliminate white regions, black regions, and low information patches. This enhanced the quality of selected tissue patches, and allowed for the preservation of meaningful histopathological structures. The selected patches showed rich cellular and tissue morphology suitable for graph representation learning.  
   
Figure 3. 5: Representative LUAD informative patch selected after tissue filtering  
Selected LUAD patch in Figure 3.5 shows that after applying tissue ratio and variance based filtering. The retained patches contain meaningful histopathological structures, while patches with excessive background or low tissue information were removed. This result supports the first objective by showing that the framework preserved diagnostically relevant tissue areas for later feature extraction and graph construction.  
   
Figure 3. 6: Representative LUSC informative patch selected after tissue filtering  
Figure 3.6 shows representative LUSC patch selected by improved filtering pipeline. These patches contain tissue rich regions with visible cellular and structural patterns. The removal of non-informative patches reduced unnecessary computational load and improved the quality of graph nodes used in later graph learning stage.  
3.1.4 Informative Patch Filtering  
The proposed framework further enhanced the patch quality by using informative patch filtering techniques based on ratio, variance threshold, saturation threshold, white threshold and black threshold analysis.  
The information and filtering stage was successfully carried out to remove:  
•	White background regions  
•	Black padded regions  
•	Low information patches  
•	Non-tissue areas  
•	Blurred tissue regions  
This filtering process greatly enhanced the overall quality of tissue patches used in the graph learning process. The filtered patches showed better tissue quality, sharper cellular structures and more uniform morphological information than unfiltered patches.  
3.1.5 Balanced Patch Selection  
The preprocessing pipeline proposed also ensures balance in the selection of patches between LUAD and LUSC data sets. This balancing process helped to decrease class imbalance in the graph learning as well as to increase classification stability.  
In LUAD 35 informative tissue patches were mined per slide during the first preprocessing pipeline. The enhanced LUSC preprocessing pipeline was used to optimize the extraction of up to 100 high quality tissue patches with richness and background noise removal. The stable construction of the graph and the better performance of graph learning were achieved by balanced patch selection during the training process of GCN.  
3.2  	Stain Normalization and color Correction  
The goal of this second objective is to normalized the stain and correct the color of the stain. The second aim of this research was to diminish the stain variation in the histopathology images by utilizing LAB-based gentle stain normalization and color correction methods. Histopathology image tend to have variations in staining as they are image from slides prepared by different laboratory processes, scanners, etc .The proposed framework tried to lower these inconsistences by using LAB-based stain normalization on both LUAD and LUSC tissue patches  
3.2.1 LAB-Based Gentle Stain Normalization  
After informative tissue patches were selected, LAB-based gentle stain normalization was applied to reduce color variation across the dataset. The normalization process adjusted the color characteristics of the tissue patches while preserving important cellular and structural information. Visual inspection of the normalized LUAD and LUSC patches demonstrated improved color consistency and reduced staining variability compared with the original patches.  
The normalization process successfully preserved:  
•	Cellular boundaries  
•	Tissue morphology  
•	Structural tissue patterns  
•	Histopathological texture information  
•	Nuclear and cytoplasmic details  
At the same time, unnecessary color differences caused by staining variations were minimized. The normalized tissue patches exhibited a more balanced and visually consistent appearance across different slides. This improved consistency provided a more reliable input for subsequent graph construction, feature extraction, and representation learning. Consequently, the normalized LUAD and LUSC tissue patches were better suited for graph-based analysis and classification tasks.  
  	  
3.2.1.1 Patch-Level Stain Normalization   
	Original LUAD Patch   	  	  	LAB-Normalized LUAD Patch  
 	   	   
	Original LUAD Patch   	  	  	LAB-Normalized LUAD Patch  
 	   	   
Figure 3. 7: Representative LUAD tissue patches before and after LAB-based gentle stain normalization  
  	  
 	Original LUSC Patch   	  	  	LAB-Normalized LUSC Patch  
 	   	                   
Original LUSC Patch                                     LAB-Normalized LUSC Patch  
 	   	   
Figure 3. 8: Representative LUSC tissue patches before and after LAB-based gentle stain normalization  
In Figure 3.7 and Figure 3.8 presents two representative tissue patches original patch and after LAB-based gentle stain normalization of LUAD and 2 representative tissue patches original patch and after LAB-based gentle stain normalization of LUSC. The visual comparison indicates that the normalization process improved stain consistency while preserving important histopathological characteristics. Cellular structures, nuclei morphology, tissue boundaries, and texture patterns remained largely unchanged after normalization. The normalized patches exhibit a more balanced color appearance and reduced stain variability, demonstrating the effectiveness of the proposed normalization approach. These results suggest that LAB-based gentle stain normalization can improve visual consistency across tissue patches while maintaining diagnostically relevant information required for subsequent graph construction and representation learning.  
3.2.2 Reconstruction of Normalized LUAD and LUSC patches  
The reconstructed normalized patches of LUAD and LUSC showed that the proposed preprocessing framework maintained tissue morphology in a wide tissue area. The reconstructed normalized patches characterized by the visualization of more uniform stain appearance and contrast, and similar to the original tissue structures. The reconstructed normalized tissue slides validated:  
•	Spatial tissue organization  
•	Tissue architecture  Morphological consistency   Diagnostic tissue patterns.  
The results show that the normalization method proposed by LAB-based gentle stain normalization successfully minimized the variations in stain without losing important histopathological features for graph representation learning.  
 
Figure 3. 9: Comparison between the original LUAD whole slide images region and the reconstructed LABnormalized tissue patch grid.  
Figure 3.9 presents a comparison between the original LUAD whole slide image region and the reconstructed LAB-normalized tissue patch grid. Only informative tissue-rich patches selected during preprocessing were reconstructed; therefore, the reconstructed image represents diagnostically relevant tissue regions rather than the complete whole slide image. The reconstructed image was generated by arranging the normalized tissue patches according to their original spatial coordinates. The visual results demonstrate that the reconstruction process successfully preserved the spatial organization and structural characteristics of the tissue while incorporating the effects of stain normalization. Important histopathological features, including cellular morphology, glandular structures, and tissue architecture, remained clearly visible after reconstruction. Furthermore, the reconstructed patch grid exhibited a more consistent staining appearance while maintaining the diagnostic information present in the original tissue region. These findings indicate that the coordinate-based reconstruction approach effectively retained tissue continuity and provided a suitable representation for subsequent graph construction and representation learning.  
 
   
Figure 3. 10: Comparison between the original LUSC whole slide image thumbnail and the coordinate-based reconstructed LUSC tissue regions.  
Figure 3.10 shows the comparison between the original LUSC whole slide image thumbnail and the coordinate-based reconstructed tissue regions obtained from the selected informative patches. Only informative tissue-rich patches selected during preprocessing were reconstructed; therefore, the reconstructed image represents diagnostically relevant tissue regions rather than the complete whole slide image. The reconstructed tissue image preserved the spatial arrangement and morphological characteristics of the selected tissue regions despite being reconstructed from a limited number of informative patches. The visual results indicate that the reconstruction process successfully retained diagnostically relevant tissue information while removing non-informative background regions.  
3.2.3 Tissue mask generation after reconstruction  
   
Figure 3. 11: Comparison of the original LUAD tissue region and initial tissue mask with the reconstructed normalized tissue patches and tissue mask after normalization.  
Figure 3.11 presents the LUAD tissue region before and after reconstruction. The initial tissue mask successfully identified tissue-rich regions in the original image, while the tissue mask generated from the reconstructed normalized patches highlighted the retained informative tissue areas. Since only selected informative patches were reconstructed, the postreconstruction mask represents diagnostically relevant tissue regions rather than the complete original image. The results indicate that important tissue structures were preserved during normalization and reconstruction, demonstrating the effectiveness of the preprocessing framework.  
   
Figure 3. 12: Comparison of the original LUSC whole slide image thumbnail and initial tissue mask with the coordinate-based reconstructed tissue regions and tissue mask after reconstruction.  
Figure 3.12 illustrates the tissue mask generation results for LUSC after coordinatebased reconstruction. The reconstructed tissue regions maintained the spatial arrangement of the selected informative patches, and the corresponding tissue mask successfully identified the retained tissue structures. Although the reconstructed mask differs from the original mask, this difference is expected because only informative tissue patches were used during reconstruction. The results confirm that the reconstruction process preserved diagnostically relevant tissue information while eliminating non-informative regions.  
3.3  	Graph Construction and Representation Learning  
The Third Objective of this research focused on graph-based representation learning using selfsupervised graph auto encoder pretraining and graph convolutional network classification.  
The proposed framework consisted of mapping the normalized tissue patches onto the graph, and modelling the KNN relationship between normalized tissue patches as graph edge. The graph-based learning approach outperformed the independent patch processing approaches in retaining tissue relationships.  
3.3.1 ResNet5o Feature Extraction  
The pretrained ResNet50 model was used for deep feature extraction of the normalized LUAD and LUSC tissue patches. High level histopathological information was represented in a 2048 dimensional feature vector for each tissue patch. The deep feature extracted include:  
•	Tissue morphology  Cellular patterns.  
•	Structural relationships.  
•	Histopathological texture information  
This extracted feature vectors were then used for graph building and graph representation learning.  
3.3.2 Graph Construction results  
After feature extraction, graphs were constructed separately for LUAD and LUSC whole slide images using K nearest neighbor connectivity.  
In the constructed Graphs:  
•	Tissue patches were represented as graph nodes  
•	Similarity Relationships were represented as graph edges  
•	K nearest neighbor connections preserved tissue relationships  
The graph construction results demonstrated successful modeling of tissue relationships among neighboring histopathology patches.  
The generated LUAD patch graphs contained approximately 24 nodes and 120 edges. The generated LUSC patch graphs contained approximately 100nodes and 500 edges.  
The graph visualizations confirmed that the proposed framework successfully modeled tissue connectivity and preserved spatial relationships among histopathology patches.  
   
Figure 3. 13: LUAD patch graph visualizations showing 24 nodes and 120 edges  
Figure 3.13 shows representative LUAD patch graphs generated  from normalized LUAD patches. In each graph, nodes represent selected tissue patches and the edges represents Knearest neighbor similarity relationships between patches. The LUAD graph examples contain 24 nodes and 120 edges, showing that the proposed framework successfully converted selected tissue patches into graph structures for representing learning.  
   
Figure 3. 14: LUSC patch graph visualizations showing 100 nodes and 500 edges.  
Figure 3.14 shows representative LUSC patch graphs generated from normalized LUSC tissue patches. Each LUSC graph contains 100 nodes and 500 edges. The large number of nodes reflects the use of 100 informative LUSC patches per slide in the improved pipeline. These graph visualizations confirm that tissue patch relationships were successfully modeled before self -supervised graph representation learning.  
3.3.3 Self supervised Graph Auto encoder pretraining  
The constructed graphs were processed using a graph auto encoder in a self-learning framework. The graph auto encoder learned graph embedding’s without depending heavily on expert annotations.  
The self-supervised pretraining stage demonstrated stable convergence during training. The pretraining loss curve showed a strong reduction from higher initial values toward lower stabilized values over 30 epochs. The decreasing pretraining loss confirmed that the graph autoencoder successfully learned meaningful structural representations from LUAD and LUSC tissue graphs.  
   
Figure 3. 15: Self-supervised graph auto encoder pretraining loss curve.  
Figure 3.15 presents the self-supervised graph auto encoder pretraining loss curve. The loss decreases sharply during the initial epochs and then gradually stabilizes around a lower value by the end of 30 epochs. This trend indicates that the graphs autoencoder learned useful structural information from LUAD and LUSC patch graphs without depending on expert labeled patch annotations.  
3.3.3 GCN Fine Tuning Results  
The graph auto encoder was fine tuned for LUAD and LUSC classification with a graph convolutional network after the graph auto encoder pre training. The graph convolutional network showed fast convergence in its supervised fine tuning. The training loss was continually reduced across training epochs, and converged towards close to zero values in later epochs. The graph convolutional network could learn the discriminative graph representation for both LUAD and LUSC classification. Stable accuracy was observed during training and validated, as evidenced by the accuracy curves. Both curves showed significant improvement in the early epochs, and after early epochs, they became stabilized at high accuracy values.  
This small gap between the accuracy of training set and the validation set validated that there was no significant over fitting while generalizing well.  
   
Figure 3. 16: GCN fine-tuning loss curve for LUAD and LUSC classification  
Figure 3.16 shows the GCN fine tuning loss curve. The loss decrease rapidly during early epochs and approaches nearly zero during epochs. This confirms that the supervised GCN classifier successfully learned discriminative graph representation for LUAD and LUSC classification.  
   
Figure 3. 17: Training loss curve of the proposed GCN classifier  
Figure 3.17 presents the training loss curve of the proposed graph classification model. The curve shows a continuous decrease in loss, indicating stable learning and effective optimization during model training.  
   
Figure 3. 18: Training and validation accuracy curve of the proposed graph learning framework  
Figure 3.18 shows the training and validation accuracy curve. Both accuracy curves reach 1.00 after early epochs and remain stable, confirming that the proposed framework achieved strong learning performance without visible instability in validation accuracy.  
3.4.5 Classification Performance  
Evaluation of classification performance was done by accuracy, precision, recall, F1 score and AUC-ROC of the proposed framework  
The graph learning framework proposed was achieved with:  
Table 3. 1: Evaluation  
Accuracy  	1.0  
Precision  	1.0  
Recall  	1.0  
F1-score  	1.0  
AUC-ROC  	1.0  
  
Table 3.1 shows the classification model achieved excellent performance across all evaluation metrics. An accuracy of 100% indicates that all test samples were correctly classified. Precision, recall, and F1-score values of 1.0 demonstrate perfect prediction quality with no false positives or false negatives. Similarly, an AUC-ROC score of 1.0 confirms the model's outstanding ability to distinguish between LUAD and LUSC classes. These results indicate that the proposed graph-based framework effectively learned discriminative features and achieved highly accurate classification performance on the evaluated dataset.  
The results illustrate the successful learning of highly discriminative graph representations for the classification of LUAD and LUSC using the proposed framework. The results show that using informative patch selection, stain normalization, self-supervised graph learning, and graph convolutional classification greatly enhanced the representation learning of histopathology images.  
3.4.6 Confusion Matrix Analysis  
The whole slide images of LUAD and LUSC exhibited perfect classification performance in the confusion matrix. The proposed frame work accurately labeled:  
•	LUAD graphs as LUAD  
•	LUSC graphs as LUSC   
No graph of LUAD was misclassified as LUSC and no graph of LUSC was misclassified as  LUAD.  
The confusion matrix verifies that the designed graph learning framework was able to learn meaningful graph embedding’s for tissue structure identification between LUAD and LUSC .The diagonal structure of the confusion matrix also further confirms the stable graph representation learning and high reliability of classification behavior.  
   
Figure 3. 19: Confusion matrix of the proposed GCN model for LUAD and LUSC    classification  
Figure 3.19 presents the confusion matrix for LUAD and LUSC classification. The matrix shows that 4 LUAD samples were correctly classified as LUAD and 3 LUSC samples were correctly classified as LUSC. No LUAD sample was misclassified as LUSC and no LUSC sample was misclassified as LUAD .This confirms that the proposed graph learning framework achieved correct classification an the evaluated test graphs.  
3.4.7 PCA Graph Embedding Visualization  
To show learned graph embedding’s produced by the proposed framework, the principal Component Analysis (PCA) was undertaken.  
Clear separation was observed between LUAD and LUSC graph embedding achieved by the PCA visualization. The cluster generated suggested that the graphs auto encoder and the graph convolutional network were able to learn discriminative representations for both cancer subtypes. The PCA embedding result showed that the proposed graph learning framework retained meaningful structural information in the embedding process.  
The PCA embedding results showed that the proposed graph learning framework retained meaningful structural information in the embedding process.  
   
Figure 3. 20: PCA visualization of graph embeddings for LUAD and LUSC sample  
The graph embedding learnt is visualized with PCA in figure 3.20. The two lung cancer subtypes (LUAD and LUSC) are clearly separated in the graph embedding’s, suggesting that the proposed graph learning framework extracted discriminative features for both lung cancer sun types.  
3.4.8 t-SNE Graph Embedding Visualization  
The t-SNE visualization further demonstrated the quality of graph embedding’s .The tSNE plot showed well separated LUAD and LUSC graph clusters with minimal overlap between classes. This  clear separation indicates that the proposed graph learning framework effectively captured discriminative tissue representations.  
The t-SNE visualization confirmed that the learned graph embedding’s contained strong class specific information useful for downstream classification tasks.  
   
Figure 3. 21: t-SNE visualization of graph embedding’s for LUAD and LUSC samples  
Figure 3.21 presents the t-SNE visualization of graph embedding’s. The LUAD and LUSC clusters are clearly separated, showing that the learned graph representation captured class specific histopathological information. This supports the effectiveness of graph based representation learning for distinguishing LUAD and LUSC whole slide images.  
3.4.9 Graph Representation Quality Analysis  
The overall graph representation learning results showed that the proposed framework was able to preserve the spatial relationships and tissue context during WSI analysis with good success.  
In contrast with the existing patch based frameworks that work independently on each patch, the proposed graoh learning framework captured:  
•	Local tissue relationships  
•	Spatial connectivity  
•	Morphological similarity  
•	Structural context   
These graph relationships improved representation learning quality and classification performance.     
 
      Chapter 4- Conclusion 
Chapter 4 - Conclusion  
This research presented a graph neural network-based framework for representation learning in whole slide histopathology images using LUAD and LUSC lung cancer datasets obtained from TCGA. The study focused on solving important challenges in computational pathology, including large whole slide image size, stain variability, background noise, loss of spatial tissue relationships, and dependence on heavily labeled datasets. To address these problems, a complete preprocessing and graph learning framework was developed that combines informative patch selection, stain normalization, graph construction, self-supervised graph representation learning, and graph convolutional network classification.  
The findings of this research show that preprocessing plays an important role in improving histopathology image analysis. The proposed patch optimization framework successfully removed non-informative regions such as white background, black padded areas, and low information tissue patches while preserving tissue rich diagnostic regions. Tissue masks and informative patch extraction results demonstrated that the proposed pipeline effectively separated meaningful tissue structures from unnecessary image regions. This improved the overall quality of tissue patches used during graph learning and reduced the influence of background noise on feature extraction.  
The stain normalization stage also produced important improvements in image consistency. Lab gentle stain normalization reduced variations in staining appearance across LUAD and LUSC slides while preserving tissue morphology and structural information. The before and after comparison results confirmed that the normalized patches became visually more consistent and suitable for graph-based learning. This demonstrates that stain normalization can improve the reliability of histopathology image representation learning by reducing the effect of laboratory and slide preparation differences.  
The graph-based learning framework successfully modeled spatial and morphological relationships among tissue patches. Instead of analyzing patches independently, the proposed framework converted normalized tissue patches into graph structures, where each patch was represented as a graph node and tissue similarity was represented as graph edges. This graph representation preserved important contextual information that is often lost in traditional patch based deep learning approaches.  
The self-supervised graph autoencoder successfully learned meaningful graph embeddings without depending heavily on expert annotations. The decreasing pretraining loss curve demonstrated that the graph autoencoder effectively captured structural relationships among tissue patches during representation learning. After graph autoencoder pretraining, the graph convolutional network classifier achieved stable and highly accurate LUAD and LUSC classification performance.  
The final classification results showed strong performance across all evaluation metrics, including accuracy, precision, recall, F1-score, and AUC-ROC. The confusion matrix, PCA visualization, and t-SNE visualization further confirmed that the proposed framework learned       
Chapter 4- Conclusion 
discriminative graph representations capable of separating LUAD and LUSC tissue patterns effectively. The clear separation between graph embeddings demonstrated that graph neural networks can preserve tissue context and improve histopathology image understanding.  
The results of this study also support the findings of earlier graph based computational pathology studies. Previous researchers reported that graph neural networks preserve tissue topology and contextual relationships more effectively than conventional CNN based patch learning approaches. The proposed framework further improved this concept by integrating preprocessing optimization, stain normalization, and self-supervised graph representation learning into a unified workflow.  
Overall, this research successfully achieved all proposed objectives. The first objective was achieved through informative tissue patch selection and filtering. The second objective was achieved through lab gentle stain normalization and color correction. The third objective was achieved through graph construction, self-supervised graph representation learning, and GCN based classification. The obtained results demonstrate that graph neural networks provide an effective and reliable framework for whole slide histopathology image analysis.  
In conclusion, the proposed framework contributes to the field of computational pathology by presenting a complete graph-based representation learning pipeline for lung cancer histopathology images. The framework improves tissue representation quality, preserves spatial tissue relationships, reduces stain variation, and supports reliable graph-based classification. The study highlights the potential of graph neural networks for developing intelligent digital pathology systems that may assist pathologists in automated cancer diagnosis and histopathology image interpretation.  
  
  	  
 	 
REFERENCES  
[1]	Adnan, M., Kalra, S. and Tizhoosh, H.R. 2020. Representation learning of histopathology images using graph neural networks. Proc. IEEE Conf. Comput. Vis. Pattern Recognit.  
Workshops 1:1–8.  
[2]	Alfasly, S., Shafique, A., Nejat, P., Khan, J., Alsaafin, A., Alabtah, G. and Tizhoosh, H.R. 2024. Rotation-agnostic image representation learning for digital pathology. arXiv preprint arXiv:2311.08359.  
[3]	Almubarak, H.A. 2025. Self-supervised learning for data augmentation in histopathology image segmentation. Sci. Rep. 15:39596.  
[4]	Anderson, D., Ramachandran, P., Trapp, J. and Fielding, A. 2025. A review of image processing and analysis of computed tomography images using deep learning methods. Journal of Medical Imaging 1:1–15  
[5]	Bazargani, R., Fazli, L., Gleave, M., Goldenberg, L., Bashashati, A. and Salcudean, S. 2024. Multi-scale relational graph convolutional network for multiple instance learning in histopathology images. Med. Image Anal. 96:103197.  
[6]	Brussee, S., Buzzanca, G., Schrader, A.M.R. and Kers, J. 2024. Graph neural networks in histopathology: emerging trends and future directions. arXiv Preprint arXiv:2406.12808.  
[7]	Carlos, L., Monroy, R., Rist, L., Breininger, K., Maier, A., Ostalecki, C., Bauer, A. and Vera, J. 2025. Graph neural networks in multi-stained pathological imaging: extended comparative analysis of radiomic features. International Journal of Computer Assisted Radiology and Surgery 20:497–505.  
[8]	Chan, T.H., Cendra, F.J., Ma, L., Yin, G. and Yu, L. 2023. Histopathology whole slide image analysis with heterogeneous graph representation learning. arXiv preprint arXiv:2307.04189.  
[9]	Chen, C., Lu, M.Y., Williamson, D.F.K., Chen, T.Y., Schaumberg, A.J. and Mahmood, F. 2022. Fast and scalable search of whole-slide images via self-supervised deep learning. 
Nat. Biomed. Eng. 6:1420–1434.  
[10]	Ding, T., Wagner, S.J., Song, A.H., Chen, R.J., Lu, M.Y., Zhang, A., Vaidya, A.J., Jaume,  
G., Shaban, M., Kim, A., Williamson, D.F.K., Robertson, H., Chen, B., Almagro-Perez, 
C., Doucet, P., Sahai, S., Chen, C., Komura, D., Kawabe, A., Ochi, M., Sato, S.,     
Yokose, T., Miyagi, Y., Ishikawa, S., Gerber, G., Peng, T., Le, L.P. and Mahmood, F. 2025. A multimodal whole-slide foundation model for pathology. Nat. Med. 31:3749– 3761.  
[11]	Ding, R., Luong, K.D., Rodriguez, E., Silva, A.C.A.L. and Hsu, W. 2024. Combining graph neural network and Mamba to capture local and global tissue spatial relationships in whole slide images. arXiv Preprint arXiv:2406.04377.  
[12]	Farace di Villaforesta, A., Magister, L.C., Barbiero, P. and Lio, P. 2023. Digital histopathology with graph neural networks: concepts and explanations for clinicians. arXiv Preprint arXiv:2312.02225.  
[13]	Fashi, P.A., Hemati, S., Babaie, M., Gonzalez, R. and Tizhoosh, H.R. 2022. A selfsupervised contrastive learning approach for whole slide image representation in digital pathology. J. Pathol. Inform. 13:100133.  
[14]	Ge, R., Chen, G., Saruta, K. and Terata, Y. 2024. Tumor detection in breast cancer pathology patches using a multi-scale multi-head self-attention ensemble network on whole slide images. Mach. Learn. Appl. 18:100592.  
[15]	Hashemian, S. and Bidgoli, A.A. 2026. EvoPS: Evolutionary patch selection in the training embedding space of whole slide images. Artif. Intell. Med. 176:103398.  
[16]	Hirra, I., Ahmad, M., Hussain, A., Ashraf, M.U., Saeed, I.A., Qadri, S.F., Alghamdi, A.M.  
and Alfakeeh, A.S. 2021. Breast cancer classification from histopathological images using patch-based deep learning modeling. IEEE Access 9:24273–24286.  
[17]	Hou, L., Samaras, D., Kurc, T.M., Gao, Y., Davis, J.E. and Saltz, J.H. 2016. Patch-based convolutional neural network for whole slide tissue image classification. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. 1:2424–2433.  
[18]	Ilse, M., Tomczak, J.M. and Welling, M. 2018. Attention-based deep multiple instance learning. Proc. Int. Conf. Mach. Learn. 80:2127–2136.  
  
