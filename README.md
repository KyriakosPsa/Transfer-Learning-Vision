# Abstract
Medical image classification is crucial for improving diagnosis and treatment, especially when human analysis is time-consuming and less precise compared to Computer-Aided Diagnosis (CAD) systems. Our study focused on classifying microscopic fungi images, addressing a significant yet frequently overlooked health threat posed by fungal infections. We evaluated the effectiveness of transfer learning using four pre-trained models **(VGG16, ResNet50, EfficientNetb0, and ViT16)** using various metrics like balanced accuracy, MCC, F1-score, and confusion matrices. We also compared deep learning models with traditional machine learning algorithms such as **Logistic Regression, Naive Bayes, and Random Forest**.  To ensure accurate model assessment, we implemented **cross-validation** for deep learning models to better assess their performance on unseen data. Notably, our findings reveal that **EfficientNet outperformed all other models, achieving a remarkably balanced accuracy of 0.9 when augmented**. Additionally, we employed GRAD-CAM for **model explainability** and visualized the Attention mechanism for ViT. These findings underscore the significant potential of deep learning models in medical image classification and their crucial role in addressing critical healthcare challenges.

---
# Important Results 
A complete 25-page report of our study can be accessed via the `Project_Report.pdf`. For brevity, we only showcase the most important figures/findings here.

Performance of classic machine learning algorithms on Histogram of Oriented Gradients (HOG) features extracted from images using OpenCV:

![image](https://github.com/user-attachments/assets/7e60d04d-484e-4c36-98b5-dc6d68daf51a)

Performance of the VGG16, EfficientNetb0, ResNet50 and Vision Transformer pre-trained deep learning architectures, with weight fine-tuning:

![image](https://github.com/user-attachments/assets/455917c5-36b6-4d87-b864-4178022807d0)

Visualization of Gradients on example microscopy images per deep learning architecture:

![image](https://github.com/user-attachments/assets/1ee34c04-d35b-4bf1-9f45-f551f6c9ed46)

Final Results:

![image](https://github.com/user-attachments/assets/edefeacd-e4b8-4d17-8895-0e956bff9224)

Based on our findings we can safely say that Deep Learning models, especially when used in the context of transfer learning, can demonstrate outstanding performance in image classification tasks compared to traditional Machine Learning algorithms. Moreover, the DL models’ generalization ability on unseen data from the same distribution is far superior, having Technical Reporting more than 40% better scores than the best ml classifier. This is in alignment with the existing literature and findings from contemporary research. Another important result of this study is the fact that EfficientNetb0 outperformed all other models, when trained both on the original and the augmented dataset, despite it having less parameters than the other models. This indicates that the multi-objective neural architecture search performed from its authors can be a promising method for the architectural design of neural networks. These findings can be a starting point for further experimentation with different data augmentation strategies as well as model architectures. Hyperparameter tuning, which was not utilized in this study could also help in achieving better scores. Furthermore, taking advantage the results of the explainability methods can also boost overall performance.

# File descriptions 

## *Other*
- `Project_Report.pdf` is the 25-page report of our study 
## *Notebooks*
- `classic-ml.ipynb` contains code for preprocessing and classifying the images utilizing classical ML algorithms
- `data_exploration.ipynb` contains code for the EDA of the dataset
- `fungi-efficientnet-classification_final.ipynb` contains code for the transfer learning of EfficientNetb0
- `visiontransformer.ipynb` contains code for the transfer learning of Vision Transformer Base 16
- `fungivgg16classification.ipynb` contains code for the transfer learning of VGG16
- `fungi-resnet-classification_final` contains code for the transfer learning of ResNet50
## *Scripts* 
- `engine.py` contains functions utilized for the training of Pytorch models
- `helper-functions.py` contains functions mostly used to visualize the results of classification and training tasks.
