# Abstract
Medical image classification is crucial for improving diagnosis and treatment, especially when human analysis is time-consuming and less precise compared to Computer-Aided Diagnosis (CAD) systems. Our study focused on classifying microscopic fungi images, addressing a significant yet frequently overlooked health threat posed by fungal infections. We evaluated the effectiveness of transfer learning using four pre-trained models **(VGG16, ResNet50, EfficientNetb0, and ViT16)** using various metrics like balanced accuracy, MCC, F1-score, and confusion matrices. We also compared deep learning models with traditional machine learning algorithms such as **Logistic Regression, Naive Bayes, and Random Forest**.  To ensure accurate model assessment, we implemented **cross-validation** for deep learning models to better assess their performance on unseen data. Notably, our findings reveal that **EfficientNet outperformed all other models, achieving a remarkably balanced accuracy of 0.9 when augmented**. Additionally, we employed GRAD-CAM for **model explainability** and visualized the Attention mechanism for ViT. These findings underscore the significant potential of deep learning models in medical image classification and their crucial role in addressing critical healthcare challenges.

---

# File descriptions 

## Other
- `Project_Report.pdf` is the 25 page report of our study 
## Notebooks
- `classic-ml.ipynb` contains code for preprocessing and classifying the images utilizing classical ML algorithms
- `data_exploration.ipynb` contains code for the EDA of the dataset
- `fungi-efficientnet-classification_final.ipynb` contains code for the transfer learning of EfficientNetb0
- `visiontransformer.ipynb` contains code for the transfer learning of Vision Transformer Base 16
- `fungivgg16classification.ipynb` contains code for the transfer learning of VGG16
- `fungi-resnet-classification_final` contains code for the transfer learning of ResNet50
## Scripts 
- `engine.py` contains functions utilized for the training of Pytorch models
- `helper-functions.py` contains functions mostly used to visualize the results of classification and training tasks.
