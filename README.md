# Detecting nail diseases with the help of AI
Presentation: [slides](https://github.com/hannesoehler/nail-disease-classification/blob/main/Nail-disease-classification.pdf)

This project consists of two parts: 

1. Detecting nail diseases in nail images
2. Segmenting the area of the nail affected by the disease to determine the severity of the disease

First, we replicated the results in [Han et al. 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0191493) and achieved more accurate results with the help of strong data augmentation (i.e., cutout, mixout) (Table 1). Furthermore, we tested our model on a test set of European nail images scraped from the web (see the presentation for more details on the datasets). As can be seen in Table 2, our model (ensemble of Efficientnet5 and Resnet152) is very good in distinguishing between healthy nails and nails with a disease. More challenging is the identification of a particular disease.

In the second part, we train a model (Yolov5x-Segmentation) to detect the area affected by the disease in order to determine the severity of the disease. As can be seen graphically in Table 3 and according to the Dice coefficient in Table 4, the model is relatively strong in segmenting the correct area.

## Table 1: Detecting nail diseases - comparison of our results to Han et al. (2018)

![image](https://user-images.githubusercontent.com/72496477/228537717-a299a12f-e35b-4ca0-b6f7-3c54e1ca6692.png)

## Table 2: Detecting nail diseases - results on the European test set

![image](https://user-images.githubusercontent.com/72496477/228537803-1e39da0f-2cce-4a59-87e2-87b40a13b22f.png)

## Table 3: Segmentation of the area affected by the disease

![image](https://user-images.githubusercontent.com/72496477/228540712-f88027f7-b773-468f-85f0-d367730786ae.png)

## Table 4: Dice coefficient 

![image](https://user-images.githubusercontent.com/72496477/228542859-3fb0a953-830c-4226-af58-0b249281885e.png)


