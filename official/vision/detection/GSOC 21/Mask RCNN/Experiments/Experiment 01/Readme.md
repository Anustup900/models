# MASK RCNN EXPERIMENT 01 - Time 

Mask RCNN is a deep neural network aimed to solve instance segmentation problem in machine learning or computer vision. In other words, it can separate different objects in a image or a video. You give it a image, it gives you the object bounding boxes, classes and masks.
There are two stages of Mask RCNN. First, it generates proposals about the regions where there might be an object based on the input image. Second, it predicts the class of the object, refines the bounding box and generates a mask in pixel level of the object based on the first stage proposal. Both stages are connected to the backbone structure. 

So first when we are talking about how to improve the present model performance under TF Model Garden Hub , we need to understand the exsisting lags on the present state of art , So in this experiment we are running the TF code base of MASK RCNN on a custom database setuped by me . The steps involved to run this experiment and other details are mentioned below : 

## Objective 

This experiment aims to understand the training time and performance time for the present model which is there on the TF model HUB . As before going into deeper skill testing we are trying to understand the basic points of the optimisation of the performing models . So we are separately training this model on a different database and drawing inference out of it .

## Database : 

As most of the tarining and evaluations of this detction and classification based models are done on COCO database , so here we are not using that, inorder to create a differnet scenario . So the steps to create a new DB in the TF record format are as follows : 

![Image 01](https://user-images.githubusercontent.com/60361231/122776227-8a44f800-d2c8-11eb-8cf2-6663987d986f.png)

