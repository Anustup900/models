# MASK RCNN EXPERIMENT 01 - Time 

Mask RCNN is a deep neural network aimed to solve instance segmentation problem in machine learning or computer vision. In other words, it can separate different objects in a image or a video. You give it a image, it gives you the object bounding boxes, classes and masks.
There are two stages of Mask RCNN. First, it generates proposals about the regions where there might be an object based on the input image. Second, it predicts the class of the object, refines the bounding box and generates a mask in pixel level of the object based on the first stage proposal. Both stages are connected to the backbone structure. 

So first when we are talking about how to improve the present model performance under TF Model Garden Hub , we need to understand the exsisting lags on the present state of art , So in this experiment we are running the TF code base of MASK RCNN on a custom database setuped by me . The steps involved to run this experiment and other details are mentioned below : 

## Objective 

This experiment aims to understand the training time and performance time for the present model which is there on the TF model HUB . As before going into deeper skill testing we are trying to understand the basic points of the optimisation of the performing models . So we are separately training this model on a different database and drawing inference out of it .

### Project Structure : 

![Image 01](https://user-images.githubusercontent.com/60361231/122776227-8a44f800-d2c8-11eb-8cf2-6663987d986f.png)

## Database : 

As most of the tarining and evaluations of this detction and classification based models are done on COCO database , so here we are not using that, inorder to create a differnet scenario . So the steps to create a new DB in the TF record format are as follows :

The Project’s repository contains train and test images for the detection of a blue Bluetooth speaker and a mug. Pick up objects you want to detect and take some pics of it with varying backgrounds, angles, and distances. Training images used in this sample project are shown below:

![1_nyHiArgwgJPb8vXB6Im7kQ](https://user-images.githubusercontent.com/60361231/122776584-ddb74600-d2c8-11eb-8c66-5c1ed24164a2.jpeg)
![1_PT_c7sYksFIMQ-s-_GhyWA](https://user-images.githubusercontent.com/60361231/122776598-df810980-d2c8-11eb-9258-5c26955b98f2.jpeg)

Once images captured , transfer it to your PC and resize it to a smaller size (given images have the size of 512 x 384) so that your training will go smoothly without running out of memory. Now rename (for better referencing later) and divide your captured images into two chunks, one chunk for training(80%) and another for testing(20%). Finally, move training images into the dataset/train_images folder and testing images into the dataset/test_images folder.

# Environment Setup :

So first we will get into the Tensorflow Model Library : 

```
git clone https://github.com/tensorflow/models.git
```
Once we have cloned this repository, change our present working directory to models/research/ and add it to our python path. If we want to add it permanently then we will have to make the changes in our .bashrc file or could add it temporarily for current session using the following command:

```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
Then we need to also run following command in order to get rid of the string_int_label_map_pb2 issue (https://github.com/tensorflow/models/issues/1595)
```
protoc object_detection/protos/*.proto --python_out=.
```
Now your Environment is all set to use TensorFlow object detection API
 
## Convert the data to Tensorflow record format

In order to use Tensorflow API, you need to feed the data in the Tensorflow record format. I have modified the script create_pet_tf_record.py given by Tensorflow and placed the same in the project repository inside the folder named as supporting_scripts. The name of the modified file is given as create_mask_rcnn_tf_record.py. All you need to do is to take this script and place it in the models/research/object_detection/dataset_tools.


Create_mask_rcnn_tf_record.py is modified in such a way that given a mask image, it should found bounding box around objects on it owns and hence you don’t need to spend extra time annotating bounding boxes but it produces wrong output if mask image has multiple objects of the same class because then it will not be able to find bounding box for each object of the same class rather it will take a bounding box encompassing all objects of that class.


If you have multiple objects of the same class in some images then use labelImg library to generate XML files with bounding boxes and then place all the XML files generated from the labelImg under dataset/train_bboxes folder. If you intend to use this method then you will have to set bboxes_provided flag as True while running create_mask_rcnn_tf_record.py otherwise set it to False. It's been forced to provide bboxes_provided flag in order to avoid the users from making mistakes.
To download the labelImg library along with its dependencies go to THIS LINK. Once you have the labelImg library downloaded on your PC, run lableImg.py. Select train_images directory by clicking on Open Dir and change the save directory to dataset/train_bboxes by clicking on Change Save Dir. Now all you need to do is to draw rectangles around the object you are planning to detect. You will need to click on Create RectBox and then you will get the cursor to label the objects. After drawing rectangles around objects, give the name for the label and save it so that Annotations will get saved as the .xml file in dataset/train_bboxes folder.


After doing the above, one last thing is still remaining before we get our Tensorflow record file. You need to create a file for the label map, in the project repository, it’s given as label.pbtxt under the dataset subfolder. In the label map, you need to provides one item for each class. Each item holds the following information: class id, class name and the pixel value of the color assigned to the class in masks. You need to notice in the given sample label.pbtxt that the last three letters of the string assigned as name of the class will be considered as the pixel value. You could find the mask pixel value by opening the mask image as a grayscale image and then check pixel value in the area where your object is. A file with name Check_pixel_values.ipynb is given under subfolder named as supporting_script to help you with this task.
Now it time to create a tfrecord file. From models/research as present working directory run the following command to create Tensorflow record):

```
!python object_detection/model_main_tf2.py \
--model_dir=/content/drive/MyDrive/Maskrcnn/CP \
--pipeline_config_path=/content/drive/MyDrive/Maskrcnn/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.config
```
## Training 

Now that we have data in the right format to feed, we could go ahead with training our model. The first thing you need to do is to select the pre-trained model you would like to use. You could check and download a pre-trained model from Tensorflow detection model zoo Github page. Once downloaded, extract all file to the folder you had created for saving the pre-trained model files. Next you need to copy models/research/object_detection/sample/configs/<your_model_name.config> and paste it in the project repo. You need to configure 5 paths in this file. Just open this file and search for PATH_TO_BE_CONFIGURED and replace it with the required path. I used pre-trained mask RCNN which is trained with inception V2 as feature extractor and I have added modified config file (along with PATH_TO_BE_CONFIGURED as the comment above lines which has been modified) for same in this repo. You could also play with other hyperparameters if you want. Now you are all set to train your model, just run the following command with models/research as present working directory

```
python object_detection/legacy/train.py --train_dir=<path_to_the folder_for_saving_checkpoints> --pipeline_config_path=<path_to_config_file>
```

```
python object_detection/legacy/train.py --train_dir=/Users/vijendra1125/Documents/tensorflow/object_detection/multi_object_mask/CP --pipeline_config_path=/Users/vijendra1125/Documents/tensorflow/object_detection/multi_object_mask/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.config
```
Let it train till loss will be below 0.2 or even lesser. once you see that loss is as low as you want then give keyboard interrupt. Checkpoints will be saved in CP folder. Now its time to generate inference graph from saved checkpoints :

```
python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=<path_to_config_file> --trained_checkpoint_prefix=<path to saved checkpoint> --output_directory=<path_to_the_folder_for_saving_inference_graph>
```

