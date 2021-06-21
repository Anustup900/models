# Error Logs Handled while model training : 

While training Custom Mask RCNN in own database sevral types of errors can arise ( that I faced during training it ) , the detailed documentation is provided here ( it can work as log report ) 

1.) TF Slim Error while upgrading TF 01 to TF 02 
```
ModuleNotFoundError: No module named 'tf_slim'
```
same thing can occur for : No offcial module found for TF , Simply add 
```
!pip install tf_slim
!pip install tf-models-official
```
2.) While upgrading from TF 01 to TF 02 : 
```
ssd_inception_v2 is not supported. See `model_builder.py` for features extractors compatible with different versions of Tensorflow
```
In most of the available solutions , since they are built on Tfx01 , so they are using previous builder files for tensorflow , go this link : https://github.com/tensorflow/models/tree/master/research/object_detection
and use TF : object_detection/model_main_tf2.py , it will work

3.) Anather error you can encounter regarding the model configuration file , same thing you have to do , going to the above mentioned link , you have to update the configuration based on the TF 02 supported documents.

4.) ![Error 01](https://user-images.githubusercontent.com/60361231/122785491-0e9b7900-d2d1-11eb-90cd-ee911616638b.PNG)

Burning out of Colab , by showing this statement while training Custom Mask RCNN
