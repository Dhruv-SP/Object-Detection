# **Object-Detection**

The python programm to scan the images, detect the object in the image, perform random process  `i.e.. rotate, gray, sharpen, mirror, smooth` on the image and then relocate the image based on the contant of the image.


The whole project is divided into ***4 parts***. 



### part 1: Libraries
---

Several libraries are used which reduced the efforts by some greater margine. Libraries as Numpy, matplotlib, imageai, csv, PIL where used.

`Numpy` is used for basic mathematical computations.

`imageai` as our object detector model.

`PIL` for image operations.

`csv and os` for file and folder operations.


### part 2: creating csv's
---

Here we create 3 different CSV's to store the separete details about the code.

First csv is about all the types that we will encounter and with their code as `t2`, then we have the subtype csv and their code as `t3`. and at last we create one for preprocessed again with its code as `t4`.


### part 3: Detect, Preprocess, Relocate.
---

This is the main part of the whole project, in this part we'll go through the code in brief explaining the code.

Innitially we are creating the dictionaries for each preprocessing with keys as all the subtypes and value set to '0', the main idea is to keep track of number of operation happened on what type of image.

Then we are creating CSV as `t1`, this csv will contain all the dedtails of all the images that will be scanned and what objects are detected in them.

Next whole segment are dedclared procedured for each preprocessing operation, here details of preprocessing are appanded to the `dest` variable and relocated and saved at the appropriate folder based on the content of the image.

Now we are upto the model, Here we get the object of detecting model, and as every model requires to be trained before being used for prediction, this also needs to be trained, but there is a way to get around this lengthy and dedicating process, 
that is by loading the pretrained model to our object. For that first we have to set the type of model that is going to be loaded which is retinanet by using `setModelTypeAsRetinaNet()`. now we set the path of pretrained model using `setModelPath` and passing the path of pretrained model to it.
Finally loading it using `loadModel()`

Great, now you have the model ready to be used for prediction.

> But before getting to the prediction, one thing needs to be discussed, the model used `detectObjectsFromImage()` which will require you to pass the path where it will output the image with boxes drown aroung detected object,
and we dont want that, so we will passs a dummy dump path where the model will output the images

Time to run the prediction on the images:

Iterate over every image by getting their path by using `global.global` in for loop, also extract the name of the image file and appand to the dump path
