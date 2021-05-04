# **Object-Detection**

The python program scans the images, detects the object in the image, performs a random preprocessing  `i.e.. rotate, gray, sharpen, mirror, smooth` on the image and then relocate the image based on the content of the image.


The whole project is divided into ***4 parts***.



### part 1: Libraries
---

Several libraries are used which reduced the efforts by some greater margin. Libraries such as Numpy, matplotlib, imageai, csv, PIL were used.

`Numpy` is used for basic mathematical computations.

`imageai` as our object detector model.

`PIL` for image operations.

`csv and os` for file and folder operations.


### part 2: creating csv's
---

Here we create 3 different CSV to store the separate details about the code.

First csv is about all the types that we will encounter and with their code as `t2`, then we have the subtype csv and their code as `t3`. and at last we create one for preprocessed again with its code as `t4`.


### part 3: Detect, Preprocess, Relocate.
---

This is the main part of the whole project, in this part we'll go through the code in brief explaining the code.

Initially we are creating the dictionaries for each preprocessing with keys as all the subtypes and values set to '0', the main idea is to keep track of the number of operations happening on what type of image.

Then we are creating CSV as `t1`, this csv will contain all the details of all the images that will be scanned and what objects are detected in them.

Next whole segment are declared procedure for each preprocessing operation, here details of preprocessing are appended to the `dest` variable and relocated and saved at the appropriate folder based on the content of the image.

Now we are upto the model, Here we get the object of detecting model, and as every model requires to be trained before being used for prediction, this also needs to be trained, but there is a way to get around this lengthy and dedicating process,
that is by loading the pretrained model to our object. For that first we have to set the type of model that is going to be loaded which is retinanet by using `setModelTypeAsRetinaNet()`. now we set the path of pretrained model using `setModelPath` and passing the path of pretrained model to it.
Finally loading it using `loadModel()`

Use the following link to download the ![Pre-trained model](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5)

Great, now you have the model ready to be used for prediction.

> But before getting to the prediction, one thing needs to be discussed, the model used `detectObjectsFromImage()` which will require you to pass the path where it will output the image with boxes drown around detected object,
and we don't want that, so we will pass a dummy dump path where the model will output the images

Time to run the prediction on the images:

Iterate over every image by getting their path by using `global.global` in for loop, also extract the name of the image file and append to the dump path.

Finally we run the model, there are 2 parameters to mention in the prediction method `detectObjectsFromImage()`which are: first, the path of image to be used for prediction and the other parameter to store the output image. The prediction method will return a list of dictionaries which has 3 keys as name, percentage_probability, box_points. We will iterate through each detected objects dictionary and will store the name of the object which has the highest percentage_probability. After getting the object name, we attach the appropriate code to the type variable `tname` and append the directory path for relocation.

Next up we generate a random number between 0 to 4 which will determine the preprocessing operation to perform and append the metadata to variables and save the image to its appropriate folder.


### part 4 A: Enter the processing details into a CSV
---
This part is actually implemented at the end of part 3 as we are iterating through the images already so no need to make another iteration and make program slower, so instead at the end of the loop we make an entry in the `t1.csv` about the sequence no, filename, destination, its type, what preprocessing is done etc.. of the image.

### part 4 B: Generate and store statistics based on preprocessing operations performed
---
Here we'll use the dictionaries that we made in the very beginning that specifically store the count of specified operation on a certain type of image. we'll iterate through them and save as `s1.csv`.
