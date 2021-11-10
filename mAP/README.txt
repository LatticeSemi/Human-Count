########Files and folder details#########

detection-results -> This folders contains predicted bounding boxes on test images in .txt file format.
		     Each test image will have their own .txt file
		     Text file will contain coordinates of boxes, class label and confidence score in a format given below.
		     
                     		<class_label> <confidence> <xmin> <ymin> <xmax> <ymax>

		     Eg.  Person 0.871781 0 13 174 244

ground-truth -> This folders contains ground-truth bounding boxes on test images in .txt file format.
		Each test image will have their own .txt file
		Text file will contain coordinates of boxes and class label in a format given below.
		     
                     		<class_label> <_> <_> <_><xmin> <ymin> <xmax> <ymax>

	        Eg. Person 0.0 0.0 0.0 2 10 173 238


main.py -> This file contain the code for calculating mAP.

output -> This folder contains folders/files which will be output of main.py file.

detections_one_by_one - This folder contains the test images on which ground truth and predicted boxes are drawn along with label but one box per image i.e. if image contains two boxes then there will be two separate images for each boxes.

output.txt -> This file contains the mAP score and Precision of each bounding boxes and other details needed for performance metrics

README.txt -> It contains details of folder and details of running the code.



######to run the code######
Run the following command on terminal.
$ python main.py --input_images <Images_Path> --ground_truth  <Ground_Truth_Path> --predictions <Prediction_Path>
