This is a University Project designed for a small-scale warehouse.

Using the YOLO object detection model, the Python script used is an early iteration of an embodied learning function, allowing the program to eventually learn from its interactions.

The dataset was curated by using free license images found on several free license websites.

For this Product to work on your local machine, please do the following carefully: 

1. Make sure that the Warehouse_Script_Updated.py and the Warehouse_Model.pt have been downloaded and saved in the same folder correctly.

2. Download Anaconda Prompt. Here is a link to use: https://www.anaconda.com/download/success

3. When installed, go to your Windows search bar and search for "Anaconda Prompt."

4. When it opens, you will want to create an environment using the following command:

	conda create-- name model-env1 python=3.12 (This creates the environment where you will use the Warehouse object detection model)
	(You can choose whatever name you want; this one is just easy to follow)

5. When the environment has been created, use the following command to enter the environment:

	conda activate model-env1 

6. Once you are in the environment, cd to the folder or location where you extracted the zip file.

	Example: cd C:\Users\Name\Desktop\Folder

	Paste that command into the prompt (your command will be different; this is just an example). 

7. Install the Ultralytics Library:
	
	pip install ultralytics (This may take a while)

8. Upgrade the PyTorch version to work with your GPU:

	pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

9. You've done the hard part, now time for the fun bit!
	Use the following command to use the script (Make sure that if you aren't using a device with a built-in webcam, you may need to connect one to a USB port)

	python Warehouse_Script.py --model Warehouse_Model.pt --source usb0 --resolution 1280x720 

10. Here is a list of arguments you can use to play around with: 

	--model (Chooses the trained and validated model to use)

	--source (You can choose an image, video, or camera to play around with)
		(Disclaimer: if you want to use an image or video file, please specify the type of file as well as the name, .jpg or .mov, for example)

	--thresh (You can decide what threshold of confidence you would like the program to see example, 0.4)

	--resolution (Changes the size of the screen that the output will be displayed on)

	--screenshot (Sets the screenshot interval by its frames, for example, 10 (Will take screenshots every 10 frames)
			(Disclaimer, if you set the interval too low, for example, 1, the screenshots of each frame will be saved to a screenshots folder, this did resulted in my PC crashing)

12. Press Q to stop

13. Look at the results in the screenshot folder and the detection log that the script has created for you.



