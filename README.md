# PythonProjectTF
Python project to demonstrate the use of Image processing, Machine Learning in getting the sum of numbers that are input by giving hand gesture
Instructions to follow:
1) First clone or download this Project Repo (PythonProjectTF)
2) Open the terminal or cmd window in the directory where you have GestureRecognition.py
3.a) Type "python3 GestureRecognition.py" -> for Python 3.x in Linux or "python GestureRecognition.py" for Python 2.x in Linux
3.b) Type "python GestureRecognition.py" -> for Python 2.x, 3.x in Windows
4) Now a window named 'Paint' will open
5) Type 'b' meaning (Background Capture)
6) Two more windows will be open
7) Use your laptop or desktop camera to align the blue rectangle in the "Tracking" window to be a static background. Then press 'b'
8) This will make the background to be taken. (You can use 'b', any number of times)
9) Now with you hand in the blue box, open only two fingers, symbolic to show victory sign. (Index and middle finger)
10) This will make the screen active and the output can be seen to be getting printed on the screen, in the Paint window.
11) After you draw a number successfully, just quickly, close your fingers to stop recording, and press 's' meaning (Save)
12) Now the image is created in /img folder.
13) You will get the acknowledgement regarding the number that is predicted, as soon as you type 's'
14) After you are done giving the input, be in any one of the three screens ("Tracking", "Paint", "contour output"), and press "ESC" key
15) If you had given atleast one input, then the sum is displayed on another window, for 2 seconds. Otherwise the program just quits.



P.S : I have not tried in Python 2.x, if it does not work, then just email me on "kirancnayak@tuta.io"
    : The program won't run for some laptops with old processors and GPUs. Even mine was working only sometimes
    : If you are not getting the output, then go to predict_interface_usage.py and change the boolean value from False to True, and run. Later revert and keep. The reason to do this is that it will train first, then it will just use the trained result.
    : I have not written the code completely. For the original project go to https://github.com/opensourcesblog/tensorflow-mnist (Credit to the person who wrote it)
