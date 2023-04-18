# Neural-net-from-scratch

![image](https://user-images.githubusercontent.com/93164951/229936294-8d106dc2-3cac-48f5-8b07-7991d111f2ec.png)

## Description
This project is a visual playground for the logistic regression algorithm. It allows you to experiment with the parameters of the neural network and the learning algorithm.<br />
Here I used the Python programming language, NumPy and Matplotlib libraries, and the PyQt5 graphics library.<br />
I did this project for educational purposes. I wanted to better understand how neural nets work, how they are trained, and how backpropagation works. So I implemented everything from scratch :)

## How to Install and Run the Project
First, you need to download the repository files. If you have Python installed (version 3.10 or higher) then all you need to do is to open the command prompt in the directory with all the files and run the following commands:<br />
1. python -m venv venv<br />
2. venv\Scripts\activate.bat<br />
3. pip install -r requirements.txt<br />
4. main.py<br />

## How to Use the Project
1. Сlick the left mouse button to put a green dot, right - red dot.<br />
2. Click the button "TRAIN" to fit the model.<br />
3. Button "Plot cost" fits the model and shows the graph of the cost function.<br />
4. Button "Tune learn-rate" does the same but for different learning rates.<br />

<img src="https://user-images.githubusercontent.com/93164951/229954990-3f00131f-0511-475d-9dab-0aea3b16d8f1.png" alt="Example Image" width="650" height="390">

5. In the "Layer sizes" text field you can specify the number of neurons in each layer of the neural net. By default, it is "2, 5, 1".<br /> 
**!IMPORTANT!** The first and last layers should always be of size 2 and 1 respectively.<br />
6. In the combobox below the "Layer sizes" you can specify the activation function for the hidden layers.<br />
7. Button "^ Compare ^" below fits the models with different activation functions and shows their graphs of the cost function.<br />
8. The green combobox and the button "^ Compare ^" below let you choose between different learning algorithms and compare their efficiency.<br />

<img src="https://user-images.githubusercontent.com/93164951/229955121-7614f730-ebbf-4021-b524-f15c0eb25a63.png" alt="Example Image" width="650" height="390">
