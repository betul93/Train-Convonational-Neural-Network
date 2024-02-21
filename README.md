# Train-Convonational-Neural-Network


# Training Your DataSet with CNN
This project is a CNN (Convolutional Neural Network) model designed for emotion classification. By training the model, it tries to determine whether people are happy or unhappy from the expressions on their faces.


## Make an Algorithm

 🟩 **Image DataSet Creation**
: Three separate files were created: Training, Test, Verification

It consists of 20 photographs, including 10 happy and 10 unhappy photographs.Inside each file are the same images.

🟩 **CNN Model Creation**

🟩 **Model Training**

🟩 **Model Evaluation**

🟩 **Forecasting**


## Libraries



| Library  | Description               |
| :-------- |:------------------------- |
| **os:** | Used for interaction with the operating system.|
| **cv2:**  | OpenCV library used for image processing and computer vision.|
| **numpy:** | Used for mathematical calculations and data manipulation.|
| **tensorflow:**| Used for building and training machine learning models.|
| **matplotlib.pyplot:** | Used to display data.|
| 🗝️**Image Processing Libraries**
| **keras.preprocessing.image:** | Used for preprocessing image data.|
| 🗝️**Model Creation Libraries** | |
| **keras.models:** | Provides the Keras modelling API.|
| **keras.layers:** | Different layer types (convolutional, pooling, fully connected, etc.)|
| 🗝️**Optimisation Libraries** | |
| **keras.optimisers** | Optimisation algorithms (RMSprop, Adam, etc.)|









## Step by step Implementation

1-) Import Necessary Libraries

2-) Check Image Shape

**cv2.imread("C:/Users/CASPER/Desktop/CV_Project/training/happy/3.jpg").shape**
Reads the image at the specified path using OpenCV (though not strictly necessary in this code).
Prints its shape,**(200, 200, 3)**, indicating height, width, and number of channels (RGB).

3-)Create Image Data Generators

**train** and **validation** objects are instantiated with **rescale=1/255** to normalize pixel values between 0 and 1, aiding the training process.

- These generators will augment training and validation data during training to enhance model robustness.

4-) Load Training and Validation Data
- **train_dataset:**
 Uses **train.flow_from_directory** to create a data generator that reads images from the "training" directory.
 
 Resizes images to (200, 200), sets batch size to 3 (number of images processed together), and uses "binary" class mode (only two classes: happy/unhappy).

- **validation_dataset:**
Similar to **train_dataset**, but reads images from the "validation" directory.

5-) Define Deep Learning Model

- **model:** A sequential CNN architecture created using **tf.keras.models.Sequential**

**Structure:**
- Convolutional layers: Extract features from images.

🔹16 filters of size 3x3 with 'relu' activation in the first layer.

Subsequent layers have more filters (32, 64) to capture increasingly complex features.
- Max pooling layers: Reduce spatial dimensions and control overfitting.
- Flatten layer: Converts 2D feature maps to a 1D vector for the fully connected layers.
- Dense layers: Classify images based on extracted features.

🔹512 units with 'relu' activation for non-linearity.

🔹Output layer with 1 unit and 'sigmoid' activation for binary classification (0: happy, 1: unhappy).


6-) Compile Model
- **model.compile:**

🔹Configures the model for training.

🔹**binary_crossentropy** loss function: Suitable for binary classification problems.

🔹'RMSprop' optimizer: Gradient-based optimization algorithm (use **learning_rate=0.001** instead of **lr**).

🔹'accuracy' metric: Tracks classification accuracy during training.

7-) Train Model
- **model_fit = model.fit:**

🔹 Trains the model using the **train_dataset** for a specified number of **epochs** (10 in this case).

🔹 The **steps_per_epoch** argument (use **steps_per_epoch** instead of **step_per_epoch**) controls the number of batches processed per epoch.

🔹 The **validation_data** argument (**validation_dataset**) is used to evaluate model performance on unseen data during training.

8-) Get Class Indices (Optional)

**validation_dataset.class_indices** would print the mapping between integer class labels and actual class names (if not present, you won't see output).

9-) Predict on Test Images

- Loop through files in the "testing" directory:
For each image **i** :

🔹 Load the image, resize it to (200, 200), and convert it to a NumPy array using **img_to_array** (from the correct module).

🔹 Expand the dimensions to add a batch axis (necessary for the model).

🔹 Predict the class probabilities using **model.predict**.

🔹 Print "happy" if the predicted probability for class 0 is greater than 0.5, otherwise print "unhappy".































