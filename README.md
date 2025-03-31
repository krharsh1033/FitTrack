# AI Fitness Trainer
Utilized Google's MediaPipe pose detection model to extract joint positions from exercise videos uploaded via React Frontend. Implemented real-time chatbot feedback with quantized TinyLlama 1.1B LLM model from Hugging Face Transformers. Other features include Django user authentification, asynchronous pose detection display and chatbot feedback using asyncio library, peak detection algorithm from scipy for counting reps from pose landmarks, and matplotlib plots for displaying reps with concentric/eccentric time spent for each rep for measuring time under tension. Please note that not 

## Video Demo
[![Watch the video](https://img.youtube.com/vi/amvVTQqxZ8/0.jpg)](https://www.youtube.com/watch?v=amvVT-QqxZ8)

## How to download:

### 1. LLM Chatbot Feedback
I incorporated Hugging Face Transformers popular text generation models including Deepseek-R1 and TinyLlama to provide the user with recommendations for improving their exercise form based off their uploaded video. I accomplished this by creating a prompt to the chatbot (in ChatML format for TinyLlama) that includes specific directions as to what it should be looking to suggest improvements on. The prompt

### Model Selection
I first started trying to implement DeepSeek-R1 7B for chatbot feedback due to the recent publications and promising potential for performing fast inference on resource constrained devices. However, after deploying the model from Hugging Face Transformers, I realized the inference time of 6039s provided an unreasonable inference time for implementing real-time or near real-time feedback to the user making the application non user-friendly. After realizing this I needed to find a model with less parameters that can be used to provide much better inference time for giving the user a positive experience. After some research into similar chat generation models, I decided to try TinyLlama 1.1B model with significantly less parameters (7B ~ 1B) to see if I could get near real-time feedback from my chatbot model while still maintaining good responses. My initial implementation of the base TinyLlama resulted in much better performance than my DeepSeek model, however, after further examination I realized there was still room for some optimization. I implemented both base models using AutoModelForCausallM from transformers library created by Hugging Face Transformers. 

### Base DeepSeek-R1 7B inference time
![Base DeepSeek-R1 7B inference time] ()

### Base TinyLlama 1.1B inference time
![Base TinyLlama 1.1B inference time] ()

### 2. Quantization
I applied 8 bit quantization to my TinyLlama 1.1B model for even faster inference speeds to ensure a postive user experience. I utlized llama_cpp to apply this 8 bit quantization to convert my training parameters from 32 bit floating point numbers to 8 bit integers. This drastically reduced the memory resource usage of my model which is important for enabling other users on less powerful systems to be able to execute the same program without any issues. Decreasing the overall size of the model from switching from 32 bit floating point numbers to 8 bit integers also decreases the overall complexity of the model ielding faster inference times. 

### Base TinyLlama 1.1B memory usage and inference time



### Quantized TinyLlama 1.1B memory usage and inference time
![3F17062C-5F1D-49A7-9A84-4F9955664026_4_5005_c](https://github.com/user-attachments/assets/853b04be-842c-443f-8c3c-8bdbd1282e29))

### 3. MediaPipe
Google's MediaPipe is a pose detection algorithm that extracts joint positions from images. I utilized this pose detection algorithm in combination with OpenCV's VideoCapture class to extract each frame from the video and append each joint position to a joint_positions list which would then be used to help calculate force_vectors, joint_angles, and range of motion for a particular exercise. 

### 4. Django Backend (Code found in /fitness_backend)

### Models

The Django backend defines many different models in models.py: UserManager, User, Exercise, DeepseekChatbot, LlamaChatbot, QuantizedLlamaChatbot, and BlokeLlamaChatbot. The User model extends Django's AbstractBaseUser to manage user authentication and profiles, including fields like username, email, body weight, and gender. It also includes custom user management via the UserManager. The Exercise model handles exercise video uploads, analysis, and storage. It stores input and output videos, analysis results like force vectors, angles, rep timings, and chatbot feedback. This model also implements asynchronous video processing and analysis, leveraging Google Cloud Storage for file handling and integrating with external libraries for pose estimation and chatbot interactions. The DeepseekChatbot class is not currently being used in my implementation, but it was my original attempt in creating a real-time chatbot feedback for exercise improvement recommendations. The LlamaChatbot class is also not currently being used in my implementation, but it is my base TinyLlama class that successfully generates workout improvement advice. 

### Views

The views.py file defines API views for user authentication, registration, and exercise video processing. LoginView handles user authentication, utilizing Django's authenticate function and Token generation for secure access. NewUserView creates new user accounts, employing User.objects.create_user which automatically hashes passwords for security. FileUploadView processes uploaded exercise videos, storing them in Google Cloud Storage and triggering asynchronous video analysis. It employs TokenAuthentication and IsAuthenticated to secure the endpoint, ensuring only authenticated users can upload files. AnalysisStatusView provides the status of the video analysis. The file makes extensive use of Django Rest Framework's APIView, Response, and Serializer for API development, and transaction.atomic() for database transaction management.

### Serializers

The serializers.py file defines serializers for the Exercise and User models, leveraging Django REST Framework's ModelSerializer. ExerciseSerializer converts Exercise model instances into JSON format, including all fields, for API responses. Similarly, UserSerializer serializes User model instances, encompassing all user attributes. These serializers facilitate data exchange between the Django backend and the frontend, streamlining the process of converting complex model data into easily consumable JSON.

### 5. React Frontend (Code found in fitness-frontend)

### App
This React component sets up routing for the application, defining paths for user login, new user registration, and the main fitness application interface.

### FitnessApp
This React component, FitnessApp, provides the frontend interface for uploading and analyzing workout videos. It manages user authentication via JWT tokens stored in localStorage, redirecting unauthenticated users to the login page. The component allows users to select an exercise, input the weight lifted, and upload an MP4 video. Upon submission, it sends the video to the Django backend for processing, displaying a loading indicator during upload and analysis. It then polls the backend for analysis status, rendering the processed video and a progress image once available. After the backend's chatbot completes its analysis, the response is displayed to the user. The component uses react-player for video playback and lucide-react for loading spinners, and leverages React hooks like useState, useEffect, and useRef for state management and DOM manipulation. It also incorporates react-router-dom for navigation.

### NewUser
This React component, NewUser, handles user registration. It provides a form for users to input their username, email, password (with password validation), name, weight, and gender. Upon submission, it sends this data to the Django backend to create a new user account. It stores the returned authentication token and user ID in localStorage and redirects the user to the FitnessApp component. It uses useState and useEffect hooks for state management and navigation, and includes password validation logic to ensure secure user credentials.

### UserLogin
This React component, UserLogin, handles user authentication. It provides a login form for users to enter their username and password, with password validation. It communicates with the Django backend to authenticate users, storing the returned authentication token and user ID in localStorage upon successful login. It also implements a login attempt counter and lockout mechanism to prevent brute-force attacks.


### Resources

Kaggle Dataset: https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video

DeepSeek-R1 7B: https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat

TinyLlama 1.1B: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
