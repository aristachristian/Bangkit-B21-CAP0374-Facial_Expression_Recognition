# Bangkit 2021 Capstone Project
## Link to Other members Github Page

Link to Speech Emotion Recognition : https://github.com/joseph-k-git/Bangkit-B21-CAP0374-Speech_Emotion_Recognition

Link to Android Code : https://github.com/aldolvk/Online-Interview

## Steps To Replicate The Project

Note : If you don't want to train the model on your own, feel free to skip to step 2.

1. Open "Model Training.ipynb" file and train the model using either the Original MMA Facial Expression Dataset or the Cleaned dataset (You can find the link to the dataset in "Other Reference" Section).
2. Save The model in the same directory as "Model Training.ipynb", and open "Model Testing using front camera.ipynb" file. You can test your trained model (or the model provided in "Other Reference Section") using image from front camera. Do not open this file in Google Collab, because the program will open your front camera.
3. Open "Model Deployed on Web.py" and run it using the command`python Model\ Deployed\ on\ Web.py`. Then, to open the app, go to your browser and type `localhost:5000`. This will display the web app.
4. To make the web app public, use command `ngrok http 5000`.

## Other References :

Link to Face-Detection-OpenCV : https://github.com/informramiz/Face-Detection-OpenCV

Link to Cleaned Dataset : https://drive.google.com/file/d/1jYrm4fk9gfo4o9Nkq2dr0g8g5gV33FRI/view?usp=sharing

Link to Original Dataset : https://www.kaggle.com/mahmoudima/mma-facial-expression

Link to our pre-trained Facial Expression Recognition Model : https://drive.google.com/file/d/1gSq9ajbBH_H2fKLLYqbTSij4atmWHKmT/view?usp=sharing

Link to Web app template and code : https://github.com/krishnaik06/Deployment-Deep-Learning-Model
