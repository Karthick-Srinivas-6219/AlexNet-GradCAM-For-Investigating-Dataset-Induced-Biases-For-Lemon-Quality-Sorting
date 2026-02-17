# 🍋🤖 AlexNet-GradCAM-For-Investigating-Dataset-Induced-Biases-For-Lemon-Quality-Sorting 🔍
This repo demonstrates the use of GradCAM visualizations to investigate if a model used for a performance critical task such as fruit sorting (AlexNet in this case) is free from dataset induced biases especially due to background textures etc. 

# Demo 👇
<video src="demo.mp4" controls width="640"></video>
[[Link to Demo]](https://youtu.be/9cGBaeQ9GtE "Click to watch")

# Overview of the pipeline
![Alt text](gradcam_working.png)

## 🚀 Features

* **AlexNet for lemon quality classification**: **Transfer learning** for a simple Binary Classification problem involving classes **Fresh/Rotten**
* **Grad-CAM**: Applied to the **last feature extraction layer of AlexNet** generates a **heatmap** which encodes the degree of influence each region in the image had on **label assignment**.
* **Interactive Streamlit UI**: Enables the user to **upload an image** and **choose the class** for which he wishes to observe Grad-Cam Visualizations. 
---

## 📂 Project Structure

```bash
.
├── AlexNet_Finetune.ipynb/              # Loads AlexNet from PyTorch model zoo and finetunes it on the lemon quality dataset.
├── Model_Testing_GradCAMViz/          # infers from the finetuned AlexNet model and demonstrates GradCAM use on the final feature extraction layer.
├── requirements.txt      # Python dependencies.
├── demo.py            # An interactive UI based GradCAM visualizer.
├── dataset/                   # Lemon quality Bi-Class dataset
       ├── train
       ├── val
       ├── test
├── frozen_model.pth  # Finetuned AlexNet model weights.
```

## 🔧 Running Dependency

To download the **Lemon Quality dataset** click on this link [[Link to download]]([https://github.com/mohamedamine99/Facial-recognition-with-dlib](https://drive.google.com/drive/folders/1duEVC9FWB5z3H0I6rEtV33JzSudWob5Q?usp=drive_link)).

Place this file inside ```models/ ```

   ```bash
   ├── models/
       ├── dlib_face_recognition_resnet_model_v1.dat
       ├── mmod_human_face_detector.dat
       ├── shape-predictor-68-face-landmarks.dat
   ```
