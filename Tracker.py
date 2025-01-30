import cv2
import mediapipe as mp
import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog, ttk
from scipy.spatial.transform import Rotation as R
import threading
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle_3d(a, b, c):
    """Calculate the 3D joint angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    dot_product = np.dot(ab, bc)
    angle = np.arccos(dot_product / (np.linalg.norm(ab) * np.linalg.norm(bc)))
    return np.degrees(angle)

def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    csv_file = open('joint_angles.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Left Hip Flexion/Extension", "Right Hip Flexion/Extension", "Left Knee Flexion/Extension", "Right Knee Flexion/Extension", "Left Ankle Dorsiflexion/Plantarflexion", "Right Ankle Dorsiflexion/Plantarflexion", "Pelvic Tilt"])
    
    frame_count = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
                knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                
                hip_angle_l = calculate_angle_3d(shoulder_l, hip_l, knee_l)
                hip_angle_r = calculate_angle_3d(shoulder_r, hip_r, knee_r)
                knee_angle_l = calculate_angle_3d(hip_l, knee_l, ankle_l)
                knee_angle_r = calculate_angle_3d(hip_r, knee_r, ankle_r)
                ankle_angle_l = calculate_angle_3d(knee_l, ankle_l, [ankle_l[0], ankle_l[1], ankle_l[2] - 0.1])
                ankle_angle_r = calculate_angle_3d(knee_r, ankle_r, [ankle_r[0], ankle_r[1], ankle_r[2] - 0.1])
                pelvic_tilt = calculate_angle_3d(hip_l, [0, hip_l[1], hip_l[2]], hip_r)
                
                csv_writer.writerow([frame_count, hip_angle_l, hip_angle_r, knee_angle_l, knee_angle_r, ankle_angle_l, ankle_angle_r, pelvic_tilt])
                frame_count += 1
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            except:
                pass
            
            image = cv2.resize(image, (800, 600))
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            video_label.config(image=image)
            video_label.image = image
            root.update_idletasks()
    
    cap.release()
    csv_file.close()

def start_live_feed():
    threading.Thread(target=process_video, args=(0,)).start()

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        threading.Thread(target=process_video, args=(file_path,)).start()

def analyze_csv():
    file_path = filedialog.askopenfilename()
    if file_path:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        plt.figure()
        plt.plot(data[:, 0], data[:, 1:], label=["L Hip", "R Hip", "L Knee", "R Knee", "L Ankle", "R Ankle", "Pelvic Tilt"])
        plt.legend()
        plt.show()

# GUI Setup
root = tk.Tk()
root.title("Gait Analysis")
root.geometry("1000x800")
root.configure(bg='#1e1e1e')

frame = tk.Frame(root, bg='#252526')
frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

title_label = tk.Label(frame, text="Gait Analysis Tool", fg="white", bg="#252526", font=("Helvetica", 16))
title_label.pack(pady=10)

video_label = tk.Label(frame, text="No Video", fg="gray", bg="#000000", width=100, height=30)
video_label.pack(pady=10)

live_button = tk.Button(frame, text="Live Feed", padx=10, pady=5, fg="white", bg="#007acc", command=start_live_feed)
live_button.pack(pady=5)

file_button = tk.Button(frame, text="Analyze Video", padx=10, pady=5, fg="white", bg="#007acc", command=open_file)
file_button.pack(pady=5)

csv_button = tk.Button(frame, text="Analyze CSV", padx=10, pady=5, fg="white", bg="#007acc", command=analyze_csv)
csv_button.pack(pady=5)

root.mainloop()