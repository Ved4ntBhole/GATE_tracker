import cv2
import mediapipe as mp
import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog, ttk
from scipy.spatial.transform import Rotation as R
import threading
from PIL import Image, ImageTk
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Fixed canvas size
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600

def calculate_angle_3d(a, b, c):
    """Calculate the 3D joint angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    dot_product = np.dot(ab, bc)
    angle = np.arccos(dot_product / (np.linalg.norm(ab) * np.linalg.norm(bc)))
    return np.degrees(angle)

def fit_to_canvas(image):
    """Resize and pad the image to fit the fixed canvas size while maintaining aspect ratio."""
    h, w, _ = image.shape
    scale_w = CANVAS_WIDTH / w
    scale_h = CANVAS_HEIGHT / h
    scale = min(scale_w, scale_h)

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    # Center the image with black borders only on one dimension
    top = (CANVAS_HEIGHT - new_h) // 2 if scale_w < scale_h else 0
    bottom = CANVAS_HEIGHT - new_h - top if scale_w < scale_h else 0
    left = (CANVAS_WIDTH - new_w) // 2 if scale_w >= scale_h else 0
    right = CANVAS_WIDTH - new_w - left if scale_w >= scale_h else 0

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
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

            image = fit_to_canvas(image)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            video_label.config(image=image, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
            video_label.image = image
            root.update_idletasks()

    cap.release()
    csv_file.close()
    plot_graphs('joint_angles.csv')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_graphs(csv_file):
    data = pd.read_csv(csv_file)
    angles = [
        "Left Hip Flexion/Extension", "Right Hip Flexion/Extension",
        "Left Knee Flexion/Extension", "Right Knee Flexion/Extension",
        "Left Ankle Dorsiflexion/Plantarflexion", "Right Ankle Dorsiflexion/Plantarflexion",
        "Pelvic Tilt"
    ]

    # Define a color map for consistency
    colors = {
        "Left Hip Flexion/Extension": "red",
        "Right Hip Flexion/Extension": "blue",
        "Left Knee Flexion/Extension": "green",
        "Right Knee Flexion/Extension": "orange",
        "Left Ankle Dorsiflexion/Plantarflexion": "purple",
        "Right Ankle Dorsiflexion/Plantarflexion": "brown",
        "Pelvic Tilt": "cyan"
    }

    fig, axes = plt.subplots(len(angles) + 1, 1, figsize=(10, 14), sharex=True)

    # Set the figure background color
    fig.patch.set_facecolor('#1f1f1f')

    # Loop through each angle and plot
    for i, angle in enumerate(angles):
        axes[i].plot(data["Frame"], data[angle], label=angle, color=colors[angle])
        axes[i].grid(color='gray', linestyle='--', linewidth=0.5)
        axes[i].set_facecolor('#1f1f1f')
        axes[i].tick_params(colors='white')
        axes[i].spines['bottom'].set_color('white')
        axes[i].spines['left'].set_color('white')
        axes[i].xaxis.label.set_color('white')
        axes[i].yaxis.label.set_color('white')

    # Combined plot
    for angle in angles:
        axes[-1].plot(data["Frame"], data[angle], label=angle, color=colors[angle])

    # Set the combined plot background color
    axes[-1].set_facecolor('#1f1f1f')
    axes[-1].xaxis.label.set_color('white')
    axes[-1].yaxis.label.set_color('white')

    # Set the combined plot labels and borders to white
    axes[-1].tick_params(axis='x', colors='white')
    axes[-1].tick_params(axis='y', colors='white')
    axes[-1].spines['bottom'].set_color('white')
    axes[-1].spines['left'].set_color('white')

    # Add a single legend for the combined plot
    axes[-1].legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.2), 
        ncol=4, 
        fontsize='medium', 
        facecolor='#1f1f1f', 
        edgecolor='white',
        labelcolor='white'  # Set the legend label color to white
    )
    axes[-1].grid(color='gray', linestyle='--', linewidth=0.5)

    # Set the figure title
    fig.suptitle("Joint Angles over Time", color='white')

    # Embed in Tkinter
    global canvas  # Keep reference to prevent garbage collection
    for widget in data_tab.winfo_children():
        widget.destroy()  # Clear previous plot

    canvas = FigureCanvasTkAgg(fig, master=data_tab)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()

def start_live_feed():
    threading.Thread(target=process_video, args=(1,)).start()

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        threading.Thread(target=process_video, args=(file_path,)).start()

def analyze_csv():
    file_path = filedialog.askopenfilename()
    if file_path:
        plot_graphs(file_path)

# GUI Setup
root = tk.Tk()
root.title("Gait Analysis")
root.geometry("1000x800")
root.configure(bg="#1e1e1e")
root.attributes('-fullscreen', True)

style = ttk.Style()
style.configure("TNotebook", background="#1e1e1e")
style.configure("TFrame", background="#252526")

notebook = ttk.Notebook(root, style="TNotebook")
notebook.pack(fill=tk.BOTH, expand=True)

# Video and Live Feed Tab
video_tab = ttk.Frame(notebook, style="TFrame")
notebook.add(video_tab, text="Video Analysis")

frame = tk.Frame(video_tab, bg='#252526')
frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

title_label = tk.Label(frame, text="Gait Analysis Tool", fg="white", bg="#252526", font=("Helvetica", 16))
title_label.pack(pady=10)

video_label = tk.Label(frame, text="No Video", fg="gray", bg="#000000", width=CANVAS_WIDTH // 10, height=CANVAS_HEIGHT // 20)
video_label.pack(pady=10)

live_button = tk.Button(frame, text="Live Feed", padx=10, pady=5, fg="white", bg="#007acc", command=start_live_feed)
live_button.pack(pady=5)

file_button = tk.Button(frame, text="Analyze Video", padx=10, pady=5, fg="white", bg="#007acc", command=open_file)
file_button.pack(pady=5)

# Data Analysis Tab
data_tab = ttk.Frame(notebook, style="TFrame")
notebook.add(data_tab, text="Data Analysis")

csv_button = tk.Button(data_tab, text="Upload CSV", padx=10, pady=5, fg="white", bg="#007acc", command=analyze_csv)
csv_button.pack(pady=10)

root.mainloop()
