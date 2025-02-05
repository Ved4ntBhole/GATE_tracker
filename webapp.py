import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

# Mediapipe initialization
mp_pose = mp.solutions.pose


def calculate_angle_3d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    dot_product = np.dot(ab, bc)
    angle = np.arccos(dot_product / (np.linalg.norm(ab) * np.linalg.norm(bc)))
    return np.degrees(angle)


def process_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    with open(output_csv, 'w', newline='') as csv_file:
        writer = pd.ExcelWriter(csv_file)
        writer.writerow(["Frame", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Pelvic Tilt"])
        frame_count = 0

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                    knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
                    knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                    ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                    ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]

                    angles = [
                        calculate_angle_3d(shoulder_l, hip_l, knee_l),
                        calculate_angle_3d(shoulder_r, hip_r, knee_r),
                        calculate_angle_3d(hip_l, knee_l, ankle_l),
                        calculate_angle_3d(hip_r, knee_r, ankle_r),
                        calculate_angle_3d(knee_l, ankle_l, [ankle_l[0], ankle_l[1], ankle_l[2] - 0.1]),
                        calculate_angle_3d(knee_r, ankle_r, [ankle_r[0], ankle_r[1], ankle_r[2] - 0.1]),
                        calculate_angle_3d(hip_l, [0, hip_l[1], hip_l[2]], hip_r),
                    ]

                    writer.writerow([frame_count, *angles])
                    frame_count += 1

    cap.release()


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Joint Angle Tracker", className="text-center mt-4 mb-4"))),
    dbc.Tabs([
        dbc.Tab(label="Analyze Video", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Video File:"),
                    dbc.Button("Browse", id="select-video", color="primary", className="mb-2"),
                    html.Div(id="video-path", children="No file selected"),
                    html.Label("Select Output CSV File:"),
                    dbc.Button("Browse", id="select-csv", color="primary", className="mb-2"),
                    html.Div(id="csv-path", children="No file selected"),
                    dbc.Button("Process Video", id="process-button", color="success", className="mt-2")
                ], width=6)
            ]),
            dbc.Row(dbc.Col(html.Div(id="process-status", className="mt-4")))
        ]),
        dbc.Tab(label="Visualize Data", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Upload CSV:"),
                    dcc.Upload(id="upload-csv", children=dbc.Button("Upload CSV", color="primary"), className="mb-4"),
                    html.Div(id="upload-status")
                ])
            ]),
            dbc.Row(dbc.Col(dcc.Graph(id="visualization")))
        ])
    ])
])


@app.callback(
    Output("video-path", "children"),
    Input("select-video", "n_clicks")
)
def select_video(n_clicks):
    if n_clicks:
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(title="Select Video File")
        return video_path if video_path else "No file selected"
    return "No file selected"


@app.callback(
    Output("csv-path", "children"),
    Input("select-csv", "n_clicks")
)
def select_csv(n_clicks):
    if n_clicks:
        root = tk.Tk()
        root.withdraw()
        csv_path = filedialog.asksaveasfilename(title="Save CSV File", defaultextension=".csv")
        return csv_path if csv_path else "No file selected"
    return "No file selected"


@app.callback(
    Output("process-status", "children"),
    Input("process-button", "n_clicks"),
    State("video-path", "children"),
    State("csv-path", "children")
)
def process_video_callback(n_clicks, video_path, csv_path):
    if n_clicks and video_path != "No file selected" and csv_path != "No file selected":
        process_video(video_path, csv_path)
        return dbc.Alert("Video processed successfully!", color="success")
    return dbc.Alert("Please select both video and output CSV file.", color="danger")


@app.callback(
    Output("visualization", "figure"),
    Input("upload-csv", "contents")
)
def visualize_csv(contents):
    if contents:
        # Implement CSV parsing and visualization logic
        pass
    return px.scatter(title="No data to display")


if __name__ == "__main__":
    app.run_server(debug=True)
