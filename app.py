import streamlit as st
import cv2 
import io
import pandas as pd 
import mediapipe as mp 
import numpy as np 
import tempfile
import time
import urllib.request 
from PIL import Image 
import Functionalities as functionalities
import ISL as isl 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

st.set_page_config(page_title="Face_Cam",
                   page_icon = ":heart:")
title = st.title('Face Mesh App using MediaPipe')
st.sidebar.title('FaceMesh Sidebar')
st.sidebar.subheader('parameters')

@st.cache()
def image_resize(image,width=None,height=None,inter=cv2.INTER_AREA):
    dim = None 
    (h,w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    
    else:
        r = width/float(w)
        dim = (width,int(h*r))
    
    #resize the image
    resized = cv2.resize(image,dim, interpolation = inter)
    return resized

app_modes = ['About App','Run on Image', 'Run on Video']
app_mode = st.sidebar.selectbox('Choose the App Mode',app_modes)
if app_mode == 'About App':
    st.write("Face Mesh is a computer vision technology that uses machine learning algorithms to detect \
             and track facial landmarks in real-time video. \
             MediaPipe is a framework for building cross-platform machine learning pipelines, \
             including computer vision and audio processing applications.")
    st.markdown("""
            <style>
            .stApp {
                background-color: rgba(0,0,0,0);
                background-image: url('https://4kwallpapers.com/images/wallpapers/refraction-black-hole-astronaut-planet-earth-outer-space-3840x2160-7559.jpg');
                background-size: 100% 100%;
                background-position: center;
                background-repeat: no-repeat;
                }
            </style>
            """,
            unsafe_allow_html=True)
    st.video('https://youtu.be/V9bzew8A1tc')
    
elif app_mode == 'Run on Image':
    title.empty()
    st.markdown("""
                <style>
                .stApp {
                background-color: rgba(0,0,0,0);
                background-size: 100% 100%;
                background-image: url(https://static.vecteezy.com/system/resources/thumbnails/006/422/170/original/hairline-motion-with-black-background-good-for-wallpaper-screensaver-free-video.jpg);
                background-position: center;
                background-repeat: no-repeat;
                }
                </style>""",
                unsafe_allow_html=True)
    st.title("Detect Faces and Contours")
    mark_down_faces = st.markdown('**Detected Faces**')
    number_faces = st.markdown("0")
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius = 1)
    st.sidebar.markdown("-----")
    mark_down_faces_number = st.sidebar.number_input('Maximum Number of Faces', 
                                                     value=2, 
                                                     min_value=1)
    st.sidebar.markdown("-----")
    mark_down_detect_conf = st.sidebar.slider('Minimum Detection Confidence', min_value=0.0,
                                              max_value=1.0,value=0.5)
    st.sidebar.markdown("-----")
    img_file_buffer = st.sidebar.file_uploader("Uplod an Image",
                                               type=["jpg","jpeg","png"])
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    else:
        demo_image = urllib.request.urlretrieve(
            'https://media.istockphoto.com/id/914478746/photo/gorgeous-young-woman-with-clean-fresh-skin-is-touching-own-face-cosmetology.jpg?s=2048x2048&w=is&k=20&c=d5ax5hnxPKZyZxQLVjd-z3TKCDMnCumjTfS5Jwc9RD0=',
            "demo.jpg"
        )
        image = np.array(Image.open("demo.jpg"))
    
    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode = True,
        max_num_faces=mark_down_faces_number,
        min_detection_confidence=mark_down_detect_conf) as face_mesh:
        
        results = face_mesh.process(image)
        out_image = image.copy()
        face_count =0
        for face_landmarks in results.multi_face_landmarks:
            face_count =face_count + 1
            mp_drawing.draw_landmarks(
                image=out_image,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=out_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            number_faces.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader("Output Image")
        st.image(out_image,use_column_width=True)
            
elif app_mode == "Run on Video":
    title.empty()
    st.set_option('deprecation.showfileUploaderEncoding',
                  False)
    use_webcam = st.sidebar.button('Use Webcam')
    stop_webcam = st.sidebar.button('Stop Webcam')
    record = st.sidebar.checkbox("Record Video")
    
    if record:
        st.checkbox("Recording", value=True)
        
    max_faces = st.sidebar.number_input("Maximum Number of Faces", value=1,min_value=1)
    st.sidebar.markdown("-----")
    mark_down_detect_conf = st.sidebar.slider('Minimum Detection Confidence', 
                                              min_value=0.0,
                                              max_value=1.0,
                                              value=0.5)
    mark_down_track_conf = st.sidebar.slider('Minimum Tracking Confidence', 
                                              min_value=0.0,
                                              max_value=1.0,
                                              value=0.5)
    st.sidebar.markdown("-----")
    st.markdown("## Output") 
    
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a  Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False) 
    
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture('Detroit_ Become Human – Launch Trailer _ PS4.mp4') 
    else:
        tffile.write(video_file_buffer.read())
        vid=cv2.VideoCapture(tffile.name)
    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    
    #Recording Part
    codec=cv2.VideoWriter_fourcc('a','v','c','1')
    out = cv2.VideoWriter('output.mp4',codec,fps,(width,height))
    
    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)
    
    fps = 0
    i=0
    
    drawing_spec = mp_drawing.DrawingSpec(thickness=0.5,
                                          circle_radius=0.5)
    
    kpi1,kpi2,kpi3 = st.columns(3)
    
    with kpi1:
        st.markdown("**FRAME RATE**")
        kpi1 = st.markdown("0")
        
    with kpi2:
        st.markdown("**DETECTED FACES**")
        kpi2 = st.markdown("0")
        
    with kpi3:
        st.markdown("**IMAGE WIDTH**")
        kpi3 = st.markdown("0")
    
    st.markdown('<hr/>',unsafe_allow_html=True)
    
    all_landmarks = []
    with mp_holistic.Holistic(
        min_detection_confidence = mark_down_detect_conf,
        min_tracking_confidence = mark_down_track_conf
    ) as holistic:
        frame_number = 0
        prevTime=0
        while vid.isOpened():
            frame_number+=1
            ret,frame = vid.read()
            if not ret:
                continue
            
            results = holistic.process(frame)
            frame.flags.writeable = True
            landmarks = functionalities.create_frame_landmark_df(results,frame_number)
            all_landmarks.append(landmarks)
            #face_count = 0
            #mp_drawing.draw_landmarks(
            #    frame,
            #    results.face_landmarks,
            #   mp_holistic.FACEMESH_CONTOURS,
            #    landmark_drawing_spec=drawing_spec,
            #    connection_drawing_spec=mp_drawing_styles
            #    .get_default_face_mesh_contours_style()
            #)
            #mp_drawing.draw_landmarks(
            #    frame,
            #    results.pose_landmarks,
            #    mp_holistic.POSE_CONNECTIONS,
            #    landmark_drawing_spec=mp_drawing_styles
            #    .get_default_pose_landmarks_style())
            
            currTime = time.time()
            fps = 1/(currTime - prevTime)
            prevTime = currTime
            
            if record:
                out.write(frame)
            
            # Dashboard
            kpi1.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2.write(f"<h1 style='text-align: center; color:red;'>{1}</h1>", unsafe_allow_html=True)
            kpi3.write(f"<h1 style='text-align: center; color:red;'>{width*height}</h1>",
                                 unsafe_allow_html=True)

            frame = cv2.resize(frame,(0,0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame,channels='BGR', use_column_width=True)
            if stop_webcam or (frame_number>23):
                vid.release()
                cv2.destroyAllWindows()
    print(all_landmarks)        
    train_parquet = pd.concat(all_landmarks).reset_index(drop=True).to_parquet("output_10.parquet")
    raw_data = functionalities.load_relevant_data_subset('output_10.parquet')    
    output = isl.final_model(raw_data)["outputs"]
    prediction = output.numpy().argmax()
    s = functionalities.ORD2SIGN[prediction]        
    st.write(s)   