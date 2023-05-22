import pandas as pd 
import numpy as np
from pathlib import Path

#DATA_DIR         =
ROWS_PER_FRAME = 543 
TRAIN_CSV_PATH   = 'train.csv'
LANDMARK_DIR     ='train_landmark_files'
LABEL_MAP_PATH   = 'sign_to_prediction_index_map.json'
train = pd.read_csv(TRAIN_CSV_PATH)
df = pd.read_parquet(train.path[0])
train['sign_ord'] = train['sign'].astype('category').cat.codes
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

def create_frame_landmark_df(results,frame):
    df_skel = df[['type','landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand =pd.DataFrame()

    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i,['x','y','z']] = [point.x,point.y,point.z]
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i,['x','y','z']] = [point.x,point.y,point.z]
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i,['x','y','z']] = [point.x,point.y,point.z]
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i,['x','y','z']] = [point.x,point.y,point.z]
            
    face = face.reset_index().rename(columns={'index':'landmark_index'}).assign(type='face')
    pose = pose.reset_index().rename(columns={'index':'landmark_index'}).assign(type='pose')
    left_hand = left_hand.reset_index().rename(columns={'index':'landmark_index'}).assign(type='left_hand')
    right_hand = right_hand.reset_index().rename(columns={'index':'landmark_index'}).assign(type='right_hand')

    landmarks = pd.concat([face,pose,left_hand,right_hand]).reset_index(drop=True)
    landmarks =  df_skel.merge(landmarks, on = ['type','landmark_index'], how = 'left')
    landmarks = landmarks.assign(frame=frame)
    return landmarks

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)
