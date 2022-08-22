import streamlit as st
import pandas as pd
import numpy as np
import os
import numpy as pd
import joblib
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import *

from ros2df import *
from tell import *


#------------------------------------------------------------
seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

scale = joblib.load("./model/scaler1.pkl")
net = Net(dim = 6, class_num= 4).cpu()
model_fp = os.path.join("./model/checkpoint_1700.tar")
checkpoint = torch.load(model_fp,map_location=torch.device('cpu'))
net.load_state_dict(checkpoint["net"], strict=False)

#--------------------PAGESTYLE--------------------------------

st.set_page_config(
     layout="wide"
 )

#---------------------SIDEBAR----------------------------------

#sb = st.sidebar("")
with st.sidebar:
    sc1, sc2 = st.columns((1,2.51))
    with sc1:
        st.image("./image/logo1.png")	
    with sc2:
        st.title("RISKVIZ TOOL")
    st.write("    ")
    st.write("    ")
    bt1 = st.button("Home Page         " ,key=1)
    bt2 = st.button("Model Instruction  ",key=2)
    bt3 = st.button("Risk Visualization" ,key=3)
    bt4 = st.button("Model Deployment" ,key=4)
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.write("    ")
    st.write("    ")

    st.sidebar.info(
        """
        This web [app](https://ricklin616-riskviz-momenta-stream-try-9ndhvy.streamlitapp.com) is maintained by Jinrui Lin.  
        Source code: <https://github.com/RickLin616/RiskVIz_Momenta>
    """
    )
 


#----------------Page: Model Instruction---------------------
if bt2:
    st.title("Model Instruction")
    st.header("1.Introduction")
    st.markdown('''
        The algorithm aims to develop a method of evaluating real-time driving risk. 
        It will use TELL, a deep clustering algorithm, to find the risk clustering centers. 
        The general workflow is composed of four parts:  
    (1) ROS2DF  
    (2) Feature Engineering  
    (3) AutoEncoder (Encoder and Decoder)  
    (4) TELL  
    The detailed information will be illustrated in the following sections.    
    ''')
    st.header("2.Model Composition")
    st.subheader("2.1. ROS2DF")
    st.markdown('''
    ROS2DF is developed based on the python library rosbag. 
    It transforms the rosbag file into pandas dataframes and paves way for feature engineering.
    Note that momenta is replacing rosbag with mfbags. This part will be further developed into MF2DF.
    ''')    
    st.subheader("2.2. Feature Engineering")
    st.markdown('''
    The current method utilizes surrogate safety metrics (SSMs) to quantify severity of traffic interaction. 
    Such SSMs are mostly defined to identify safety-critical situations. 
    We will do feature engineering based on SSMs and incorporate our domain knowledge into our clustering model. 
    ''')
    st.subheader("2.3. AutoEncoder")   
    st.markdown('''
    The encoder and decoder will both be 3-layers(TBD) fully connected anto-encoder. 
    It makes feature reduction and decomposition.
    ''') 
    st.subheader("2.4. TELL:a continous deep clustering model")
    
    st.header("3.Reference")
    st.markdown('''
    
    ''')

#-------------------------PAGE: Model Deployment---------------------------
if bt4:
    st.title("Model Deployment Instruction")
    st.header("1.Local Deployment")
    st.markdown('''
    The following instruction illustrates how to deploy the app on a local computer. 
    Before moving to specific steps, I highly encourage you to deploy this app in a new virtual environment. 

    After that your can analysis the rosbag file using the local url.
    The file of this tool is attached and please download it.
    Then open the terminal of your file and establish the environment according to the requirement.txt with the following code:
    ''' )
    st.code('''
    $pip install -r requirement.txt
    ''')
    st.markdown('''
    After all libararies are installed, run the main.py with the following code.
    ''')
    st.code('''
    $streamlit run ./main.py
    ''')
    st.markdown('''
    Open the link displayed in the terminal and the deployment is finished.
    ''')
    st.header("2.Cloud Deployment") 
    st.markdown('''
    Depends which type of cloud momenta is using. 
    The following link might be helpful
    ''')


#--------------------------PAGE: Home Page-------------------------------
elif bt1:
    col11,col12,col11 = st.columns((0.25,1,0.2))
    with col12:
        st.title("DRIVING RISK VISUALIZATION TOOL")
    col11,col12,col11 = st.columns((0.71,1,0.2))
    with col12:
        st.markdown("SAFETY IS THE HIGHEST PRIORITY")
    st.image("./image/DJI_0372.jpg")

#--------------------------PAGE: Risk Visualization-------------------
else:    
    st.title("Risk Visualization")
    file_path = st.text_input('Enter the path of your rosbag file', 'output.bag')

    with st.spinner('Wait for it. The neural network is drawing the risk graph...'):
        df = ros2df(file=file_path)
        st.table(df.head())

    df["TTC"] = (-df["TTC"]).apply(logistic_transformer)
    df[["RelativeVelocity_x","RelativeVelocity_y","RelativeHeadingYaw","TTC","C2CRate","U_car"]] = (
        scale.transform(X=df[["RelativeVelocity_x","RelativeVelocity_y","RelativeHeadingYaw","TTC","C2CRate","U_car"]])
        )
    df[["RelativeVelocity_x","RelativeVelocity_y","RelativeHeadingYaw","TTC","C2CRate","U_car"]].to_excel("./temporary_file/indicators_.xlsx")
    data = DDSafetyDataset(path = "./temporary_file/indicators_.xlsx")
    data_loader_test = DataLoader(data, batch_size=df.shape[0], shuffle= False, drop_last=True)  # ConcatDataset([train_dataset, test_dataset])
    feature, labels, hard, soft = inference(data_loader_test, net)
    soft = soft @ np.array([0,1,2,3])
    df["soft"] = 3-soft
    df["hard"] = 3-hard

    col1, col2 = st.columns([3,1])
    with col1:
        st.line_chart(df.groupby("TimeStamp")["soft"].max(), width=900, height=380, use_container_width = True)
    with col2:
        st.line_chart(df["RelativeVelocity_x"], width=300, height=105, use_container_width = True)
        st.line_chart(df["RelativeVelocity_y"], width=300, height=105, use_container_width = True)
        st.line_chart(df["RelativeHeadingYaw"], width=300, height=105, use_container_width = True)    
    
