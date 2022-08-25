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


# ------------------------------------------------------------
seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
"""Returns `True` if the user had the correct password."""

def password_entered():
"""Checks whether a password entered by the user is correct."""
    if st.session_state["password"] == st.secrets["password"]:
        st.session_state["password_correct"] = True
        del st.session_state["password"]  # don't store password
    else:
        st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("Password incorrect")
        return False
    else:
        # Password correct.
        return True

# --------------------PAGESTYLE--------------------------------
st.set_page_config(
    layout="wide"
    )

if check_password():
    # ---------------------SIDEBAR----------------------------------

    with st.sidebar:
        sc1, sc2 = st.columns((1, 2.0))
        with sc1:
            st.image("./image/cuhksz.png")
        with sc2:
            st.title("RISKVIZ WEB")
        st.write("    ")
        st.write("    ")
        bt1 = st.button("Home Page         ", key=1)
        bt2 = st.button("Model Instruction  ", key=2)
        bt3 = st.button("Risk Visualization", key=3)
        bt4 = st.button("Model Deployment",  key=4)
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
            The source code will not open to the pubic.  
            Connected the author by 120090527@link.cuhk.edu.cn if you need technical help.    
        """
        )

    # ----------------Page: Model Instruction---------------------
    if bt2:
        st.title("Model Instruction")
        st.header("1.Introduction")
        st.markdown('''
            The algorithm aims to develop a method of evaluating the real-time collision risk.
            It will use TELL, a deep clustering algorithm, to find the risk level. 
            The risk level ranges from 0 to 3. The final resualt is the weighted average of those four level. 
            The general workflow is composed of three parts:  
            (1) ROS2DF  
            (2) Feature Engineering  
            (3) TELL  
            The detailed information will be illustrated in the following sections.
        ''')
        st.header("2.Model Composition")
        st.subheader("2.1. ROS2DF")
        st.markdown('''
        ROS2DF is developed based on the python library rosbag.
        It transfers the rosbag file into pandas dataframes and paves way for futher feature engineering.
        Note that momenta is replacing rosbag with mfbags. This part will be further developed into MF2DF.
        ''')
        st.subheader("2.2. Feature Engineering")
        st.markdown('''
        It's hard to measure the level of safety using only raw data. That's why feature engineering is used.
        During this step, raw data will be transfered into six safety-related criteria named surrogate safety metrics (SSMs),
        to quantify severity of traffic interaction.
        We will do feature engineering based on SSMs, incorporate our domain knowledge into the clustering model and enhance the explanablity of the model.
        The detailed information is listed as follows
        ''')
        st.markdown('''
        (1) U_car
        The car potential (U_car) is used to to keep the vehicle a safe distance from each obstacle
        by building a potential that rises to infinite stength approaching any part of the vehicle.
        The foundation of the car potential is the Yukawa potential.
        ''')
        st.latex(r"U_{car,m}(K)=A_{car}{e^{-\alpha K}\over K}")
        st.markdown('''
        where K is a distance measurement to the $m^{th}$ obstacle car.
        The detailed informationa bout U_car can be find in the paper
        < Artificial Potential Functions for Highway Driving with Collision Avoidance >.''')
        st.markdown('''
        (2) C2C Rate  
        C2C Rate indicates the severity of the collision point.
        A basic intuition is that the front collision is more sever than the side collision.
        When there is no potential collision risk, its default value is set to -1 for convenience.''')
        st.markdown('''
        (3) TTC  
        TTC is widely used to measure the driving safety.
        It means the time until a collision between the vehicles would occur if they continued on their present course at their present speeds.
        ''')
        st.markdown('''
        Note:
        More two-dimension SSMs (like DRAC, PTE, and PSD) can be added in the future to enhance the model.
        ''')
        st.subheader("2.3. TELL")
        st.markdown('''
        TELL is a continous deep clusterin algorithm, which uses neurual network to find the clustering center.
        The encoder and decoder will be both 4-layers fully connected anto-encoder, which is applied to makes feature reduction and decomposition.
        ''')
        st.image("./image/TELL_workflow.PNG")
        st.header("3.Further Improvement")
        st.markdown('''
        (1)Data  
        Users can colleted some extreme cases into the training data. ''')
        st.markdown('''
        (2)Feature engineering  
        More SSMs or other criteria can be used in this step. However, remember to change the parameter in TELL network.''')
        st.markdown('''
        (3)Multimodality  
        Raw lader data and image data may also serve as an input in the future. ''')
        st.header('''
        4.Reference
        ''')
        st.markdown('''
        Mahmud, S. S., Ferreira, L., Hoque, M. S., & Tavassoli, A. (2017). Application of proximal surrogate indicators for safety evaluation: A review of recent developments and research needs. IATSS research, 41(4), 153-163.
        ''')
        st.markdown('''Peng, X., Li, Y., Tsang, I. W., Zhu, H., Lv, J., & Zhou, J. T. (2022). XAI Beyond Classification: Interpretable Neural Clustering. J. Mach. Learn. Res., 23, 6-1.''')
        st.markdown(
            '''Souman, J., Adjenughwure, K., & Tejada, A. (2021). Quantification of Safe Driving.''')
        st.markdown('''Tejada, A., Manders, J., Snijders, R., Paardekooper, J. P., & de Hair-Buijssen, S. (2020, September). Towards a Characterization of Safe Driving Behavior for Automated Vehicles Based on Models of “Typical” Human Driving Behavior. In 2020 IEEE 23rd International Conference on Intelligent Transportation Systems(ITSC)(pp. 1-6). IEEE.
        ''')

    # -------------------------PAGE: Model Deployment---------------------------
    elif bt4:
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
        $pip install - r requirement.txt
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
        Given that Momenta is using AWS, the following link might be helpful,
        ''')
        st.markdown("1.Deployment on AWS with authentication [LINK](https://discuss.streamlit.io/t/deployment-on-aws-with-authentication/4073)")
        st.markdown("2.New deployment option AWS app runner [LINK](https://discuss.streamlit.io/t/new-deployment-option-aws-app-runner/13084)")
        st.markdown("3.Azure Deployment / AWS Deployment of Streamlit apps [LINK](https://discuss.streamlit.io/t/azure-deployment-aws-deployment-of-streamlit-apps/13234)")
        st.markdown("More information can be found in the community of streamlit [LINK](https://discuss.streamlit.io)")

    # --------------------------PAGE: Home Page-------------------------------
    elif bt1:
        col11,col12,col11 = st.columns((0.25,1,0.2))
        with col12:
            st.title("DRIVING RISK VISUALIZATION TOOL")
        col11,col12,col11 = st.columns((0.71,1,0.2))
        with col12:
            st.markdown("SAFETY IS THE HIGHEST PRIORITY")
        st.image("./image/DJI_0372.jpg")
    # --------------------------PAGE: Risk Visualization-------------------
    else:    
        st.title("Risk Visualization")
        file_path = st.text_input('Enter the path of your rosbag file', './output.bag')	
        with st.spinner('Wait for it. The neural network is drawing the risk graph...'):
            df = ros2df(file=file_path)
            st.table(df.head())

        df["TTC"] = (-df["TTC"]).apply(logistic_transformer)
        df[["RelativeVelocity_x","RelativeVelocity_y","RelativeHeadingYaw","TTC","C2CRate","U_car"]] = (
            scale.transform(X=df[["RelativeVelocity_x","RelativeVelocity_y","RelativeHeadingYaw","TTC","C2CRate","U_car"]])
            )
        # df_ = df[["RelativeVelocity_x","RelativeVelocity_y","RelativeHeadingYaw","TTC","C2CRate","U_car"]]
        data = DDSafetyDataset(df[["RelativeVelocity_x","RelativeVelocity_y","RelativeHeadingYaw","TTC","C2CRate","U_car"]])
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
        
