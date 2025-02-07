import joblib
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import streamlit.components.v1 as components


# Text in first column
st.markdown("<div style='font-size:36px; font-weight:bold;text-align:center'>M701D Series Gas Turbines</div>", unsafe_allow_html=True)
st.image('index_im01.jpg', caption='', use_column_width=True)
st.markdown("<div style='font-size:20px; font-weight:bold;'>Anomaly Detection for CCW (Closed Circuit Water) system</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div style='font-size:18px; font-weight:bold;color:blue'>Data Scientist: Reza Pishva</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size:18px; font-weight:bold;color:blue'>Quality Assurance Engineer: Almas Baharlooie</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.write("The CCW (Closed Circuit Water) system in the M701D Series Gas Turbines is a critical component for cooling and maintaining the efficiency of the turbine. Here's a brief overview:")
st.write("1. Purpose: The CCW system helps in cooling the turbine components, particularly the blades and vanes, to prevent overheating and maintain optimal performance.")
st.write("2. Components: It typically includes a network of pipes, pumps, and heat exchangers that circulate water or a water-glycol mixture through the turbine")
st.write("3. Operation: The system continuously circulates the cooling fluid, absorbing heat from the hot turbine parts and transferring it to a heat exchanger where it is dissipated.")
st.write("4. Benefits: By maintaining lower temperatures, the CCW system enhances the durability and lifespan of the turbine components, improves efficiency, and reduces the risk of thermal stress and damage.")

st.write("The model was trained using historical data, with careful tuning of parameters.")
st.markdown("<br>", unsafe_allow_html=True)
df2 = joblib.load('df.joblib')
st.table(df2)

st.write("The features used to train the model are the following:")
st.write("1. GEN.INLET AIR TEMP.")
st.write("2. GEN.OUTLET AIR TEMP.")
st.write("3. C.W.INLET TEMP.")
st.write("4. C.W. OUTLET TEMP.")
st.write("5. C.W PUMP OUT LET PRESS")
st.write("6. C.W.INLET PRESSUR")
st.write("7. C.W OUT LET PRESS")

st.write("The silhouette score of the model indicates good clustering performance.")
st.write("You may select the value of each feature to determine whether the condition based on our selection is normal or abnormal:")
st.markdown(
    """
    <style>
    .stNumberInput .st-af {
        width: 150px; /* Adjust the width as needed */
    }
    .stSlider > div[role="slider"] {
        width: 150px; /* Adjust the width as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Setting the min and max values
input1 = st.slider('GEN.INLET AIR TEMP.', min_value=0, max_value=100, value=40)
input2 = st.slider('GEN.OUTLET AIR TEMP.', min_value=0, max_value=100, value=69)
input3 = st.slider('C.W.INLET TEMP.', min_value=0, max_value=100, value=37)
input4 = st.slider('C.W. OUTLET TEMP.', min_value=0, max_value=100, value=43)
input5 = st.slider('C.W PUMP OUT LET PRESS', min_value=0.0, max_value=10.0, value=4.9 , step=0.1)
input6 = st.slider('C.W.INLET PRESSUR', min_value=0.0, max_value=10.0, value=4.4 , step=0.1)
input7 = st.slider('C.W OUT LET PRESS', min_value=0.0, max_value=10.0, value=4.2 , step=0.1)

# Create a DataFrame with the input data
new_data = pd.DataFrame([{
    'GEN.INLET AIR TEMP.': input1,
    'GEN.OUTLET AIR TEMP.': input2, 
    'C.W.INLET TEMP.': input3, 
    'C.W. OUTLET TEMP.': input4, 
    'C.W PUMP OUT LET PRESS': input5,
    'C.W.INLET PRESSUR': input6,
    'C.W OUT LET PRESS': input7
}])
# Load the Agg. model, scaler, and clusters
scaler = joblib.load('scaler_ccw_agg.joblib')
clusters = joblib.load('clusters_ccw_agg.joblib')
knearest_model = joblib.load('model2_ccw_agg.joblib')
# Load the kmeans model, scaler, and clusters
scaler2 = joblib.load('scaler_ccw_kmeans.joblib')
clusters2 = joblib.load('clusters_ccw_kmeans.joblib')
knearest_model2 = joblib.load('model2_ccw_kmeans.joblib')
# Load the gaussion model, scaler, and clusters
scaler3 = joblib.load('scaler_ccw_gaussian.joblib')
clusters3 = joblib.load('clusters_ccw_gaussian.joblib')
gaussian_model3 = joblib.load('model2_ccw_gaussian.joblib')
# Load the spectral clustering model, scaler, and clusters
scaler4 = joblib.load('scaler_ccw_spectral.joblib')
clusters4 = joblib.load('clusters_ccw_spectral.joblib')
spectral_model4 = joblib.load('model2_ccw_spectral.joblib')
# Load the spectral clustering model, scaler, and clusters
scaler5 = joblib.load('scaler_ccw_affinity.joblib')
clusters5 = joblib.load('clusters_ccw_affinity.joblib')
affinity_model5 = joblib.load('model2_ccw_affinity.joblib')
df5 = joblib.load('df_ccw_affinity.joblib')
# Load the dbscan model, scaler, and clusters
scaler6 = joblib.load('scaler_ccw_dbscan.joblib')
clusters6 = joblib.load('clusters_ccw_dbscan.joblib')
knearest_model6 = joblib.load('model2_ccw_dbscan.joblib')
# # Load the autoencoder model, scaler, and clusters
# scaler4 = joblib.load('scaler_vib_temp_auto.joblib')
# clusters4 = joblib.load('clusters_vib_temp_auto.joblib')
# knearest_model4 = joblib.load('model2_vib_temp_auto.joblib')
# Load the dbscan. model, scaler, and clusters
# scaler5 = joblib.load('scaler_ccw_dbscan.joblib')
# clusters5 = joblib.load('clusters_ccw_dbscan.joblib')
# knearest_model5 = joblib.load('model2_ccw_dbscan.joblib')
# # Load the isolationforest. model, scaler, and clusters
# scaler6 = joblib.load('scaler_vib_temp_iso.joblib')
# clusters6 = joblib.load('clusters_vib_temp_iso.joblib')
# knearest_model6 = joblib.load('model2_vib_temp_iso.joblib')


# Scale the input data using the same scaler
scaled_data = scaler.transform(new_data)
scaled_data_kmeans = scaler2.transform(new_data)
scaled_data_gaussion = scaler3.transform(new_data)
scaled_data_spectral = scaler4.transform(new_data)
scaled_data_affinity = scaler5.transform(new_data)
scaled_data_dbscan = scaler6.transform(new_data)
# Find the nearest cluster
_, indices = knearest_model.kneighbors(scaled_data)
_, indices2 = knearest_model2.kneighbors(scaled_data)
_, indices3 = gaussian_model3.kneighbors(scaled_data)
_, indices4 = spectral_model4.kneighbors(scaled_data)
_, indices5 = affinity_model5.kneighbors(scaled_data)
_, indices6 = knearest_model6.kneighbors(scaled_data)

predicted_cluster = clusters[indices[0][0]]
predicted_cluster2 = clusters2[indices2[0][0]]
predicted_cluster3 = clusters3[indices3[0][0]]
predicted_cluster4 = clusters4[indices4[0][0]]
predicted_cluster5 = clusters5[indices5[0][0]]
predicted_cluster6 = clusters6[indices6[0][0]]

# Predict the cluster for the input data
agg =""
kmeans =""
gaussian =""
autoencoder =""
dbscan =""
isolationforest=""
spectral=""
affinity=""
if st.button('Predict Cluster'):
    if (clusters[indices[0][0]]==0 or clusters[indices[0][0]]==1):
        agg = "Normal"      
    else:  
        agg = "Abnormal"       
    if (clusters2[indices2[0][0]]==0 or clusters2[indices2[0][0]]==1 or
        clusters2[indices2[0][0]]==2 or clusters2[indices2[0][0]]==4):
        kmeans = "Normal"
    else:
        kmeans = "Abnormal"
    if (clusters3[indices3[0][0]]==0 or clusters3[indices3[0][0]]==1 or 
        clusters3[indices3[0][0]]==3 or clusters3[indices3[0][0]]==5):
        gaussian = "Normal"
    else:
        gaussian = "Abnormal"  
    if (clusters4[indices4[0][0]]==0 or clusters4[indices4[0][0]]==2 or
        clusters4[indices4[0][0]]==7):
        spectral = "Normal"
    else:
        spectral = "Abnormal"      
    if (clusters5[indices5[0][0]]==1 or
        clusters5[indices5[0][0]]==6 or
        clusters5[indices5[0][0]]==8 or
        clusters5[indices5[0][0]]==38 or
        clusters5[indices5[0][0]]==75 or
        clusters5[indices5[0][0]]==10 or
        clusters5[indices5[0][0]]==63 or
        clusters5[indices5[0][0]]==25 or
        clusters5[indices5[0][0]]==28 or
        clusters5[indices5[0][0]]==40 or
        clusters5[indices5[0][0]]==68 or
        clusters5[indices5[0][0]]==0 or
        clusters5[indices5[0][0]]==5 or
        clusters5[indices5[0][0]]==50 or
        clusters5[indices5[0][0]]==9 or
        clusters5[indices5[0][0]]==27 or
        clusters5[indices5[0][0]]==21 or
        clusters5[indices5[0][0]]==35 or
        clusters5[indices5[0][0]]==20 or
        clusters5[indices5[0][0]]==22 or
        clusters5[indices5[0][0]]==23 or
        clusters5[indices5[0][0]]==24 or
        clusters5[indices5[0][0]]==18 or
        clusters5[indices5[0][0]]==52 or
        clusters5[indices5[0][0]]==54 or
        clusters5[indices5[0][0]]==16 or
        clusters5[indices5[0][0]]==15 or
        clusters5[indices5[0][0]]==49):
        affinity = "Abnormal"
    else:
        affinity = "Normal"   
    if (clusters6[indices6[0][0]]==-1 or clusters6[indices6[0][0]]==0 or 
        clusters6[indices6[0][0]]==3 or clusters6[indices6[0][0]]==4 or 
        clusters6[indices6[0][0]]==5):
        dbscan = "Abnormal"
    else:
        dbscan = "Normal"             

def get_color(value):
    if value == "Normal":
        return 'font-size:18px;font-weight: bold;color: green;'
    elif value == "Abnormal":
        return 'font-size:18px;font-weight: bold;color: red;'
    else:
        return ''


html_table = f"""
<table style="width:100%; border-collapse: collapse;">
  <tr>
    <th style="border: 1px solid black; padding: 8px;">Clustering algorithm</th>
    <th style="border: 1px solid black; padding: 8px;">Result</th>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;">Agglomerative Clustering</td>
    <td style="border: 1px solid black; padding: 8px;{get_color(agg)}">{agg}</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;">KMeans Clustering</td>
    <td style="border: 1px solid black; padding: 8px;{get_color(kmeans)}">{kmeans}</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;">Gaussian Clustering</td>
    <td style="border: 1px solid black; padding: 8px;{get_color(gaussian)}">{gaussian}</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;">Spectral Clustering</td>
    <td style="border: 1px solid black; padding: 8px;{get_color(spectral)}">{spectral}</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;">Affinity propagation Clustering</td>
    <td style="border: 1px solid black; padding: 8px;{get_color(affinity)}">{affinity}</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;">DBSCAN Clustering</td>
    <td style="border: 1px solid black; padding: 8px;{get_color(dbscan)}">{dbscan}</td>
  </tr>
</table>
"""
# Use st.markdown to display the HTML table
st.markdown(html_table, unsafe_allow_html=True)


st.markdown("<div style='margin-top:50px;font-size:24px;color:white;font-weight:bold;height:40px;background-color:#4C585B;border-radius:5px;text-align:center'>Agglomerative clustering</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.write(agg)
st.write(
    """
    <style>
    .dataframe th, .dataframe td {
        text-align: center;

    }
    </style>
    """,
    unsafe_allow_html=True
)
data = {
    'Cluster': [0,1,2,3,4],
    'Count': [3353,2170,1,20,4]
}
df = pd.DataFrame(data)
col1, col2 = st.columns([2,1])
with col1:
    st.write("This model has been trained by 5548 instances.")
    st.write("In this model, I have chosen 5 clusters. Clusters 2, 3, and 4 have been selected as abnormal conditions, and they contain 0.45 percent of the data.")
    st.write("In the lower part there are Siluouette plot and PCA plot. I have reduced the dimensions into 2 so that we can see data points distribution for different clusters.")
    st.write("As you can see there are two large clusters and three small ones. Increasing the number of clusters did not help, and we always end up with many large clusters and two or three small ones.")
with col2:
    st.table(df)
col1, col2 = st.columns(2)
with col1:
    st.image('pca-agg.png', caption='Data distribution', use_column_width=True)
with col2:
    st.image('sil-agg.png', caption='Evaluation by Silhouette', use_column_width=True) 


st.markdown("<div style='font-size:24px;color:white;font-weight:bold;height:40px;background-color:#4C585B;border-radius:5px;text-align:center'>KMeans clustering</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.write(kmeans)
st.write(
    """
    <style>
    .dataframe th, .dataframe td {
        text-align: center;

    }
    </style>
    """,
    unsafe_allow_html=True
)
data = {
    'Cluster': [1, 2, 3, 4, 5],
    'Count': [2246, 1390, 686, 310, 600]
}
df = pd.DataFrame(data)
col1, col2 = st.columns([2,1])
with col1:
    st.write("This model has been trained by 5232 instances.")
    st.write("In this model I have reached to 5 clusters and it was close to the result of elbow plot and cluster 4 has been selected as abnormal condition. About 5.92 percent of data points has been selected as abnormal data.")
    # st.write("In the lower part there are Siluouette plot and PCA plot.")
    st.write("As you can see these clusters are closer to 1 compare to agglomerative model and the average score is more than that model and have fewer negative values so it might be more reliable rather than previous clustering model.")
with col2:
    st.table(df)

col1, col2 = st.columns(2)
with col1:
    st.image('pca-kmeans.png', caption='Data distribution', use_column_width=True)
with col2:
    st.image('sil-kmeans.png', caption='Evaluation by Silhouette', use_column_width=True) 
st.markdown("<div style='font-size:24px;color:white;font-weight:bold;height:40px;background-color:#4C585B;border-radius:5px;text-align:center'>Gaussian Mixture Model</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.write(gaussian)
st.write(
    """
    <style>
    .dataframe th, .dataframe td {
        text-align: center;

    }
    </style>
    """,
    unsafe_allow_html=True
)
data = {
    'Cluster': [0,1, 2, 3, 4 ,5],
    'Count': [709,2226,282,1216,193,614]
}
df = pd.DataFrame(data)
col1, col2 = st.columns([2,1])
with col1:
    st.write("This model has been trained by 5240 instances.")
    st.write("In this model, I have chosen 6 clusters. Clusters 2 and 4 have been identified as abnormal conditions, and 14 percent of the instances have been classified as abnormal data points.")
    # st.write("In the lower part there are Siluouette plot and PCA plot.")
    st.write("As you can see Average Silhouette Score is equal to 0.15 and it is far from 1.")
with col2:
    st.table(df)

col1, col2 = st.columns(2)
with col1:
    st.image('pca-gaussian.png', caption='Data distribution', use_column_width=True)
with col2:
    st.image('sil-gaussian.png', caption='Evaluation by Silhouette', use_column_width=True) 


st.markdown("<div style='font-size:24px;color:white;font-weight:bold;height:40px;background-color:#4C585B;border-radius:5px;text-align:center'>Affinity propagation Clustering</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.write(affinity)
st.write(
    """
    <style>
    .dataframe th, .dataframe td {
        text-align: center;

    }
    </style>
    """,
    unsafe_allow_html=True
)
data = {
    'Cluster': [1, 2, 3, 4, 5, 6 , 7 , 8],
    'Count': [1386,202,1658,104,440,14,152,1284]
}
df = pd.DataFrame(data)
col1, col2 = st.columns([2,1])
with col1:
    st.write("This model has been trained by 5240 instances.")
    st.write("In this model I have chosen 8 clusters and clusters 5 and 6 have been selected as abnormal condition.In this model I have considered 12.5 percent of instences as abnormal data.")
    # st.write("In the lower part there are Siluouette plot and PCA plot.")
    st.write("As you can see Average Silhouette Score is equal to 0.367 and it is far from 1.")
with col2:

    st.markdown(
    """
    <style>
    .dataframe-container {
        margin-top:-150px;
        height: 100px;
        overflow-y: scroll;
    }
    </style>
    """,
    unsafe_allow_html=True)
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(df5)
    st.markdown('</div>', unsafe_allow_html=True)
  

# col1, col2 = st.columns(2)
# with col1:
#     st.image('pca-affinity.png', caption='Data distribution', use_column_width=True)
# with col2:
#     st.image('sil-affinity.png', caption='Evaluation by Silhouette', use_column_width=True)     




st.markdown("<div style='font-size:24px;color:white;font-weight:bold;height:40px;background-color:#4C585B;border-radius:5px;text-align:center'>DBSCAN</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.write(dbscan)
st.write(
    """
    <style>
    .dataframe th, .dataframe td {
        text-align: center;

    }
    </style>
    """,
    unsafe_allow_html=True
)
data = {
    'Cluster': [-1, 0, 1, 2, 3, 4, 5],
    'Count': [95,194,2902,1800,37,99,113]
}
df = pd.DataFrame(data)
col1, col2 = st.columns([2,1])
with col1:
    st.write("This model has been trained with 5240 instances. In this model, I have reached 7 clusters, and the hyperparameters are chosen as eps=0.7 and min_samples=10. Clusters 1 and 2 have been selected as the normal condition. In this model, I have considered 10 percent of instances as abnormal data. As you can see, the Average Silhouette Score is 0.06, which is far from 1 compared to earlier models, and there are many negative values among the clusters in the silhouette plot.")    
with col2:
    st.table(df)

col1, col2 = st.columns(2)
with col1:
    st.image('pca-dbscan.png', caption='Data distribution', use_column_width=True)
with col2:
    st.image('sil-dbscan.png', caption='Evaluation by Silhouette', use_column_width=True)     


st.markdown("<div style='font-size:24px;color:white;font-weight:bold;height:40px;background-color:#4C585B;border-radius:5px;text-align:center'>Spectral Clustering</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.write(autoencoder)
st.write(
    """
    <style>
    .dataframe th, .dataframe td {
        text-align: center;

    }
    </style>
    """,
    unsafe_allow_html=True
)
data = {
    'Cluster': [-1, 0, 1, 2, 3, 4, 5],
    'Count': [95,194,2902,1800,37,99,113]
}
df = pd.DataFrame(data)
col1, col2 = st.columns([2,1])
with col1:
    st.write("This model has been trained by 5240 instances.")
    st.write("In this model I have chosen 8 clusters and clusters 5 and 6 have been selected as abnormal condition.In this model I have considered 12.5 percent of instences as abnormal data.")
    # st.write("In the lower part there are Siluouette plot and PCA plot.")
    st.write("As you can see Average Silhouette Score is equal to 0.367 and it is far from 1.")
with col2:
    st.table(df)

col1, col2 = st.columns(2)
with col1:
    st.image('pca-spectral.png', caption='Data distribution', use_column_width=True)
with col2:
    st.image('sil-spectral.png', caption='Evaluation by Silhouette', use_column_width=True)     

  

  
