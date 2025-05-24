import joblib
import pandas as pd
import pymysql
import time
import random

# Load the Agg. model, scaler, and clusters
scaler = joblib.load('joblib/scaler_ccw_agg.joblib')
clusters = joblib.load('joblib/clusters_ccw_agg.joblib')
knearest_model = joblib.load('joblib/model2_ccw_agg.joblib')
# Load the kmeans model, scaler, and clusters
scaler2 = joblib.load('joblib/scaler_ccw_kmeans.joblib')
clusters2 = joblib.load('joblib/clusters_ccw_kmeans.joblib')
knearest_model2 = joblib.load('joblib/model2_ccw_kmeans.joblib')
# Load the gaussion model, scaler, and clusters
scaler3 = joblib.load('joblib/scaler_ccw_gaussian.joblib')
clusters3 = joblib.load('joblib/clusters_ccw_gaussian.joblib')
gaussian_model3 = joblib.load('joblib/model2_ccw_gaussian.joblib')
# Load the spectral clustering model, scaler, and clusters
scaler4 = joblib.load('joblib/scaler_ccw_spectral.joblib')
clusters4 = joblib.load('joblib/clusters_ccw_spectral.joblib')
spectral_model4 = joblib.load('joblib/model2_ccw_spectral.joblib')
# Load the spectral clustering model, scaler, and clusters
scaler5 = joblib.load('joblib/scaler_ccw_affinity.joblib')
clusters5 = joblib.load('joblib/clusters_ccw_affinity.joblib')
affinity_model5 = joblib.load('joblib/model2_ccw_affinity.joblib')
df5 = joblib.load('joblib/df_ccw_affinity.joblib')
# Load the dbscan model, scaler, and clusters
scaler6 = joblib.load('joblib/scaler_ccw_dbscan.joblib')
clusters6 = joblib.load('joblib/clusters_ccw_dbscan.joblib')
knearest_model6 = joblib.load('joblib/model2_ccw_dbscan.joblib')
# Load the isolationforest. model, scaler, and clusters
scaler7 = joblib.load('joblib/scaler_ccw_iso.joblib')
clusters7 = joblib.load('joblib/clusters_ccw_iso.joblib')
knearest_model7 = joblib.load('joblib/model2_ccw_iso.joblib')
# Load the autoencoder model, scaler, and clusters
scaler8 = joblib.load('joblib/scaler_ccw_optics.joblib')
clusters8 = joblib.load('joblib/clusters_ccw_optics.joblib')
knearest_model8 = joblib.load('joblib/model2_ccw_optics.joblib')
# Load the birch. model, scaler, and clusters
scaler9 = joblib.load('joblib/scaler_ccw_birch.joblib')
clusters9 = joblib.load('joblib/clusters_ccw_birch.joblib')
knearest_model9= joblib.load('joblib/model2_ccw_birch.joblib')
# Load the lof. model, scaler, and clusters
scaler10 = joblib.load('joblib/scaler_ccw_lof.joblib')
clusters10 = joblib.load('joblib/clusters_ccw_lof.joblib')
knearest_model10= joblib.load('joblib/model2_ccw_lof.joblib')
# Load the one class svm. model, scaler, and clusters
scaler11 = joblib.load('joblib/scaler_ccw_svm.joblib')
clusters11 = joblib.load('joblib/clusters_ccw_svm.joblib')
knearest_model11= joblib.load('joblib/model2_ccw_svm.joblib')

# Initialize last processed timestamp
last_processed_timestamp = None  

def fetch_next_values():
    """Fetches three random 'Value' entries from dsas_table for AssetID 8312, 8313, and 8314."""
    try:
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            database='dsas'
        )
        cursor = connection.cursor()

        query = """
            SELECT AssetID, Value 
            FROM dsas_ccw
            WHERE unitID = 11 AND AssetID IN (8330,8331,8332,8333,8335,8338,8339)
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        cursor.close()
        connection.close()

        if rows:
            asset_values = {row[0]: row[1] for row in rows}  # Create dictionary {AssetID: Value}
            
            # Ensure we have values for all three AssetIDs before random selection
            if all(asset_id in asset_values for asset_id in [8330,8331,8332,8333,8335,8338,8339]):
                sensor_values_list = [
                    random.choice([v for k, v in rows if k == 8330]),
                    random.choice([v for k, v in rows if k == 8331]),
                    random.choice([v for k, v in rows if k == 8332]),
                    random.choice([v for k, v in rows if k == 8333]),
                    random.choice([v for k, v in rows if k == 8335]),
                    random.choice([v for k, v in rows if k == 8338]),
                    random.choice([v for k, v in rows if k == 8339])
                ]
                return sensor_values_list
        
        return []
    except pymysql.MySQLError as e:
        print(f"Database Error: {e}")
        return []

def detect_and_store():
    """Continuously fetches new data in batches, applies anomaly detection, and stores results."""
    while True:
        sensor_values_list = fetch_next_values()

        if len(sensor_values_list) == 7:  # Ensure we have exactly three values
            # Construct DataFrame for anomaly detection
            new_data = pd.DataFrame([{
                'GEN.INLET AIR TEMP.': sensor_values_list[0],
                'GEN.OUTLET AIR TEMP.': sensor_values_list[1],
                'C.W.INLET TEMP.': sensor_values_list[2], 
                'C.W. OUTLET TEMP.': sensor_values_list[3], 
                'C.W PUMP OUT LET PRESS': sensor_values_list[4],
                'C.W.INLET PRESSUR': sensor_values_list[5],
                'C.W OUT LET PRESS': sensor_values_list[6]
            }])

            # Scale the input data using the same scaler
            scaled_data = scaler.transform(new_data)
            scaled_data_kmeans = scaler2.transform(new_data)
            scaled_data_gaussion = scaler3.transform(new_data)
            scaled_data_spectral = scaler4.transform(new_data)
            scaled_data_affinity = scaler5.transform(new_data)
            scaled_data_dbscan = scaler6.transform(new_data)
            scaled_data_iso= scaler7.transform(new_data)
            scaled_data_optics= scaler8.transform(new_data)
            scaled_data_birch= scaler9.transform(new_data)
            scaled_data_lof= scaler10.transform(new_data)
            scaled_data_svm= scaler11.transform(new_data)

            # Find the nearest cluster
            _, indices = knearest_model.kneighbors(scaled_data)
            _, indices2 = knearest_model2.kneighbors(scaled_data_kmeans)
            _, indices3 = gaussian_model3.kneighbors(scaled_data_gaussion)
            _, indices4 = spectral_model4.kneighbors(scaled_data_spectral)
            _, indices5 = affinity_model5.kneighbors(scaled_data_affinity)
            _, indices6 = knearest_model6.kneighbors(scaled_data_dbscan)
            _, indices7 = knearest_model7.kneighbors(scaled_data_iso)
            _, indices8 = knearest_model8.kneighbors(scaled_data_optics)
            _, indices9 = knearest_model9.kneighbors(scaled_data_birch)
            _, indices10 = knearest_model10.kneighbors(scaled_data_lof)
            _, indices11 = knearest_model11.kneighbors(scaled_data_svm)

            predicted_cluster = clusters[indices[0][0]]
            predicted_cluster2 = clusters2[indices2[0][0]]
            predicted_cluster3 = clusters3[indices3[0][0]]
            predicted_cluster4 = clusters4[indices4[0][0]]
            predicted_cluster5 = clusters5[indices5[0][0]]
            predicted_cluster6 = clusters6[indices6[0][0]]
            predicted_cluster7 = clusters7[indices7[0][0]]
            predicted_cluster8 = clusters8[indices8[0][0]]
            predicted_cluster9 = clusters9[indices9[0][0]]
            predicted_cluster10 = clusters10[indices10[0][0]]
            predicted_cluster11 = clusters11[indices11[0][0]]

            # Determine normal or abnormal condition
            result1 = 0.528 if predicted_cluster in [0,1] else -0.528
            result2 = 0.413 if predicted_cluster2 in [0,1,2,4] else -0.413
            result3 = 0.152 if predicted_cluster3 in [0,1,3,5] else -0.152
            result4 = 0.333 if predicted_cluster4 in [0,2,6] else -0.333
            result5 = -0.307 if predicted_cluster5 in [1,6,8,38,75,10,63,25,28,40,68,0,5,50,9,27,21,35,20,22,23,24,18,52,54,16,15,49] else 0.307
            result6 = -0.323 if predicted_cluster6 in [-1,0,3,4,5] else 0.323
            result7 = 0.397 if predicted_cluster7 == 1 else -0.397
            result8 = 0.291 if predicted_cluster8 in [0,1] else -0.291
            result9 = -0.340 if predicted_cluster9 in [1,7,9] else 0.340
            result10 = 0.178 if predicted_cluster10 == 1 else -0.178
            result11 = 0.178 if predicted_cluster11 == 1 else -0.178 

            result12 = result1 + result2 + result3 + result4 + result5 + result6 + result7 + result8 + result9 + result10 + result11

            result = "Normal" if result12 > 0 else "Abnormal"

            # Store result in `dsas_results`
            model_name = "Anomaly detection for main transformer"
            unitID = 11
            system = "CCW"

            try:
                connection = pymysql.connect(
                    host='localhost',
                    user='root',
                    password='',
                    database='dsas'
                )
                cursor = connection.cursor()

                query = """
                    INSERT INTO results_dsas_mhi_ccw_11(inputs,results,model_name,unitID,system,score) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (str(sensor_values_list), result, model_name, unitID, system,result12))
                connection.commit()

                print(f"Inserted into results_dsas_mhi_ccw_11 → Values: {sensor_values_list}, Result: {result}")

                cursor.close()
                connection.close()

            except pymysql.MySQLError as e:
                print(f"Database Error: {e}")

        print("Waiting 10 seconds before fetching next valid batch...")
        time.sleep(10)  # Wait before fetching the next batch

# Start continuous detection process
detect_and_store()



