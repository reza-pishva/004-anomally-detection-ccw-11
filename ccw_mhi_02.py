import joblib
import pandas as pd
import pymysql
import time

# Load the Agg. model, scaler, and clusters
scaler = joblib.load('joblib/scaler_main_trans_agg.joblib')
clusters = joblib.load('joblib/clusters_main_trans_agg.joblib')
knearest_model = joblib.load('joblib/model2_main_trans_agg.joblib')
# Load the kmeans model, scaler, and clusters
scaler2 = joblib.load('joblib/scaler_main_trans_kmeans.joblib')
clusters2 = joblib.load('joblib/clusters_main_trans_kmeans.joblib')
knearest_model2 = joblib.load('joblib/model2_main_trans_kmeans.joblib')
# Load the gaussion model, scaler, and clusters
scaler3 = joblib.load('joblib/scaler_main_trans_gaussian.joblib')
clusters3 = joblib.load('joblib/clusters_main_trans_gaussian.joblib')
gaussian_model3 = joblib.load('joblib/model2_main_trans_gaussian.joblib')
# Load the spectral clustering model, scaler, and clusters
scaler4 = joblib.load('joblib/scaler_main_trans_spectral.joblib')
clusters4 = joblib.load('joblib/clusters_main_trans_spectral.joblib')
spectral_model4 = joblib.load('joblib/model2_main_trans_spectral.joblib')
# Load the spectral clustering model, scaler, and clusters
scaler5 = joblib.load('joblib/scaler_main_trans_affinity.joblib')
clusters5 = joblib.load('joblib/clusters_main_trans_affinity.joblib')
affinity_model5 = joblib.load('joblib/model2_main_trans_affinity.joblib')
df5 = joblib.load('joblib/df_main_trans_affinity.joblib')
# Load the dbscan model, scaler, and clusters
scaler6 = joblib.load('joblib/scaler_main_trans_dbscan.joblib')
clusters6 = joblib.load('joblib/clusters_main_trans_dbscan.joblib')
knearest_model6 = joblib.load('joblib/model2_main_trans_dbscan.joblib')
# Load the isolationforest. model, scaler, and clusters
scaler7 = joblib.load('joblib/scaler_main_trans_isolation.joblib')
clusters7 = joblib.load('joblib/clusters_main_trans_isolation.joblib')
knearest_model7 = joblib.load('joblib/model2_main_trans_isolation.joblib')
# Load the autoencoder model, scaler, and clusters
scaler8 = joblib.load('joblib/scaler_main_trans_optics.joblib')
clusters8 = joblib.load('joblib/clusters_main_trans_optics.joblib')
knearest_model8 = joblib.load('joblib/model2_main_trans_optics.joblib')
# Load the birch. model, scaler, and clusters
scaler9 = joblib.load('joblib/scaler_main_trans_birch.joblib')
clusters9 = joblib.load('joblib/clusters_main_trans_birch.joblib')
knearest_model9= joblib.load('joblib/model2_main_trans_birch.joblib')
# Load the lof. model, scaler, and clusters
scaler10 = joblib.load('joblib/scaler_main_trans_lof.joblib')
clusters10 = joblib.load('joblib/clusters_main_trans_lof.joblib')
knearest_model10= joblib.load('joblib/model2_main_trans_lof.joblib')
# Load the one class svm. model, scaler, and clusters
scaler11 = joblib.load('joblib/scaler_main_trans_svm.joblib')
clusters11 = joblib.load('joblib/clusters_main_trans_svm.joblib')
knearest_model11= joblib.load('joblib/model2_main_trans_svm.joblib')

# Initialize last processed timestamp
last_processed_timestamp = None  

def fetch_next_values():
    """Fetches the next three new 'Value' entries from dsas_table, ensuring all required AssetIDs exist."""
    global last_processed_timestamp
    
    try:
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            database='dsas'
        )
        cursor = connection.cursor()

        query = """
            SELECT AssetID, Value, timestamp
            FROM dsas_mhi_main_trans
            WHERE unitID = 12 AND AssetID IN (8312, 8313, 8314)
            AND (timestamp > %s OR %s IS NULL)
            ORDER BY timestamp ASC
            LIMIT 3;
        """
        cursor.execute(query, (last_processed_timestamp, last_processed_timestamp))
        rows = cursor.fetchall()

        cursor.close()
        connection.close()

        if rows:
            asset_ids = {row[0] for row in rows}  # Extract unique AssetIDs
            if asset_ids == {8312, 8313, 8314}:  # Check if all IDs are present
                last_processed_timestamp = rows[-1][2]  # Update timestamp to last processed row
                return [row[1] for row in rows]  # Extract values
            else:
                print("Skipping batch due to missing AssetIDs.")
        
        return []
    except pymysql.MySQLError as e:
        print(f"Database Error: {e}")
        return []

def detect_and_store():
    """Continuously fetches new data in batches, applies anomaly detection, and stores results."""
    while True:
        sensor_values_list = fetch_next_values()

        if len(sensor_values_list) == 3:  # Ensure we have exactly three values
            # Construct DataFrame for anomaly detection
            new_data = pd.DataFrame([{
                'OIL TEMPRATURE': sensor_values_list[0],
                'WINDING TEMP.': sensor_values_list[1], 
                'TAP POSITION': sensor_values_list[2]
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
            result1 = 0.377 if predicted_cluster == 0 else -0.377
            result2 = 0.425 if predicted_cluster2 == 0 else -0.425
            result3 = 0.382 if predicted_cluster3 == 0 else -0.382
            result4 = 0.646 if predicted_cluster4 in [0, 1, 2] else -0.646
            result5 = -0.5 if predicted_cluster5 in [27,28,30,37,39,40,42,47,49,57,59,71,73,149,151,178,181,0,2,60,20,24,189,183,185,194,199,207,211,212,215,216,218,235] else 0.5
            result6 = 0.122 if predicted_cluster6 in [0,1,2,3] else -0.122
            result7 = 0.646 if predicted_cluster7 == 1 else -0.646
            result8 = -0.375 if predicted_cluster8 == 1 else 0.375 
            result9 = 0.020 if predicted_cluster9 in [0,1,2,3] else -0.020
            result10 = 0.295 if predicted_cluster10 == 1 else -0.295
            result11 = 0.048 if predicted_cluster11 == 1 else -0.048 

            result12 = result1 + result2 + result3 + result4 + result5 + result6 + result7 + result8 + result9 + result10 + result11
            result = "Normal" if result12 > 0 else "Abnormal"

            # Store result in `dsas_results`
            model_name = "Anomaly detection for main transformer"
            unitID = 12
            system = "Main Transformer"

            try:
                connection = pymysql.connect(
                    host='localhost',
                    user='root',
                    password='',
                    database='dsas'
                )
                cursor = connection.cursor()

                query = """
                    INSERT INTO results_dsas_mhi_main_trans_12(inputs, results, model_name, unitID, system) 
                    VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(query, (str(sensor_values_list), result, model_name, unitID, system))
                connection.commit()

                print(f"Inserted into results_dsas_mhi_main_trans_12 → Values: {sensor_values_list}, Result: {result}")

                cursor.close()
                connection.close()

            except pymysql.MySQLError as e:
                print(f"Database Error: {e}")

        print("Waiting 10 seconds before fetching next valid batch...")
        time.sleep(10)  # Wait before fetching the next batch

# Start continuous detection process
detect_and_store()




