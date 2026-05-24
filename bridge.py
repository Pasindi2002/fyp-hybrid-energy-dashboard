import serial
import csv
import os
import joblib
import warnings
import pymongo
from datetime import datetime
from urllib.parse import quote_plus

warnings.filterwarnings("ignore")

# ── MongoDB Atlas connection ─────────────────────────────────────
password = quote_plus("Pasindi@2002.22")
uri = f"mongodb+srv://en22198822_db_user:{password}@cluster0.zrezdhz.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(uri)
db = client['FYP_Database']
battery_col = db['battery_data']

# ── Load AI model ────────────────────────────────────────────────
teg_model = joblib.load('Models/teg_voltage_rf_model.pkl')

pico_port = "COM12"
baud_rate = 9600
csv_file  = "battery.csv"

if hasattr(teg_model, 'feature_names_in_'):
    features = list(teg_model.feature_names_in_)
else:
    features = ['voltage_mV', 'temp_C', 'teg_count', 'voltage_V',
                'dtemp_dt', 'dvoltage_dt', 'temp_rolling_3',
                'voltage_rolling_3', 'heatsink_bin']

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["Raw_Volts", "SOC_Percentage", "Predicted_TEG_Volts"])

print("Bridge started. Writing to MongoDB Atlas + local CSV...")

try:
    ser = serial.Serial(pico_port, baud_rate, timeout=1)
    print(f"Connected to Pico on {pico_port}\n")

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()

            if "Battery Voltage:" in line:
                try:
                    voltage = float(line.split(":")[1].replace("V", "").strip())
                    soc = max(0, min(100, ((voltage - 2.0) / (3.3 - 2.0)) * 100))

                    # Build AI input
                    input_row = []
                    for col in features:
                        col_lower = col.lower()
                        if "mv" in col_lower:
                            input_row.append(voltage * 1000.0)
                        elif "v" in col_lower or "volt" in col_lower:
                            input_row.append(voltage)
                        elif "temp" in col_lower:
                            input_row.append(52.7)
                        elif "count" in col_lower:
                            input_row.append(4.0)
                        else:
                            input_row.append(0.0)

                    predicted_teg = float(teg_model.predict([input_row])[0])

                    # 1. Save to local CSV (backup)
                    with open(csv_file, mode='a', newline='') as f:
                        csv.writer(f).writerow([voltage, round(soc, 2), round(predicted_teg, 4)])

                    # 2. Save to MongoDB Atlas cloud (dashboard reads this)
                    doc = {
                        "raw_volts":           round(voltage, 4),
                        "soc_percentage":      round(soc, 2),
                        "predicted_teg_volts": round(predicted_teg, 4),
                        "source":              "pico_member3",
                        "timestamp":           datetime.utcnow()
                    }
                    battery_col.insert_one(doc)

                    print(f"✅ Cloud saved | Volts: {voltage}V | SOC: {soc:.1f}% | TEG: {predicted_teg:.4f}V")

                except Exception as e:
                    print(f"Parse error: {e}")

except serial.SerialException:
    print(f"Error: Could not open {pico_port}. Check connection.")
except KeyboardInterrupt:
    print("\nBridge stopped.")
    if 'ser' in locals() and ser.is_open:
        ser.close()