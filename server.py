from flask import Flask, request, jsonify
from flask_cors import CORS
import pymongo
from datetime import datetime
from urllib.parse import quote_plus
import os

app = Flask(__name__)
CORS(app)

# Use environment variable on Render, fallback to local for testing
MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    password = quote_plus("Pasindi@2002.22")
    MONGO_URI = f"mongodb+srv://en22198822_db_user:{password}@cluster0.zrezdhz.mongodb.net/?retryWrites=true&w=majority"

client = pymongo.MongoClient(MONGO_URI)
db     = client['FYP_Database']
collection = db['microgrid_data']

# ── Health check (Render needs this to know server is alive) ─────
@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "TEG Server running", "time": datetime.utcnow().isoformat()}), 200

# ── Receive data from ESP32s (Member 1 & 2) ──────────────────────
@app.route('/api/microgrid', methods=['POST'])
def receive_microgrid_data():
    try:
        incoming_data = request.get_json()
        incoming_data['timestamp'] = datetime.utcnow()
        collection.insert_one(incoming_data)
        print(f"Saved: {incoming_data}")
        return jsonify({"status": "success", "message": "Data saved to cloud"}), 201
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

# ── GET latest records (for dashboard polling) ───────────────────
@app.route('/api/microgrid/latest', methods=['GET'])
def get_latest():
    try:
        limit = int(request.args.get('limit', 50))
        records = list(collection.find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        for r in records:
            if 'timestamp' in r:
                r['timestamp'] = r['timestamp'].isoformat()
        return jsonify(records), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── GET summary (latest single record) ───────────────────────────
@app.route('/api/microgrid/summary', methods=['GET'])
def get_summary():
    try:
        m1 = collection.find_one({"source": "solar"}, {"_id": 0}, sort=[("timestamp", -1)])
        m2 = collection.find_one({"source": "hotpot"}, {"_id": 0}, sort=[("timestamp", -1)])
        latest = collection.find_one({}, {"_id": 0}, sort=[("timestamp", -1)])

        def clean(doc):
            if doc and 'timestamp' in doc:
                doc['timestamp'] = doc['timestamp'].isoformat()
            return doc

        return jsonify({
            "solar":  clean(m1)  or {},
            "hotpot": clean(m2)  or {},
            "latest": clean(latest) or {}
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    print(f"Server starting on port {port}...")
    app.run(host='0.0.0.0', port=port)
