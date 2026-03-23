import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from threading import Lock

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pickle
from scipy.signal import butter, filtfilt, find_peaks, iirnotch
from werkzeug.utils import secure_filename

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_ROOT, "ct_config.json")
DEFAULT_CONFIG = {
    "server_base": "http://localhost:5000/",
    "serial_port": "COM3",
    "serial_baud": 115200,
    "scan_seconds": 20,
    "model_file": "model.pkl",
    "local_db": "pico_local.db",
}


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                if k not in data:
                    data[k] = v
            return data
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


CFG = load_config()
DB_PATH = os.path.join(APP_ROOT, CFG.get("local_db", "pico_local.db"))
DATA_DIR = os.path.join(APP_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

state_lock = Lock()
scan_state = {
    "queued_patient_id": 0,
    "queued_at": 0.0,
    "running_patient_id": 0,
    "running_since": 0.0,
    "last_result_ts": 0.0,
    "last_result_scan_id": 0,
}


METRIC_COLUMNS = {
    "hr": "REAL",
    "sbp": "REAL",
    "dbp": "REAL",
    "ptt": "REAL",
    "hrv_rmssd": "REAL",
    "spo2": "REAL",
    "breathing_rate": "REAL",
    "hr_status": "TEXT",
    "bp_status": "TEXT",
    "health_status": "TEXT",
    "risk_score": "INTEGER",
}

MODEL_PATH = os.path.join(APP_ROOT, CFG.get("model_file", "model.pkl"))
_model_cache = None
_model_mtime = None


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def bandpass(signal, fs, lowcut=0.5, highcut=40.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def detect_r_peaks(ecg, fs=250):
    try:
        signal = bandpass(ecg, fs)
    except Exception:
        signal = ecg

    diff = np.diff(signal)
    diff = np.append(diff, 0)
    squared = diff ** 2
    window = int(0.150 * fs)
    if window < 1:
        window = 1
    ma = np.convolve(squared, np.ones(window) / window, mode="same")

    try:
        peaks, _ = find_peaks(ma, height=np.mean(ma) * 1.2, distance=int(0.25 * fs))
    except Exception:
        peaks, _ = find_peaks(ma, distance=int(0.25 * fs))
    return peaks


def detect_ppg_peaks(ppg, fs=50):
    b, a = butter(2, 8 / (0.5 * fs), btype="low")
    filt = filtfilt(b, a, ppg)
    try:
        peaks, _ = find_peaks(filt, distance=int(0.3 * fs), height=np.mean(filt) * 1.0)
    except Exception:
        peaks, _ = find_peaks(filt, distance=int(0.3 * fs))
    return peaks


def extract_features_for_prediction(timestamps, ecg, red, ir):
    if len(ecg) < 50:
        return None

    t_arr = np.array(timestamps)
    diffs = np.diff(t_arr)
    if len(diffs) == 0 or np.mean(diffs) <= 0:
        return None

    fs = 1000.0 / np.mean(diffs)
    rpeaks = detect_r_peaks(np.array(ecg), fs=int(round(fs)))
    if len(rpeaks) < 2:
        return None

    rr_intervals = np.diff(t_arr[rpeaks]) / 1000.0
    if len(rr_intervals) == 0 or np.mean(rr_intervals) == 0:
        return None

    hrv_rmssd = 0.0
    try:
        rr_ms = np.diff(t_arr[rpeaks])
        if len(rr_ms) >= 2:
            hrv_rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    except Exception:
        hrv_rmssd = 0.0

    hr = 60.0 / np.mean(rr_intervals)
    hr_std = float(np.std(60.0 / rr_intervals))

    ppg_signal = np.array(ir) if np.sum(ir) > 0 else np.array(red)
    if len(ppg_signal) < 20:
        return None

    ppg_fs = max(20, int(round(len(ppg_signal) / ((timestamps[-1] - timestamps[0]) / 1000.0))))
    ppg_peaks = detect_ppg_peaks(ppg_signal, fs=ppg_fs)

    breathing_rate = 0.0
    try:
        from scipy.fft import rfft, rfftfreq
        yf = np.abs(rfft(ppg_signal - np.mean(ppg_signal)))
        xf = rfftfreq(len(ppg_signal), d=1.0 / float(ppg_fs))
        mask = (xf >= 0.15) & (xf <= 0.4)
        if np.any(mask):
            band_freqs = xf[mask]
            band_power = yf[mask]
            if len(band_power) > 0 and np.max(band_power) > 0:
                dominant_freq = float(band_freqs[np.argmax(band_power)])
                breathing_rate = dominant_freq * 60.0
    except Exception:
        breathing_rate = 0.0

    spo2 = 0.0
    try:
        red_arr = np.array(red, dtype=float)
        ir_arr = np.array(ir, dtype=float)
        ac_red = float(np.std(red_arr))
        dc_red = float(np.mean(red_arr)) if float(np.mean(red_arr)) != 0 else 1.0
        ac_ir = float(np.std(ir_arr))
        dc_ir = float(np.mean(ir_arr)) if float(np.mean(ir_arr)) != 0 else 1.0
        ratio = (ac_red / dc_red) / (ac_ir / dc_ir) if (ac_ir / dc_ir) != 0 else 0.0
        spo2 = float(np.clip(round(110.0 - 25.0 * ratio, 1), 70.0, 100.0))
    except Exception:
        spo2 = 0.0

    ptt_list = []
    for rp in rpeaks:
        r_time = t_arr[rp]
        idx = np.where(t_arr >= r_time)[0]
        if len(idx) == 0:
            continue
        start_idx = idx[0]
        p_candidates = [p for p in ppg_peaks if p >= start_idx]
        if len(p_candidates) == 0:
            continue
        p_time = t_arr[p_candidates[0]]
        ptt_list.append((p_time - r_time) / 1000.0)

    avg_ptt = float(np.mean(ptt_list)) if len(ptt_list) > 0 else 0.0
    avg_ptt = avg_ptt * 0.15
    sbp = max(90, min(170, round(120 - (80 * avg_ptt), 1)))
    dbp = max(60, min(110, round(80 - (50 * avg_ptt), 1)))

    return {
        "mean_hr": float(hr),
        "hr_std": float(hr_std),
        "avg_ptt": float(avg_ptt),
        "num_rpeaks": int(len(rpeaks)),
        "num_ppg_peaks": int(len(ppg_peaks)),
        "sbp_est": sbp,
        "dbp_est": dbp,
        "hrv_rmssd": float(hrv_rmssd),
        "spo2": float(spo2),
        "breathing_rate": float(breathing_rate),
    }


def parse_scan_csv(csv_path):
    import csv

    timestamps = []
    ecg = []
    red = []
    ir = []

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        field_map = {name.strip().lower(): name for name in (reader.fieldnames or [])}

        ts_key = field_map.get("ts_ms") or field_map.get("ts")
        ecg_key = field_map.get("ecg")
        red_key = field_map.get("red")
        ir_key = field_map.get("ir")

        if not all([ts_key, ecg_key, red_key, ir_key]):
            raise ValueError("CSV must contain columns: ts_ms (or ts), ecg, red, ir")

        for row in reader:
            try:
                timestamps.append(int(float(row.get(ts_key, 0))))
                ecg.append(int(float(row.get(ecg_key, 0))))
                red.append(int(float(row.get(red_key, 0))))
                ir.append(int(float(row.get(ir_key, 0))))
            except Exception:
                continue

    if len(timestamps) < 10:
        raise ValueError("CSV has insufficient valid samples")

    duration_sec = max(0.0, (timestamps[-1] - timestamps[0]) / 1000.0)
    return timestamps, ecg, red, ir, duration_sec


def load_model():
    global _model_cache, _model_mtime

    if not os.path.exists(MODEL_PATH):
        return None

    mtime = os.path.getmtime(MODEL_PATH)
    if _model_cache is not None and _model_mtime == mtime:
        return _model_cache

    with open(MODEL_PATH, "rb") as f:
        _model_cache = pickle.load(f)
    _model_mtime = mtime
    return _model_cache


def calculate_risk_and_status(features, prediction):
    hr = features.get("mean_hr") if features else None
    sbp = features.get("sbp_est") if features else None
    dbp = features.get("dbp_est") if features else None

    if hr is None:
        hr_status = "--"
    elif hr < 60:
        hr_status = "Bradycardia"
    elif hr > 100:
        hr_status = "Tachycardia"
    else:
        hr_status = "Normal"

    if sbp is None or dbp is None:
        bp_status = "--"
    elif sbp < 90 or dbp < 60:
        bp_status = "Hypotension"
    elif sbp < 120 and dbp < 80:
        bp_status = "Normal"
    elif sbp < 130:
        bp_status = "Elevated"
    elif sbp < 140:
        bp_status = "Stage 1 Hypertension"
    else:
        bp_status = "Stage 2 Hypertension"

    risk_score = 0
    if features:
        if hr is not None and (hr > 100 or hr < 60):
            risk_score += 2
        if sbp is not None and sbp > 130:
            risk_score += 2
        if features.get("avg_ptt", 0) > 0.3:
            risk_score += 2
        if str(prediction).lower() != "normal":
            risk_score += 4

    risk_score = min(risk_score, 10)
    health_status = "Stable" if str(prediction).lower() == "normal" else "Risk detected"
    return hr_status, bp_status, health_status, risk_score


def build_personalized_suggestions(features, prediction):
    points = []
    pred = str(prediction or "").lower()

    hr = float(features.get("mean_hr")) if features and features.get("mean_hr") is not None else None
    sbp = float(features.get("sbp_est")) if features and features.get("sbp_est") is not None else None
    dbp = float(features.get("dbp_est")) if features and features.get("dbp_est") is not None else None
    spo2 = float(features.get("spo2")) if features and features.get("spo2") is not None else None
    breathing = float(features.get("breathing_rate")) if features and features.get("breathing_rate") is not None else None
    ptt = float(features.get("avg_ptt")) if features and features.get("avg_ptt") is not None else None

    if pred == "normal":
        points.append("Current scan looks stable. Keep a consistent routine and recheck every 2-4 weeks.")
    else:
        points.append("This scan shows risk-like patterns. Repeat scan after 10 minutes rest and share report with a clinician.")

    if spo2 is not None:
        if spo2 < 85:
            points.append("SpO2 is in critical range. Seek urgent medical care if breathlessness, chest pain, or dizziness is present.")
        elif spo2 < 90:
            points.append("SpO2 is low. Prioritize breathing rest, avoid exertion, and schedule medical review soon.")
        elif spo2 < 95:
            points.append("SpO2 is borderline. Add daily walking and breathing exercises, then monitor trends over next scans.")
        else:
            points.append("SpO2 is in normal range. Maintain hydration, sleep quality, and regular light activity.")

    if hr is not None:
        if hr > 100:
            points.append("Heart rate is elevated. Reduce caffeine/stimulants today and repeat scan in a calm seated state.")
        elif hr < 60:
            points.append("Heart rate is below usual resting range. Recheck symptoms and review with clinician if persistent.")
        else:
            points.append("Heart rate is within expected range. Continue at least 30 minutes of moderate activity most days.")

    if sbp is not None and dbp is not None:
        if sbp >= 140 or dbp >= 90:
            points.append("Blood pressure is high. Reduce salt intake, track BP for 7 days, and plan clinical follow-up.")
        elif sbp >= 130 or dbp >= 80:
            points.append("Blood pressure is mildly elevated. Focus on lower sodium meals and regular sleep schedule.")
        else:
            points.append("Blood pressure is controlled. Keep current diet and hydration habits.")

    if breathing is not None:
        if breathing > 20:
            points.append("Breathing rate is elevated. Practice 5 minutes of slow breathing and rescan after rest.")
        elif breathing < 12:
            points.append("Breathing rate is low. Recheck while fully awake; seek care if weakness or confusion appears.")

    if ptt is not None and ptt > 0.28:
        points.append("Pulse transit time is prolonged. Keep stress low and track repeat scans for trend confirmation.")

    return "\n".join([f"- {p}" for p in points])


def run_prediction_for_csv(csv_path):
    timestamps, ecg, red, ir, duration_sec = parse_scan_csv(csv_path)
    features = extract_features_for_prediction(timestamps, ecg, red, ir)

    if not features:
        return {
            "prediction": "unknown",
            "explainable": "Feature extraction failed",
            "suggestion": "Rescan or upload cleaner CSV data.",
            "hr": None,
            "sbp": None,
            "dbp": None,
            "ptt": None,
            "hr_status": "--",
            "bp_status": "--",
            "health_status": "Risk detected",
            "risk_score": 4,
            "duration_sec": duration_sec,
        }

    model = load_model()
    prediction = "unknown"
    explainable = "No model available"

    if model is not None:
        feat_vec = np.array([
            features["mean_hr"],
            features["hr_std"],
            features["avg_ptt"],
            features["num_rpeaks"],
            features["num_ppg_peaks"],
        ]).reshape(1, -1)

        try:
            prediction = str(model.predict(feat_vec)[0])
        except Exception:
            prediction = "unknown"

        try:
            importances = model.feature_importances_
            feat_names = ["mean_hr", "hr_std", "avg_ptt", "num_rpeaks", "num_ppg_peaks"]
            imp_list = sorted(zip(feat_names, importances), key=lambda x: -x[1])
            explainable = "Feature importances: " + ", ".join([f"{n}:{i:.2f}" for n, i in imp_list])
        except Exception:
            explainable = "No feature importance available"

    # Safety layer: treat very abnormal physiology as risk even if model underfits.
    if prediction.lower() == "normal":
        if (
            features["mean_hr"] < 55
            or features["mean_hr"] > 110
            or features["sbp_est"] >= 140
            or features["dbp_est"] >= 90
            or features["avg_ptt"] > 0.28
            or features["hr_std"] > 20
        ):
            prediction = "possible_block"

    suggestion = build_personalized_suggestions(features, prediction)

    hr_status, bp_status, health_status, risk_score = calculate_risk_and_status(features, prediction)

    return {
        "prediction": prediction,
        "explainable": explainable,
        "suggestion": suggestion,
        "hr": round(float(features.get("mean_hr")), 1),
        "sbp": float(features.get("sbp_est")),
        "dbp": float(features.get("dbp_est")),
        "ptt": round(float(features.get("avg_ptt")), 4),
        "hrv_rmssd": round(float(features.get("hrv_rmssd", 0.0)), 2),
        "spo2": round(float(features.get("spo2", 0.0)), 1),
        "breathing_rate": round(float(features.get("breathing_rate", 0.0)), 1),
        "hr_status": hr_status,
        "bp_status": bp_status,
        "health_status": health_status,
        "risk_score": int(risk_score),
        "duration_sec": duration_sec,
    }


def ensure_schema():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            started_at TEXT,
            finished_at TEXT,
            raw_csv TEXT,
            prediction TEXT,
            explainable TEXT,
            suggestion TEXT,
            hr REAL,
            sbp REAL,
            dbp REAL,
            ptt REAL,
            hrv_rmssd REAL,
            spo2 REAL,
            breathing_rate REAL,
            hr_status TEXT,
            bp_status TEXT,
            health_status TEXT,
            risk_score INTEGER
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            blood_group TEXT,
            phone TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    existing_cols = {
        row["name"] for row in cur.execute("PRAGMA table_info(scans)").fetchall()
    }
    for col_name, col_type in METRIC_COLUMNS.items():
        if col_name not in existing_cols:
            cur.execute(f"ALTER TABLE scans ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()


def backfill_patients_from_scans():
    conn = get_db_connection()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT DISTINCT s.patient_id
        FROM scans s
        LEFT JOIN patients p ON p.id = s.patient_id
        WHERE s.patient_id IS NOT NULL AND s.patient_id > 0 AND p.id IS NULL
        ORDER BY s.patient_id ASC
        """
    ).fetchall()

    for r in rows:
        pid = int(r["patient_id"])
        cur.execute(
            """
            INSERT INTO patients (id, name, age, gender, blood_group, phone)
            VALUES (?, ?, NULL, NULL, NULL, NULL)
            """,
            (pid, f"Patient #{pid}"),
        )

    conn.commit()
    conn.close()


def parse_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def now_iso():
    return datetime.utcnow().isoformat()


def get_request_value(key, default=None):
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        if key in payload:
            return payload.get(key)
    if key in request.form:
        return request.form.get(key)
    return request.args.get(key, default)


def row_to_dict(row):
    return dict(row) if row is not None else None


def resolve_csv_path(raw_csv_path):
    if not raw_csv_path:
        return None

    if os.path.isabs(raw_csv_path):
        return raw_csv_path

    return os.path.join(APP_ROOT, raw_csv_path)


def mark_scan_queued(patient_id):
    with state_lock:
        ts = time.time()
        scan_state["queued_patient_id"] = patient_id
        scan_state["queued_at"] = ts
        scan_state["running_patient_id"] = patient_id
        scan_state["running_since"] = ts


def pop_queued_scan():
    with state_lock:
        patient_id = scan_state["queued_patient_id"]
        if patient_id > 0:
            scan_state["queued_patient_id"] = 0
            scan_state["queued_at"] = 0.0
            return {"run_scan": 1, "patient_id": int(patient_id)}
        return {"run_scan": 0, "patient_id": 0}


def mark_scan_completed(scan_id):
    with state_lock:
        scan_state["last_result_ts"] = time.time()
        scan_state["last_result_scan_id"] = int(scan_id)
        scan_state["running_patient_id"] = 0
        scan_state["running_since"] = 0.0


def current_running_state():
    with state_lock:
        running_patient_id = scan_state["running_patient_id"]
        running_since = scan_state["running_since"]
        last_result_ts = scan_state["last_result_ts"]

    now = time.time()
    scan_window = max(20, parse_int(CFG.get("scan_seconds"), default=20) or 20)
    max_running_seconds = scan_window + 20
    running = False
    if running_patient_id and running_since > 0:
        not_expired = (now - running_since) <= max_running_seconds
        no_result_yet = last_result_ts < running_since
        running = bool(not_expired and no_result_yet)

    return {
        "running": running,
        "patient_id": int(running_patient_id if running else 0),
    }


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


@app.route("/api/config", methods=["GET"])
def api_config():
    return jsonify({
        "scan_seconds": parse_int(CFG.get("scan_seconds"), default=20) or 20,
        "server_base": CFG.get("server_base", "http://localhost:5000/"),
    })


@app.route("/api/patients", methods=["GET"])
def get_patients():
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT
            p.id,
            p.name,
            p.age,
            p.gender,
            p.blood_group,
            p.phone,
            p.created_at,
            ls.started_at AS last_scan_date,
            ls.prediction AS last_prediction
        FROM patients p
        LEFT JOIN (
            SELECT s1.patient_id, s1.started_at, s1.prediction
            FROM scans s1
            INNER JOIN (
                SELECT patient_id, MAX(id) AS max_id
                FROM scans
                GROUP BY patient_id
            ) s2
            ON s1.patient_id = s2.patient_id AND s1.id = s2.max_id
        ) ls
        ON p.id = ls.patient_id
        ORDER BY p.id DESC
        """
    ).fetchall()
    conn.close()

    return jsonify({"patients": [dict(r) for r in rows]})


@app.route("/api/patients", methods=["POST"])
def create_patient():
    body = request.get_json(silent=True) or {}

    name = (body.get("name") or "").strip()
    age = parse_int(body.get("age"), default=None)
    gender = (body.get("gender") or "").strip() or None
    blood_group = (body.get("blood_group") or "").strip() or None
    phone = (body.get("phone") or "").strip() or None

    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO patients (name, age, gender, blood_group, phone)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, age, gender, blood_group, phone),
    )
    patient_id = cur.lastrowid
    conn.commit()

    patient = conn.execute(
        "SELECT id, name, age, gender, blood_group, phone, created_at FROM patients WHERE id = ?",
        (patient_id,),
    ).fetchone()
    conn.close()

    return jsonify({"success": True, "patient": row_to_dict(patient)})


@app.route("/api/patients/<int:patient_id>", methods=["GET"])
def get_patient(patient_id):
    conn = get_db_connection()
    patient = conn.execute(
        """
        SELECT
            p.id,
            p.name,
            p.age,
            p.gender,
            p.blood_group,
            p.phone,
            p.created_at,
            ls.started_at AS last_scan_date,
            ls.prediction AS last_prediction
        FROM patients p
        LEFT JOIN (
            SELECT s1.patient_id, s1.started_at, s1.prediction
            FROM scans s1
            INNER JOIN (
                SELECT patient_id, MAX(id) AS max_id
                FROM scans
                GROUP BY patient_id
            ) s2
            ON s1.patient_id = s2.patient_id AND s1.id = s2.max_id
        ) ls ON p.id = ls.patient_id
        WHERE p.id = ?
        """,
        (patient_id,),
    ).fetchone()
    conn.close()

    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    return jsonify({"patient": row_to_dict(patient)})


@app.route("/api/patients/<int:patient_id>", methods=["PUT"])
def update_patient(patient_id):
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    age = parse_int(body.get("age"), default=None)
    gender = (body.get("gender") or "").strip() or None
    blood_group = (body.get("blood_group") or "").strip() or None
    phone = (body.get("phone") or "").strip() or None

    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    exists = conn.execute("SELECT id FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if not exists:
        conn.close()
        return jsonify({"error": "Patient not found"}), 404

    cur.execute(
        """
        UPDATE patients
        SET name = ?, age = ?, gender = ?, blood_group = ?, phone = ?
        WHERE id = ?
        """,
        (name, age, gender, blood_group, phone, patient_id),
    )
    conn.commit()
    updated = conn.execute(
        "SELECT id, name, age, gender, blood_group, phone, created_at FROM patients WHERE id = ?",
        (patient_id,),
    ).fetchone()
    conn.close()

    return jsonify({"success": True, "patient": row_to_dict(updated)})


@app.route("/api/patients/<int:patient_id>", methods=["DELETE"])
def delete_patient(patient_id):
    conn = get_db_connection()
    cur = conn.cursor()

    patient = conn.execute(
        "SELECT id, name FROM patients WHERE id = ?",
        (patient_id,),
    ).fetchone()
    if not patient:
        conn.close()
        return jsonify({"success": False, "error": "Patient not found"}), 404

    scan_rows = conn.execute(
        "SELECT id, raw_csv FROM scans WHERE patient_id = ?",
        (patient_id,),
    ).fetchall()

    cur.execute("DELETE FROM scans WHERE patient_id = ?", (patient_id,))
    deleted_scans = cur.rowcount if cur.rowcount is not None else len(scan_rows)
    cur.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    conn.commit()
    conn.close()

    for row in scan_rows:
        abs_csv_path = resolve_csv_path(row["raw_csv"])
        if abs_csv_path and os.path.exists(abs_csv_path):
            try:
                os.remove(abs_csv_path)
            except Exception:
                pass

    with state_lock:
        if scan_state["queued_patient_id"] == patient_id:
            scan_state["queued_patient_id"] = 0
            scan_state["queued_at"] = 0.0
        if scan_state["running_patient_id"] == patient_id:
            scan_state["running_patient_id"] = 0
            scan_state["running_since"] = 0.0

    return jsonify(
        {
            "success": True,
            "deleted_patient_id": patient_id,
            "deleted_scans": int(deleted_scans),
        }
    )


@app.route("/api/patients/<int:patient_id>/scans", methods=["GET"])
def get_patient_scans(patient_id):
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT
            id,
            patient_id,
            started_at,
            finished_at,
            raw_csv,
            prediction,
            explainable,
            suggestion,
            hr,
            sbp,
            dbp,
            ptt,
            hrv_rmssd,
            spo2,
            breathing_rate,
            hr_status,
            bp_status,
            health_status,
            risk_score
        FROM scans
        WHERE patient_id = ?
        ORDER BY id DESC
        """,
        (patient_id,),
    ).fetchall()
    conn.close()

    return jsonify({"scans": [dict(r) for r in rows]})


@app.route("/api/patients/<int:patient_id>/trend", methods=["GET"])
def get_patient_trend(patient_id):
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT started_at, hr, spo2, hrv_rmssd, breathing_rate
        FROM scans
        WHERE patient_id = ?
        ORDER BY started_at DESC
        LIMIT 10
        """,
        (patient_id,),
    ).fetchall()
    conn.close()

    rows = list(reversed(rows))

    labels = []
    hr = []
    spo2 = []
    hrv_rmssd = []
    breathing_rate = []

    for row in rows:
        labels.append(row["started_at"])
        hr.append(row["hr"])
        spo2.append(row["spo2"])
        hrv_rmssd.append(row["hrv_rmssd"])
        breathing_rate.append(row["breathing_rate"])

    return jsonify(
        {
            "labels": labels,
            "hr": hr,
            "spo2": spo2,
            "hrv_rmssd": hrv_rmssd,
            "breathing_rate": breathing_rate,
        }
    )


@app.route("/api/scans/<int:scan_id>", methods=["GET"])
def get_scan(scan_id):
    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT
            s.id,
            s.patient_id,
            p.name AS patient_name,
            p.age AS patient_age,
            p.gender AS patient_gender,
            p.blood_group AS patient_blood_group,
            p.phone AS patient_phone,
            s.started_at,
            s.finished_at,
            s.raw_csv,
            s.prediction,
            s.explainable,
            s.suggestion,
            s.hr,
            s.sbp,
            s.dbp,
            s.ptt,
            s.hrv_rmssd,
            s.spo2,
            s.breathing_rate,
            s.hr_status,
            s.bp_status,
            s.health_status,
            s.risk_score
        FROM scans s
        LEFT JOIN patients p ON p.id = s.patient_id
        WHERE s.id = ?
        """,
        (scan_id,),
    ).fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Scan not found"}), 404

    return jsonify({"scan": row_to_dict(row)})


@app.route("/api/scans/<int:scan_id>/waveform", methods=["GET"])
def get_scan_waveform(scan_id):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT id, raw_csv FROM scans WHERE id = ?",
        (scan_id,),
    ).fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Scan not found"}), 404

    raw_csv_path = row["raw_csv"]
    abs_csv_path = resolve_csv_path(raw_csv_path)

    if not abs_csv_path or not os.path.exists(abs_csv_path):
        return jsonify({"error": "CSV file not found for this scan"}), 404

    try:
        timestamps, ecg, red, ir, _ = parse_scan_csv(abs_csv_path)
    except Exception as e:
        return jsonify({"error": f"Failed to parse scan CSV: {str(e)}"}), 400

    ppg = ir if np.sum(ir) > 0 else red

    # Keep payload lightweight for frontend animation.
    max_points = 2500
    step = max(1, int(len(timestamps) / max_points))
    ts_ds = timestamps[::step]
    ecg_ds = ecg[::step]
    ppg_ds = ppg[::step]

    sample_rate_hz = 250.0
    if len(ts_ds) > 1:
        dt = (ts_ds[-1] - ts_ds[0]) / max(1, (len(ts_ds) - 1))
        if dt > 0:
            sample_rate_hz = 1000.0 / dt

    return jsonify(
        {
            "waveform": {
                "scan_id": scan_id,
                "count": len(ts_ds),
                "sample_rate_hz": round(sample_rate_hz, 2),
                "ts_ms": ts_ds,
                "ecg": ecg_ds,
                "ppg": ppg_ds,
            }
        }
    )


@app.route("/api/scan/trigger", methods=["POST"])
def trigger_scan():
    body = request.get_json(silent=True) or {}
    patient_id = parse_int(body.get("patient_id"), default=0)
    if patient_id <= 0:
        return jsonify({"success": False, "error": "Valid patient_id is required"}), 400

    conn = get_db_connection()
    patient = conn.execute("SELECT id FROM patients WHERE id = ?", (patient_id,)).fetchone()
    conn.close()
    if not patient:
        return jsonify({"success": False, "error": "Patient not found"}), 404

    mark_scan_queued(patient_id)
    return jsonify({"success": True, "queued": True, "patient_id": patient_id})


@app.route("/api/scan/upload", methods=["POST"])
def upload_scan_csv():
    patient_id = parse_int(get_request_value("patient_id"), default=0)
    if patient_id <= 0:
        return jsonify({"success": False, "error": "Valid patient_id is required"}), 400

    conn = get_db_connection()
    patient = conn.execute("SELECT id FROM patients WHERE id = ?", (patient_id,)).fetchone()
    conn.close()
    if not patient:
        return jsonify({"success": False, "error": "Patient not found"}), 404

    uploaded_csv = request.files.get("raw_csv") or request.files.get("file")
    if not uploaded_csv or not uploaded_csv.filename:
        return jsonify({"success": False, "error": "CSV file is required"}), 400

    safe_name = secure_filename(uploaded_csv.filename)
    if not safe_name.lower().endswith(".csv"):
        return jsonify({"success": False, "error": "Only .csv files are supported"}), 400

    file_name = f"scan_upload_{int(time.time())}_{patient_id}_{safe_name}"
    abs_path = os.path.join(DATA_DIR, file_name)
    uploaded_csv.save(abs_path)
    raw_csv_path = os.path.relpath(abs_path, APP_ROOT).replace("\\", "/")

    try:
        pred = run_prediction_for_csv(abs_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"CSV processing failed: {str(e)}"}), 400

    finished_dt = datetime.utcnow()
    started_dt = finished_dt - timedelta(seconds=max(1, int(round(pred.get("duration_sec", 20.0)))))
    started_at = started_dt.isoformat()
    finished_at = finished_dt.isoformat()

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO scans (
            patient_id,
            started_at,
            finished_at,
            raw_csv,
            prediction,
            explainable,
            suggestion,
            hr,
            sbp,
            dbp,
            ptt,
            hrv_rmssd,
            spo2,
            breathing_rate,
            hr_status,
            bp_status,
            health_status,
            risk_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            started_at,
            finished_at,
            raw_csv_path,
            pred["prediction"],
            pred["explainable"],
            pred["suggestion"],
            pred["hr"],
            pred["sbp"],
            pred["dbp"],
            pred["ptt"],
            pred["hrv_rmssd"],
            pred["spo2"],
            pred["breathing_rate"],
            pred["hr_status"],
            pred["bp_status"],
            pred["health_status"],
            pred["risk_score"],
        ),
    )
    scan_id = cur.lastrowid
    conn.commit()
    conn.close()

    mark_scan_completed(scan_id)

    return jsonify({
        "success": True,
        "scan_id": scan_id,
        "prediction": pred["prediction"],
        "hr": pred["hr"],
        "sbp": pred["sbp"],
        "dbp": pred["dbp"],
        "risk_score": pred["risk_score"],
    })


@app.route("/api/scan/status", methods=["GET"])
def api_scan_status():
    return jsonify(pop_queued_scan())


@app.route("/runScan.php", methods=["GET"])
def php_scan_status():
    action = request.args.get("action", "")
    if action == "status":
        return jsonify(pop_queued_scan())
    return jsonify({"run_scan": 0, "patient_id": 0})


@app.route("/api/scan/result", methods=["POST"])
@app.route("/updateResult.php", methods=["POST"])
def receive_result():
    patient_id = parse_int(get_request_value("patient_id"), default=0)
    if patient_id <= 0:
        return jsonify({"success": False, "error": "Valid patient_id is required"}), 400

    started_at = get_request_value("started_at") or now_iso()
    finished_at = get_request_value("finished_at") or now_iso()
    prediction = get_request_value("prediction")
    explainable = get_request_value("explainable") or get_request_value("explainable_ai_output")
    suggestion = get_request_value("suggestion")

    hr = parse_float(get_request_value("hr"), default=None)
    sbp = parse_float(get_request_value("sbp"), default=None)
    dbp = parse_float(get_request_value("dbp"), default=None)
    ptt = parse_float(get_request_value("ptt"), default=None)
    hrv_rmssd = parse_float(get_request_value("hrv_rmssd"), default=None)
    spo2 = parse_float(get_request_value("spo2"), default=None)
    breathing_rate = parse_float(get_request_value("breathing_rate"), default=None)
    hr_status = get_request_value("hr_status")
    bp_status = get_request_value("bp_status")
    health_status = get_request_value("health_status")
    risk_score = parse_int(get_request_value("risk_score"), default=None)

    raw_csv_path = get_request_value("raw_csv")
    uploaded_csv = request.files.get("raw_csv")
    if uploaded_csv and uploaded_csv.filename:
        safe_name = secure_filename(uploaded_csv.filename)
        if not safe_name.lower().endswith(".csv"):
            safe_name = f"{safe_name}.csv"
        file_name = f"scan_{int(time.time())}_{patient_id}_{safe_name}"
        abs_path = os.path.join(DATA_DIR, file_name)
        uploaded_csv.save(abs_path)
        raw_csv_path = os.path.relpath(abs_path, APP_ROOT).replace("\\", "/")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO scans (
            patient_id,
            started_at,
            finished_at,
            raw_csv,
            prediction,
            explainable,
            suggestion,
            hr,
            sbp,
            dbp,
            ptt,
            hrv_rmssd,
            spo2,
            breathing_rate,
            hr_status,
            bp_status,
            health_status,
            risk_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            started_at,
            finished_at,
            raw_csv_path,
            prediction,
            explainable,
            suggestion,
            hr,
            sbp,
            dbp,
            ptt,
            hrv_rmssd,
            spo2,
            breathing_rate,
            hr_status,
            bp_status,
            health_status,
            risk_score,
        ),
    )
    scan_id = cur.lastrowid
    conn.commit()
    conn.close()

    mark_scan_completed(scan_id)
    return jsonify({"success": True, "scan_id": scan_id})


@app.route("/api/scan/live", methods=["GET"])
def get_live_scan():
    patient_id = parse_int(request.args.get("patient_id"), default=0)
    conn = get_db_connection()
    if patient_id > 0:
        row = conn.execute(
            """
            SELECT
                s.id,
                s.patient_id,
                p.name AS patient_name,
                s.started_at,
                s.finished_at,
                s.prediction,
                s.explainable,
                s.suggestion,
                s.hr,
                s.sbp,
                s.dbp,
                s.ptt,
                s.hrv_rmssd,
                s.spo2,
                s.breathing_rate,
                s.hr_status,
                s.bp_status,
                s.health_status,
                s.risk_score
            FROM scans s
            LEFT JOIN patients p ON p.id = s.patient_id
            WHERE s.patient_id = ?
            ORDER BY s.id DESC
            LIMIT 1
            """,
            (patient_id,),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT
                s.id,
                s.patient_id,
                p.name AS patient_name,
                s.started_at,
                s.finished_at,
                s.prediction,
                s.explainable,
                s.suggestion,
                s.hr,
                s.sbp,
                s.dbp,
                s.ptt,
                s.hrv_rmssd,
                s.spo2,
                s.breathing_rate,
                s.hr_status,
                s.bp_status,
                s.health_status,
                s.risk_score
            FROM scans s
            LEFT JOIN patients p ON p.id = s.patient_id
            ORDER BY s.id DESC
            LIMIT 1
            """
        ).fetchone()
    conn.close()

    running = current_running_state()["running"]
    if row:
        return jsonify({
            "scan": row_to_dict(row),
            "status": "completed" if not running else "pending",
        })

    return jsonify({"scan": None, "status": "pending"})


@app.route("/api/scan/running", methods=["GET"])
def get_scan_running():
    return jsonify(current_running_state())


@app.errorhandler(Exception)
def handle_error(err):
    return jsonify({"success": False, "error": str(err)}), 500


ensure_schema()
backfill_patients_from_scans()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
