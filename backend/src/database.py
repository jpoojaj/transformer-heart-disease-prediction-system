import sqlite3
import json
import hashlib
from datetime import datetime

DB_PATH = 'heart_health.db'


def make_hashed_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _get_connection():
    return sqlite3.connect(DB_PATH)


def init_database():
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('PRAGMA foreign_keys = ON')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        name TEXT,
        role TEXT CHECK(role IN ('patient', 'doctor', 'admin')) NOT NULL DEFAULT 'patient',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        password_hash TEXT NOT NULL,
        email TEXT,
        phone_no TEXT,
        age INTEGER,
        is_active INTEGER DEFAULT 1
    )
    ''')

    cursor.execute("PRAGMA table_info(users)")
    user_columns = [row[1] for row in cursor.fetchall()]
    if 'phone_no' not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN phone_no TEXT")
    if 'age' not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN age INTEGER")
    if 'is_active' not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1")
    if 'notify_doctor' not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN notify_doctor INTEGER DEFAULT 0")

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        input_json TEXT,
        risk_score REAL,
        risk_category TEXT,
        model_name TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS explanations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER NOT NULL,
        method TEXT,
        explanation_json TEXT,
        summary_text TEXT,
        FOREIGN KEY(assessment_id) REFERENCES assessments(id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS doctor_patient (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id INTEGER NOT NULL,
        patient_id INTEGER NOT NULL,
        FOREIGN KEY(doctor_id) REFERENCES users(id),
        FOREIGN KEY(patient_id) REFERENCES users(id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS doctor_patient_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id INTEGER NOT NULL,
        patient_id INTEGER NOT NULL,
        note TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(doctor_id) REFERENCES users(id),
        FOREIGN KEY(patient_id) REFERENCES users(id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        assessment_id INTEGER NOT NULL,
        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        type TEXT,
        message TEXT,
        resolved INTEGER DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(assessment_id) REFERENCES assessments(id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS assessment_explainability (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER NOT NULL,
        feature TEXT NOT NULL,
        impact REAL NOT NULL,
        direction TEXT NOT NULL,
        explanation TEXT NOT NULL,
        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(assessment_id) REFERENCES assessments(id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ecg_assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        predicted_label TEXT,
        confidence REAL,
        probabilities_json TEXT,
        patient_summary TEXT,
        doctor_summary TEXT,
        FOREIGN KEY(assessment_id) REFERENCES assessments(id),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS lifestyle_coach_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        doctor_id INTEGER,
        plan_name TEXT NOT NULL,
        description TEXT,
        plan_duration_days INTEGER,
        target_activity_hours REAL,
        target_weight REAL,
        target_bp_systolic INTEGER,
        target_bp_diastolic INTEGER,
        status TEXT CHECK(status IN ('active', 'completed', 'cancelled')) DEFAULT 'active',
        progress_notes TEXT,
        streak_count INTEGER DEFAULT 0,
        longest_streak INTEGER DEFAULT 0,
        completed_count INTEGER DEFAULT 0,
        last_completion_date TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(doctor_id) REFERENCES users(id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS telehealth_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        doctor_id INTEGER NOT NULL,
        sender_id INTEGER NOT NULL,
        message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(patient_id) REFERENCES users(id),
        FOREIGN KEY(doctor_id) REFERENCES users(id),
        FOREIGN KEY(sender_id) REFERENCES users(id)
    )
    ''')

    # Migration: Add plan_duration_days column if it doesn't exist
    cursor.execute("PRAGMA table_info(lifestyle_coach_plans)")
    lifestyle_columns = [row[1] for row in cursor.fetchall()]
    if 'plan_duration_days' not in lifestyle_columns:
        cursor.execute("ALTER TABLE lifestyle_coach_plans ADD COLUMN plan_duration_days INTEGER")

    conn.commit()
    conn.close()


def add_user(username, password, name=None, role='patient', email=None, phone_no=None, age=None):
    if not (isinstance(password, str) and len(password) == 6 and password.isdigit()):
        raise ValueError("Password must be a 6-digit number")

    password_hash = make_hashed_password(password)
    conn = _get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
        INSERT INTO users (username, name, role, password_hash, email, phone_no, age)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (username, name, role, password_hash, email, phone_no, age))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        if 'UNIQUE constraint failed' in str(exc):
            raise ValueError("Username already exists")
        raise
    finally:
        conn.close()


def get_user_by_username(username):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user


def authenticate_user(username, password):
    user = get_user_by_username(username)
    if not user:
        return None
    return user if user[5] == make_hashed_password(password) else None


def get_user_by_id(user_id):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user


def save_assessment(user_id, risk_score, risk_category, model_name, input_json, trigger_alerts=True):
    conn = _get_connection()
    cursor = conn.cursor()






    

    cursor.execute('''
    INSERT INTO assessments (user_id, input_json, risk_score, risk_category, model_name)
    VALUES (?, ?, ?, ?, ?)
    ''', (user_id, input_json, risk_score, risk_category, model_name))

    assessment_id = cursor.lastrowid
    conn.commit()
    conn.close()

    if trigger_alerts:
        check_and_create_alerts(user_id, assessment_id, risk_score)
        try:
            check_and_create_improvement_alerts(user_id, assessment_id, risk_score)
        except Exception:
            pass
    return assessment_id


def save_ecg_assessment(user_id, predicted_label, confidence, probabilities, patient_summary=None, doctor_summary=None):
    if user_id is None or probabilities is None:
        return None

    if isinstance(probabilities, dict):
        prob_map = {str(k): float(v) for k, v in probabilities.items()}
    else:
        prob_map = {str(idx): float(val) for idx, val in enumerate(probabilities)}

    max_prob = max(prob_map.values()) if prob_map else 0.0
    payload = json.dumps({
        'type': 'ecg',
        'prediction': predicted_label,
        'confidence': confidence,
        'probabilities': prob_map
    })

    assessment_id = save_assessment(
        user_id=user_id,
        risk_score=float(max_prob) * 100.0,
        risk_category=predicted_label,
        model_name='ECG Ensemble',
        input_json=payload,
        trigger_alerts=False
    )

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        INSERT INTO ecg_assessments (assessment_id, user_id, predicted_label, confidence, probabilities_json, patient_summary, doctor_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            assessment_id,
            user_id,
            predicted_label,
            confidence,
            json.dumps(prob_map),
            patient_summary,
            doctor_summary
        )
    )
    conn.commit()
    conn.close()
    
    # Add alert for critical ECG findings
    if confidence > 0.30 and predicted_label not in ['Normal']:
        # High alert for any abnormal finding exceeding 30% confidence
        alert_message = f"Critical ECG finding detected: {predicted_label} with {confidence*100:.1f}% confidence"
        insert_alert(user_id, assessment_id, 'high_risk', alert_message)
    elif confidence > 0.25 and predicted_label not in ['Normal']:
        # Medium confidence abnormal finding - create alert
        alert_message = f"Significant ECG finding detected: {predicted_label} with {confidence*100:.1f}% confidence"
        insert_alert(user_id, assessment_id, 'risk_increase', alert_message)
    
    return assessment_id


def get_ecg_assessments_for_user(user_id, limit=25):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT id, assessment_id, ts, predicted_label, confidence, probabilities_json, patient_summary, doctor_summary
        FROM ecg_assessments
        WHERE user_id = ?
        ORDER BY ts DESC
        LIMIT ?
        ''',
        (user_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_ecg_assessments_for_doctor(doctor_id, limit=50):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT ea.id, ea.assessment_id, ea.ts, ea.predicted_label, ea.confidence, ea.probabilities_json,
               ea.patient_summary, ea.doctor_summary,
               u.id, u.name, u.username
        FROM ecg_assessments ea
        JOIN doctor_patient dp ON dp.patient_id = ea.user_id
        JOIN users u ON u.id = ea.user_id
        WHERE dp.doctor_id = ?
        ORDER BY ea.ts DESC
        LIMIT ?
        ''',
        (doctor_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_recent_ecg_assessments(limit=50):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT ea.id, u.username, u.name, ea.ts, ea.predicted_label, ea.confidence
        FROM ecg_assessments ea
        JOIN users u ON u.id = ea.user_id
        ORDER BY ea.ts DESC
        LIMIT ?
        ''',
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_latest_assessment_id(user_id):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT id FROM assessments
        WHERE user_id = ?
        ORDER BY ts DESC, id DESC
        LIMIT 1
        ''',
        (user_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def ensure_chat_assessment(user_id):
    latest_id = get_latest_assessment_id(user_id)
    if latest_id:
        return latest_id
    placeholder_payload = json.dumps({'type': 'chat_placeholder'})
    return save_assessment(
        user_id=user_id,
        risk_score=0.0,
        risk_category='Chat Placeholder',
        model_name='Chat',
        input_json=placeholder_payload,
        trigger_alerts=False
    )


def add_chat_message(patient_id, sender_role, sender_name, message_text):
    if not patient_id or not message_text.strip():
        return None
    assessment_id = ensure_chat_assessment(patient_id)
    payload = json.dumps({
        'sender_role': sender_role,
        'sender_name': sender_name,
        'message': message_text.strip()
    })
    alert_id = insert_alert(patient_id, assessment_id, 'chat_message', payload)
    return alert_id


def get_chat_messages_for_patient(patient_id, limit=200):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT id, ts, message
        FROM alerts
        WHERE user_id = ? AND type = 'chat_message'
        ORDER BY ts ASC
        LIMIT ?
        ''',
        (patient_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def save_explanation(assessment_id, method, explanation_json, summary_text):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO explanations (assessment_id, method, explanation_json, summary_text)
    VALUES (?, ?, ?, ?)
    ''', (assessment_id, method, json.dumps(explanation_json) if isinstance(explanation_json, dict) else explanation_json, summary_text))
    explanation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return explanation_id


def insert_alert(user_id, assessment_id, alert_type, message):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO alerts (user_id, assessment_id, type, message)
    VALUES (?, ?, ?, ?)
    ''', (user_id, assessment_id, alert_type, message))
    alert_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return alert_id


def check_and_create_alerts(user_id, assessment_id, risk_score, threshold=10.0):
    if risk_score is None:
        return

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT risk_score FROM assessments
    WHERE user_id = ? AND risk_score IS NOT NULL
    ORDER BY ts DESC, id DESC LIMIT 3
    ''', (user_id,))
    rows = cursor.fetchall()
    conn.close()

    scores = [row[0] for row in rows]
    alert_needed = False

    if len(scores) >= 2 and (scores[0] - scores[1]) >= threshold:
        alert_needed = True
    elif len(scores) == 3 and scores[0] > scores[1] > scores[2]:
        alert_needed = True

    if alert_needed:
        previous = scores[1] if len(scores) >= 2 else None
        diff_msg = f" from {previous:.1f}%" if previous is not None else ''
        insert_alert(
            user_id,
            assessment_id,
            'risk_increase',
            f"Risk increased to {risk_score:.1f}%{diff_msg}"
        )

def _get_user_notify_doctor(user_id):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT notify_doctor FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()
    return bool(row[0]) if row and row[0] is not None else False

def set_user_notify_doctor(user_id, notify=True):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET notify_doctor = ? WHERE id = ?', (1 if notify else 0, user_id))
    conn.commit()
    conn.close()

def check_and_create_improvement_alerts(user_id, assessment_id, risk_score, threshold=10.0):
    if risk_score is None:
        return
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT id, risk_score, input_json, ts
    FROM assessments
    WHERE user_id = ? AND risk_score IS NOT NULL
    ORDER BY ts DESC, id DESC LIMIT 2
    ''', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    if len(rows) < 2:
        return
    latest_id, latest_score, latest_input_json, latest_ts = rows[0]
    prev_id, prev_score, prev_input_json, prev_ts = rows[1]
    if prev_score is None:
        return
    decreased = prev_score - latest_score >= threshold
    bmi_drop = False
    try:
        latest = json.loads(latest_input_json) if latest_input_json else {}
        prev = json.loads(prev_input_json) if prev_input_json else {}
        latest_bmi_cat = latest.get('bmi_category')
        prev_bmi_cat = prev.get('bmi_category')
        if isinstance(latest_bmi_cat, int) and isinstance(prev_bmi_cat, int):
            bmi_drop = prev_bmi_cat is not None and latest_bmi_cat is not None and latest_bmi_cat < prev_bmi_cat
    except Exception:
        bmi_drop = False
    if decreased or bmi_drop:
        if _get_user_notify_doctor(user_id):
            msg = 'Risk decreased significantly' if decreased else 'BMI category improved'
            insert_alert(user_id, assessment_id, 'progress_milestone', msg)


def save_explainability(assessment_id, contributions):
    if not contributions:
        return
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.executemany(
        '''INSERT INTO assessment_explainability (assessment_id, feature, impact, direction, explanation)
           VALUES (?, ?, ?, ?, ?)''',
        [
            (
                assessment_id,
                contrib.get('feature'),
                contrib.get('impact'),
                contrib.get('direction'),
                contrib.get('explanation'),
            )
            for contrib in contributions
        ]
    )
    conn.commit()
    conn.close()


def get_latest_explainability(user_id, limit_features=5):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT e.feature, e.impact, e.direction, e.explanation
        FROM assessment_explainability e
        JOIN assessments a ON a.id = e.assessment_id
        WHERE a.user_id = ?
        ORDER BY a.ts DESC, e.id ASC
        LIMIT ?
        ''',
        (user_id, limit_features)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            'feature': row[0],
            'impact': row[1],
            'direction': row[2],
            'explanation': row[3],
        }
        for row in rows
    ]


def get_explainability_trend(user_id, limit_assessments=5):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT a.ts, e.feature, e.impact, e.direction
        FROM assessment_explainability e
        JOIN assessments a ON a.id = e.assessment_id
        WHERE a.user_id = ?
        ORDER BY a.ts DESC, e.id ASC
        LIMIT ?
        ''',
        (user_id, limit_assessments * 5)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_unresolved_alerts_for_user(user_id):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT id, user_id, assessment_id, ts, type, message
    FROM alerts
    WHERE user_id = ? AND resolved = 0 AND type != 'chat_message'
    ORDER BY ts DESC
    ''', (user_id,))
    alerts = cursor.fetchall()
    conn.close()
    return alerts


def get_unresolved_alerts_for_doctor(doctor_id):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT a.id, a.user_id, a.assessment_id, a.ts, a.type, a.message
    FROM alerts a
    JOIN doctor_patient dp ON dp.patient_id = a.user_id
    WHERE dp.doctor_id = ? AND a.resolved = 0 AND a.type != 'chat_message'
    ORDER BY a.ts DESC
    ''', (doctor_id,))
    alerts = cursor.fetchall()
    conn.close()
    return alerts


def get_unresolved_alerts_for_admin():
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT id, user_id, assessment_id, ts, type, message
    FROM alerts
    WHERE resolved = 0 AND type != 'chat_message'
    ORDER BY ts DESC
    ''')
    alerts = cursor.fetchall()
    conn.close()
    return alerts


def resolve_alert(alert_id):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE alerts SET resolved = 1 WHERE id = ?', (alert_id,))
    conn.commit()
    conn.close()


def get_most_critical_alert_for_doctor(doctor_id):
    """Get the most critical unresolved alert for a doctor's patients."""
    conn = _get_connection()
    cursor = conn.cursor()
    # Prioritize high_risk and risk_increase alerts, order by timestamp desc
    cursor.execute('''
    SELECT a.id, a.user_id, a.assessment_id, a.ts, a.type, a.message, u.username, u.name
    FROM alerts a
    JOIN users u ON u.id = a.user_id
    JOIN doctor_patient dp ON dp.patient_id = a.user_id
    WHERE dp.doctor_id = ? AND a.resolved = 0
    ORDER BY 
        CASE 
            WHEN a.type = 'high_risk' THEN 1
            WHEN a.type = 'risk_increase' THEN 2
            WHEN a.type = 'reminder' THEN 3
            WHEN a.type = 'patient_message' THEN 4
            ELSE 5
        END,
        a.ts DESC
    LIMIT 1
    ''', (doctor_id,))
    alert = cursor.fetchone()
    conn.close()
    return alert


def acknowledge_alert(alert_id):
    """Mark an alert as acknowledged by the patient."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE alerts SET resolved = 1 WHERE id = ?', (alert_id,))
    conn.commit()
    conn.close()


def get_patients_for_doctor(doctor_id):
    """Return patients linked to the doctor with latest assessment info."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            u.id,
            u.username,
            u.name,
            u.age,
            (
                SELECT risk_score
                FROM assessments a
                WHERE a.user_id = u.id
                ORDER BY a.ts DESC, a.id DESC
                LIMIT 1
            ) AS risk_score,
            (
                SELECT risk_category
                FROM assessments a
                WHERE a.user_id = u.id
                ORDER BY a.ts DESC, a.id DESC
                LIMIT 1
            ) AS risk_category,
            (
                SELECT ts
                FROM assessments a
                WHERE a.user_id = u.id
                ORDER BY a.ts DESC, a.id DESC
                LIMIT 1
            ) AS last_assessment_ts,
            (
                SELECT COUNT(*)
                FROM alerts al
                WHERE al.user_id = u.id AND al.resolved = 0
            ) AS unresolved_alerts
        FROM doctor_patient dp
        JOIN users u ON u.id = dp.patient_id
        WHERE dp.doctor_id = ?
        ORDER BY
            CASE WHEN last_assessment_ts IS NULL THEN 1 ELSE 0 END,
            last_assessment_ts DESC,
            u.name
        """,
        (doctor_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    patients = []
    for row in rows:
        (
            patient_id,
            username,
            name,
            age,
            risk_score,
            risk_category,
            last_assessment_ts,
            unresolved_alerts,
        ) = row
        patients.append(
            {
                "id": patient_id,
                "username": username,
                "name": name,
                "age": age,
                "risk_score": risk_score,
                "risk_category": risk_category,
                "last_assessment_ts": last_assessment_ts,
                "unresolved_alerts": unresolved_alerts,
            }
        )
    return patients


def link_patient_to_doctor(patient_id, doctor_username):
    doctor = get_user_by_username(doctor_username)
    if not doctor:
        return False, "Doctor username not found."
    doctor_id, _, doctor_name, doctor_role = doctor[0], doctor[1], doctor[2], doctor[3]
    if doctor_role != 'doctor':
        return False, "Selected user is not registered as a doctor."
    if doctor_id == patient_id:
        return False, "You cannot link to yourself."

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT 1 FROM doctor_patient WHERE doctor_id = ? AND patient_id = ?',
        (doctor_id, patient_id)
    )
    exists = cursor.fetchone()
    if exists:
        conn.close()
        return False, "This doctor is already linked to your account."

    cursor.execute(
        'INSERT INTO doctor_patient (doctor_id, patient_id) VALUES (?, ?)',
        (doctor_id, patient_id)
    )
    conn.commit()
    conn.close()
    return True, f"Linked with Dr. {doctor_name or doctor[1]}."


def get_doctors_for_patient(patient_id):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT dp.id, u.id, u.username, u.name, u.email, u.phone_no
        FROM doctor_patient dp
        JOIN users u ON u.id = dp.doctor_id
        WHERE dp.patient_id = ?
        ORDER BY u.name
        ''',
        (patient_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    doctors = []
    for link_id, doctor_id, username, name, email, phone in rows:
        doctors.append(
            {
                'link_id': link_id,
                'doctor_id': doctor_id,
                'username': username,
                'name': name,
                'email': email,
                'phone_no': phone,
            }
        )
    return doctors


def get_all_doctor_patient_links():
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT dp.id, d.name, d.username, p.name, p.username
        FROM doctor_patient dp
        JOIN users d ON d.id = dp.doctor_id
        JOIN users p ON p.id = dp.patient_id
        ORDER BY d.name, p.name
        '''
    )
    rows = cursor.fetchall()
    conn.close()
    links = []
    for row in rows:
        link_id, doctor_name, doctor_username, patient_name, patient_username = row
        links.append(
            {
                'link_id': link_id,
                'doctor_name': doctor_name,
                'doctor_username': doctor_username,
                'patient_name': patient_name,
                'patient_username': patient_username,
            }
        )
    return links


def add_link_by_ids(doctor_username, patient_username):
    doctor = get_user_by_username(doctor_username)
    patient = get_user_by_username(patient_username)
    if not doctor or not patient:
        return False, "Invalid doctor or patient username"
    if doctor[3] != 'doctor' or patient[3] != 'patient':
        return False, "Roles do not match doctor/patient"
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM doctor_patient WHERE doctor_id = ? AND patient_id = ?', (doctor[0], patient[0]))
    exists = cursor.fetchone()
    if exists:
        conn.close()
        return False, "Link already exists"
    cursor.execute('INSERT INTO doctor_patient (doctor_id, patient_id) VALUES (?, ?)', (doctor[0], patient[0]))
    conn.commit()
    conn.close()
    return True, "Link created"


def unlink_doctor_patient(link_id):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM doctor_patient WHERE id = ?', (link_id,))
    conn.commit()
    conn.close()


def delete_user(user_id):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM doctor_patient_notes WHERE doctor_id = ? OR patient_id = ?', (user_id, user_id))
    cursor.execute('DELETE FROM doctor_patient WHERE doctor_id = ? OR patient_id = ?', (user_id, user_id))
    cursor.execute('DELETE FROM alerts WHERE user_id = ?', (user_id,))
    cursor.execute('DELETE FROM explanations WHERE assessment_id IN (SELECT id FROM assessments WHERE user_id = ?)', (user_id,))
    cursor.execute('DELETE FROM assessments WHERE user_id = ?', (user_id,))
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()


def get_all_users():
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''SELECT id, username, name, role, email, phone_no, age, is_active, created_at FROM users ORDER BY created_at DESC'''
    )
    users = cursor.fetchall()
    conn.close()
    return users


def update_user_role(user_id, new_role):
    if new_role not in ('patient', 'doctor', 'admin'):
        raise ValueError('Invalid role')
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET role = ? WHERE id = ?', (new_role, user_id))
    conn.commit()
    conn.close()


def set_user_active_status(user_id, is_active=True):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET is_active = ? WHERE id = ?', (1 if is_active else 0, user_id))
    conn.commit()
    conn.close()


def get_system_counts():
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'patient'")
    patients = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'doctor'")
    doctors = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    admins = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM assessments')
    assessments = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM alerts WHERE resolved = 0')
    unresolved_alerts = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM doctor_patient')
    links = cursor.fetchone()[0]
    conn.close()
    return {
        'total_users': total_users,
        'patients': patients,
        'doctors': doctors,
        'admins': admins,
        'assessments': assessments,
        'unresolved_alerts': unresolved_alerts,
        'links': links,
    }


def get_recent_assessments(limit=50):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT a.id, u.username, u.name, a.ts, a.risk_score, a.risk_category, a.model_name
        FROM assessments a
        JOIN users u ON u.id = a.user_id
        ORDER BY a.ts DESC, a.id DESC
        LIMIT ?
        ''',
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_alerts(limit=200):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT a.id, u.username, u.name, a.ts, a.type, a.message, a.resolved
        FROM alerts a
        JOIN users u ON u.id = a.user_id
        ORDER BY a.ts DESC
        LIMIT ?
        ''',
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def log_audit_action(admin_id, action, details=''):
    conn = _get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO audit_logs (admin_id, action, details) VALUES (?, ?, ?)',
            (admin_id, action, details)
        )
        conn.commit()
    except sqlite3.OperationalError as exc:
        if 'no such table: audit_logs' in str(exc).lower():
            conn.close()
            init_database()
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO audit_logs (admin_id, action, details) VALUES (?, ?, ?)',
                (admin_id, action, details)
            )
            conn.commit()
        else:
            raise
    finally:
        conn.close()


def get_recent_audit_logs(limit=50):
    conn = _get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            SELECT al.id, u.username, al.action, al.details, al.created_at
            FROM audit_logs al
            LEFT JOIN users u ON u.id = al.admin_id
            ORDER BY al.created_at DESC, al.id DESC
            LIMIT ?
            ''',
            (limit,)
        )
        logs = cursor.fetchall()
    except sqlite3.OperationalError as exc:
        if 'no such table: audit_logs' in str(exc).lower():
            conn.close()
            init_database()
            return []
        raise
    else:
        conn.close()
        return logs


def add_patient_note(doctor_id, patient_id, note):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO doctor_patient_notes (doctor_id, patient_id, note) VALUES (?, ?, ?)''',
        (doctor_id, patient_id, note)
    )
    conn.commit()
    conn.close()


def get_patient_notes(doctor_id, patient_id, limit=10):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT note, created_at FROM doctor_patient_notes
        WHERE doctor_id = ? AND patient_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        ''',
        (doctor_id, patient_id, limit)
    )
    notes = cursor.fetchall()
    conn.close()
    return notes


def get_patient_history(user_id):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT id, ts, input_json, risk_score, risk_category, model_name
    FROM assessments WHERE user_id = ? ORDER BY ts DESC
    ''', (user_id,))
    assessments = cursor.fetchall()

    cursor.execute('''
    SELECT id, assessment_id, ts, type, message, resolved
    FROM alerts WHERE user_id = ? ORDER BY ts DESC
    ''', (user_id,))
    alerts = cursor.fetchall()

    cursor.execute('''
    SELECT e.id, e.assessment_id, e.method, e.explanation_json, e.summary_text
    FROM explanations e
    JOIN assessments a ON a.id = e.assessment_id
    WHERE a.user_id = ?
    ''', (user_id,))
    explanations = cursor.fetchall()

    conn.close()
    return {
        'assessments': assessments,
        'alerts': alerts,
        'explanations': explanations
    }


# ===== Lifestyle Coach Functions =====

def add_lifestyle_plan(user_id, plan_name, description, doctor_id=None, 
                       plan_duration_days=None, target_activity_hours=None, 
                       target_weight=None, target_bp_systolic=None, 
                       target_bp_diastolic=None):
    """Create a new lifestyle plan for a patient"""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO lifestyle_coach_plans 
    (user_id, doctor_id, plan_name, description, plan_duration_days,
     target_activity_hours, target_weight, target_bp_systolic, target_bp_diastolic)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, doctor_id, plan_name, description, plan_duration_days,
          target_activity_hours, target_weight, target_bp_systolic, target_bp_diastolic))
    conn.commit()
    plan_id = cursor.lastrowid
    conn.close()
    return plan_id


def get_lifestyle_plans(user_id, status=None):
    """Get lifestyle plans for a patient"""
    conn = _get_connection()
    cursor = conn.cursor()
    if status:
        cursor.execute('''
        SELECT * FROM lifestyle_coach_plans 
        WHERE user_id = ? AND status = ? 
        ORDER BY created_at DESC
        ''', (user_id, status))
    else:
        cursor.execute('''
        SELECT * FROM lifestyle_coach_plans 
        WHERE user_id = ? 
        ORDER BY created_at DESC
        ''', (user_id,))
    plans = cursor.fetchall()
    conn.close()
    return plans


def update_lifestyle_plan(plan_id, status=None, progress_notes=None, 
                         mark_completed=False):
    """Update a lifestyle plan with progress"""
    conn = _get_connection()
    cursor = conn.cursor()
    
    # Get current plan data
    cursor.execute('SELECT * FROM lifestyle_coach_plans WHERE id = ?', (plan_id,))
    plan = cursor.fetchone()
    if not plan:
        conn.close()
        return False
    
    updates = []
    params = []
    
    if status:
        updates.append('status = ?')
        params.append(status)
    
    if progress_notes:
        updates.append('progress_notes = ?')
        params.append(progress_notes)
    
    # Handle streak logic
    if mark_completed:
        current_streak = plan[11]  # streak_count
        longest_streak = plan[12]  # longest_streak
        completed_count = plan[13]  # completed_count
        last_completion = plan[14]  # last_completion_date
        
        # Check if this is a consecutive completion (within 2 days)
        from datetime import datetime, timedelta
        today = datetime.now().date()
        if last_completion:
            last_date = datetime.fromisoformat(last_completion).date()
            days_diff = (today - last_date).days
            if days_diff == 1:
                # Consecutive day
                current_streak += 1
            elif days_diff == 0:
                # Same day, don't increment
                pass
            else:
                # Streak broken (but still count this completion)
                current_streak = 1
        else:
            current_streak = 1
        
        longest_streak = max(longest_streak, current_streak)
        completed_count += 1
        
        updates.append('streak_count = ?')
        params.append(current_streak)
        updates.append('longest_streak = ?')
        params.append(longest_streak)
        updates.append('completed_count = ?')
        params.append(completed_count)
        updates.append('last_completion_date = ?')
        params.append(today.isoformat())
        
        # Check if plan should be completed based on duration
        plan_duration = plan[5]  # plan_duration_days at index 5
        if plan_duration and completed_count >= plan_duration:
            updates.append('status = ?')
            params.append('completed')
    
    updates.append('updated_at = CURRENT_TIMESTAMP')
    params.append(plan_id)
    
    if updates:
        query = f"UPDATE lifestyle_coach_plans SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        conn.commit()
    
    conn.close()
    return True


# ===== Telehealth Messaging Functions =====

def add_telehealth_message(patient_id, doctor_id, sender_id, message):
    """Add a telehealth message between doctor and patient"""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO telehealth_messages (patient_id, doctor_id, sender_id, message)
    VALUES (?, ?, ?, ?)
    ''', (patient_id, doctor_id, sender_id, message))
    conn.commit()
    conn.close()


def get_telehealth_messages(patient_id, doctor_id, limit=50):
    """Get telehealth messages between a doctor and patient"""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT tm.id, tm.sender_id, u.username, u.name, u.role, tm.message, tm.created_at
    FROM telehealth_messages tm
    JOIN users u ON u.id = tm.sender_id
    WHERE tm.patient_id = ? AND tm.doctor_id = ?
    ORDER BY tm.created_at ASC
    LIMIT ?
    ''', (patient_id, doctor_id, limit))
    messages = cursor.fetchall()
    conn.close()
    return messages


if __name__ == '__main__':
    init_database()
