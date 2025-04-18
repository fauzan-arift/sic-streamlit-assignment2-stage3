import streamlit as st
import cv2
import time
import json
import requests
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO
import google.generativeai as genai  # Import Google's Generative AI library

# === Konfigurasi Ubidots ===
BIDOTS_TOKEN = st.secrets["UBIDOTS_TOKEN"]
DEVICE_LABEL = st.secrets["DEVICE_LABEL"]
VARIABLE_CAMERA = st.secrets["VARIABLE_CAMERA"]
VARIABLE_LIGHT = st.secrets["VARIABLE_LIGHT"]
VARIABLE_COUNT = st.secrets["VARIABLE_COUNT"]

BROKER = st.secrets["BROKER"]
PORT = int(st.secrets["PORT"])

# === Gemini API Configuration ===
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# === Init State ===
st.session_state.setdefault("log", [])
st.session_state.setdefault("camera_on", False)
st.session_state.setdefault("lamp", 0)
st.session_state.setdefault("count", 0)
st.session_state.setdefault("last_sent", 0)
st.session_state.setdefault("camera_option", "Laptop Webcam")
st.session_state.setdefault("esp32_url", "")
st.session_state.setdefault("stop_clicks", 0)
st.session_state.setdefault("activity_history", [])
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("ai_enabled", False)

# === Load YOLOv8 Model ===
if "model" not in st.session_state:
    st.session_state.model = YOLO("yolov8n.pt")
model = st.session_state.model


# === MQTT Setup ===
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        st.session_state.log.append("✅ Terhubung ke Ubidots")
    else:
        st.session_state.log.append(f"❌ Gagal terhubung, kode: {rc}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload)
        val = payload.get("value", 0)
        st.session_state.camera_on = bool(val)
    except Exception as e:
        st.session_state.log.append(f"Error MQTT: {e}")


def setup_mqtt():
    client = mqtt.Client()
    client.username_pw_set(UBIDOTS_TOKEN, "")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.subscribe(f"/v2.0/devices/{DEVICE_LABEL}/{VARIABLE_CAMERA}")
    client.loop_start()
    return client


st.session_state.setdefault("client", setup_mqtt())


# === Fungsi Kirim ke Ubidots ===
def send_ubidots(var, val):
    try:
        topic = f"/v2.0/devices/{DEVICE_LABEL}/{var}"
        payload = json.dumps({"value": val})
        st.session_state.client.publish(topic, payload)
        st.session_state.log.append(f"📤 Kirim: {var} = {val}")

        # Log activity for AI summary
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        activity = f"[{timestamp}] Kirim {var}={val} ke Ubidots"
        st.session_state.activity_history.append(activity)
    except Exception as e:
        st.session_state.log.append(f"❌ Gagal kirim ke Ubidots: {e}")


# === Setup Gemini Model ===
def setup_gemini_model():
    # Create a Gemini model instance
    # Choose the model based on your needs (gemini-pro is text-only)
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model


if "gemini_model" not in st.session_state:
    try:
        st.session_state.gemini_model = setup_gemini_model()
    except Exception as e:
        st.session_state.log.append(f"❌ Error saat menyiapkan Gemini: {str(e)}")


# === Extract IP from user input ===
def extract_ip_address(text):
    import re
    # Pattern for IPv4 address with optional port (e.g., 192.168.1.1:81)
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b'
    matches = re.findall(ip_pattern, text)
    if matches:
        return matches[0]
    return None


# === Validate IP address ===
def is_valid_ip_address(ip):
    import re
    # Basic validation for IPv4 address format
    pattern = r'^(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?$'
    if not re.match(pattern, ip):
        return False

    # Check each octet is in valid range (0-255)
    parts = ip.split(':')[0].split('.')
    return all(0 <= int(part) <= 255 for part in parts)


# === Fungsi AI Assistant dengan Gemini ===
def generate_ai_response(prompt):
    try:
        # Create context with the activity history
        context = """
        Kamu adalah asisten AI untuk sistem IoT. Kamu dapat membantu:
        1. Memberikan ringkasan aktivitas sistem
        2. Mengontrol perangkat IoT seperti kamera dan lampu
        3. Memberikan informasi tentang jumlah orang terdeteksi
        4. Memberikan saran untuk pengaturan sistem
        
        Perangkat yang tersedia:
        - Kamera (camera): on/off
        - Lampu (lamp): on/off
        - Sensor jumlah orang (people)
        """

        # Add activity history to context
        if st.session_state.activity_history:
            context += "\n\nRiwayat aktivitas terbaru:\n"
            for activity in st.session_state.activity_history[-10:]:
                context += f"- {activity}\n"

        # Setup chat history for context
        chat_history = []
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history[-5:]:  # Last 5 messages for context
                # Create content properly according to Gemini's requirements
                content = [{"text": msg["content"]}]
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({"role": role, "parts": content})

        # Add the latest context and prompt
        messages = [
            {"role": "user", "parts": [{"text": context}]},
            {"role": "model", "parts": [{"text": "Saya mengerti konteks sistem IoT dan siap membantu."}]},
        ]

        # Add chat history if available
        if chat_history:
            messages.extend(chat_history)

        # Add the current prompt
        messages.append({"role": "user", "parts": [{"text": prompt}]})

        # Generate response
        response = st.session_state.gemini_model.generate_content(messages)

        return response.text
    except Exception as e:
        return f"❌ Error AI: {str(e)}"


# === Generate Activity Summary ===
def generate_activity_summary():
    # This function generates a summary of recent system activities
    try:
        prompt = "Berikan ringkasan aktivitas sistem berdasarkan riwayat aktivitas terbaru."
        return generate_ai_response(prompt)
    except Exception as e:
        return f"❌ Error saat menghasilkan ringkasan: {str(e)}"


# === Process AI Commands ===
def process_ai_command(response, user_input):
    # Check for commands in the user input and AI response
    commands = {
        "camera_on": ["nyalakan kamera", "hidupkan kamera", "aktifkan kamera"],
        "camera_off": ["matikan kamera", "nonaktifkan kamera"],
        "lamp_on": ["nyalakan lampu", "hidupkan lampu"],
        "lamp_off": ["matikan lampu", "padamkan lampu"],
        "set_esp32_ip": ["gunakan ip", "alamat ip", "hubungkan ke ip", "gunakan esp32", "pakai esp32"]
    }

    combined_text = (user_input + " " + response).lower()
    executed_commands = []

    # Check if there's an IP address in the user input
    ip_address = extract_ip_address(user_input)
    if ip_address and is_valid_ip_address(ip_address):
        st.session_state.esp32_url = ip_address
        st.session_state.camera_option = "ESP32-CAM"
        executed_commands.append(f"URL ESP32-CAM diatur ke {ip_address}")
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        st.session_state.activity_history.append(f"[{timestamp}] URL ESP32-CAM diatur ke {ip_address}")
    elif any(keyword in user_input.lower() for keyword in ["ip", "alamat", "esp32"]):
        # Detected IP-related keywords but couldn't find valid IP
        executed_commands.append("Format IP address tidak valid. Gunakan format: 192.168.1.100")

    # Check for camera and other commands
    for cmd, triggers in commands.items():
        for trigger in triggers:
            if trigger in combined_text:
                if cmd == "camera_on" and not st.session_state.camera_on:
                    st.session_state.camera_on = True
                    send_ubidots(VARIABLE_CAMERA, 1)
                    executed_commands.append("Kamera dinyalakan")
                elif cmd == "camera_off" and st.session_state.camera_on:
                    st.session_state.camera_on = False
                    send_ubidots(VARIABLE_CAMERA, 0)
                    executed_commands.append("Kamera dimatikan")
                elif cmd == "lamp_on" and st.session_state.lamp == 0:
                    st.session_state.lamp = 1
                    send_ubidots(VARIABLE_LIGHT, 1)
                    executed_commands.append("Lampu dinyalakan")
                elif cmd == "lamp_off" and st.session_state.lamp == 1:
                    st.session_state.lamp = 0
                    send_ubidots(VARIABLE_LIGHT, 0)
                    executed_commands.append("Lampu dimatikan")
                break

    return executed_commands


# === Fungsi Streaming ESP32-CAM MJPEG ===
def process_esp32_frame(frame, model):
    """Process ESP32-CAM frame for human detection"""
    try:
        # Pre-process frame
        frame = cv2.resize(frame, (416, 416))
        
        # Run detection
        results = model.predict(
            frame,
            verbose=False,
            conf=0.45,    # Lower confidence threshold for ESP32-CAM
            iou=0.45,
            agnostic_nms=True,
            max_det=10    # Limit detections for performance
        )
        
        # Get person detections only
        boxes = results[0].boxes
        people_boxes = [box for box in boxes if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.45]
        
        return people_boxes, len(people_boxes)
        
    except Exception as e:
        st.session_state.log.append(f"⚠️ Detection error: {str(e)}")
        return [], 0


def esp32_stream_generator(url):
    """Generator untuk streaming ESP32-CAM dengan deteksi manusia."""
    st.session_state.log.append(f"🔄 Mencoba menghubungkan ke ESP32-CAM: {url}")
    
    # Gunakan sesi HTTP untuk koneksi yang efisien dengan timeout yang tepat
    session = requests.Session()
    session.headers.update({
        'Connection': 'keep-alive',
        'Accept': 'multipart/x-mixed-replace; boundary=frame',
        'Cache-Control': 'no-cache, no-store'
    })
    
    # Konfigurasi adapter dengan keepalive dan retry yang optimal
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=1,
        pool_maxsize=1,
        max_retries=3,
        pool_block=False
    )
    session.mount('http://', adapter)
    
    # Variabel untuk pembatasan rate frame
    last_frame_time = time.time()
    frame_rate_limit = 0.03  # ~30fps maksimum
    
    try:
        # Sambungkan ke stream dengan timeout yang tepat
        stream = session.get(url, stream=True, timeout=(3, 2))  # (connect timeout, read timeout)
        
        if stream.status_code != 200:
            st.session_state.log.append(f"❌ Gagal koneksi: HTTP {stream.status_code}")
            yield None
            return
        
        # Buffer dengan ukuran yang lebih besar untuk frame processing yang lebih efisien
        byte_buffer = bytearray()
        
        # Konstanta untuk optimasi pencarian marker
        jpeg_start = b'\xff\xd8'  # JPEG start marker
        jpeg_end = b'\xff\xd9'    # JPEG end marker
        
        for chunk in stream.iter_content(chunk_size=8192):  # Buffer chunk yang lebih besar
            if not st.session_state.camera_on:
                break
                
            # Rate limiting untuk mencegah overload
            current_time = time.time()
            if current_time - last_frame_time < frame_rate_limit:
                continue
                
            byte_buffer.extend(chunk)
            
            # Cari start marker dari posisi yang lebih efisien
            start_pos = byte_buffer.find(jpeg_start)
            
            # Jika tidak ada start marker, kosongkan buffer untuk menghindari overhead
            if start_pos == -1:
                byte_buffer = bytearray()
                continue
                
            # Hapus data sebelum marker awal untuk menghemat memori
            if start_pos > 0:
                byte_buffer = byte_buffer[start_pos:]
                start_pos = 0
                
            # Cari end marker
            end_pos = byte_buffer.find(jpeg_end, start_pos + 2)
            
            if end_pos != -1:
                # Ekstrak frame lengkap
                frame_data = bytes(byte_buffer[start_pos:end_pos + 2])
                
                # Reset buffer ke data setelah frame saat ini
                byte_buffer = byte_buffer[end_pos + 2:]
                
                try:
                    # Decode frame dengan optimasi
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Update waktu frame terakhir untuk rate limiting
                        last_frame_time = current_time
                        
                        # Pre-resize frame untuk konsistensi dan performa
                        if frame.shape[0] > 480 or frame.shape[1] > 640:
                            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                        
                        # Process frame for detection
                        detection_frame = frame.copy()
                        people_boxes, count = process_esp32_frame(detection_frame, model)
                        
                        # Update state if count changed
                        if count != st.session_state.count:
                            st.session_state.count = count
                            st.session_state.lamp = 1 if count > 0 else 0
                            
                            # Send to Ubidots less frequently
                            current_time = time.time()
                            if current_time - st.session_state.last_sent > 3.0:
                                send_ubidots(VARIABLE_COUNT, count)
                                send_ubidots(VARIABLE_LIGHT, st.session_state.lamp)
                                st.session_state.last_sent = current_time
                        
                        # Draw detections on frame
                        if count > 0:
                            h, w = frame.shape[:2]
                            h_ratio = h / 416
                            w_ratio = w / 416
                            
                            for box in people_boxes:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                x1, y1 = int(x1 * w_ratio), int(y1 * h_ratio)
                                x2, y2 = int(x2 * w_ratio), int(y2 * h_ratio)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add count overlay
                        cv2.putText(frame, f"Jumlah Orang: {count}", 
                                   (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, (255, 100, 100), 1.5)
                        
                        yield frame
                    else:
                        # Jika frame rusak, lanjutkan ke frame berikutnya
                        continue
                except Exception as e:
                    st.session_state.log.append(f"⚠️ Frame error: {str(e)}")
                    # Lanjutkan ke frame berikutnya daripada keluar
                    continue
    
    except requests.exceptions.ConnectTimeout:
        st.session_state.log.append("❌ Timeout saat menghubungkan ke ESP32-CAM")
        yield None
    except requests.exceptions.ReadTimeout:
        st.session_state.log.append("❌ Timeout saat membaca data dari ESP32-CAM")
        yield None
    except requests.exceptions.ConnectionError:
        st.session_state.log.append("❌ Koneksi ke ESP32-CAM terputus")
        yield None
    except Exception as e:
        st.session_state.log.append(f"❌ Error ESP32-CAM: {str(e)}")
        yield None
    
    finally:
        # Pastikan resource dibebaskan dengan baik
        if 'stream' in locals():
            try:
                stream.close()
            except:
                pass
        if 'session' in locals():
            try:
                session.close()
            except:
                pass


# === Format ESP32 URL ===
def format_esp32_url(url):
    # Format URL dengan benar untuk ESP32-CAM
    if not url.startswith("http"):
        url = "http://" + url

    # Pastikan port 81 ada
    if ":81" not in url:
        # Hapus port lain jika ada
        if ":" in url[8:]:  # 8 karakter = "http://" 
            parts = url.split(":")
            url = parts[0] + ":" + parts[1].split("/")[0]  # Ambil bagian host saja
            url = url + ":81"
        else:
            url = url + ":81"

    # Pastikan path /stream ada di akhir
    if not url.endswith("/stream"):
        # Hapus path lain jika ada
        if "/" in url[8:] and ":" in url:
            # Ambil base URL dengan port
            base_parts = url.split("/")
            url = base_parts[0] + "//" + base_parts[2]
            if ":" not in url:
                url = url + ":81"
            url = url + "/stream"
        else:
            url = url + "/stream"

    return url


# === Add this helper function at the top level ===
def optimize_frame_for_detection(frame):
    """Optimize frame before YOLO detection"""
    # Ensure consistent size
    frame = cv2.resize(frame, (416, 416))  # Smaller size for faster detection
    
    # Enhance contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return frame


# === UI ===
st.title("🧠 AI Camera App to Control IoT Devices")
st.write("Aplikasi ini menggunakan YOLOv8 untuk mendeteksi orang dan mengontrol perangkat IoT seperti kamera dan lampu.")
st.warning(
    "⚠️ Webcam lokal tidak dapat digunakan di Streamlit Cloud karena OpenCV `cv2.VideoCapture(0)` tidak didukung. "
    "Harap gunakan ESP32-CAM atau perangkat kamera lain yang dapat diakses melalui jaringan."
)

# Create tabs for different functionality
tab1, tab2 = st.tabs(["Kamera & Deteksi", "AI Assistant"])

with tab1:
    st.selectbox("Pilih Kamera", ["Laptop Webcam", "ESP32-CAM"], key="camera_option")
    if st.session_state.camera_option == "ESP32-CAM":
        st.text_input("Masukkan URL IP ESP32-CAM", key="esp32_url")

    start = st.button("🔄 Mulai Kamera")
    stop = st.button("⛔ Stop Kamera")

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    detail_placeholder = st.empty()

with tab2:
    st.header("💬 AI Assistant (Powered by Google Gemini)")

    # Toggle for AI Assistant
    st.checkbox("Aktifkan AI Assistant", key="ai_enabled")

    if st.session_state.ai_enabled:
        # Generate summary button
        if st.button("🔍 Dapatkan Ringkasan Aktivitas"):
            summary = generate_activity_summary()
            st.session_state.chat_history.append({"role": "user", "content": "Berikan ringkasan aktivitas sistem"})
            st.session_state.chat_history.append({"role": "assistant", "content": summary})

        # Chat interface
        user_input = st.text_input("Ketik perintah atau pertanyaan:",
                                   placeholder="contoh: nyalakan kamera, matikan lampu, berapa orang terdeteksi?")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Generate AI response
            ai_response = generate_ai_response(user_input)

            # Process any commands in the response, passing both user input and AI response
            executed_cmds = process_ai_command(ai_response, user_input)
            if executed_cmds:
                ai_response += "\n\n*Tindakan yang dilakukan:*\n" + "\n".join([f"- {cmd}" for cmd in executed_cmds])

            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

        # Display chat history
        st.subheader("Riwayat Chat")
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**🧑 Anda:** {message['content']}")
                else:
                    st.markdown(f"**🤖 AI Assistant:** {message['content']}")

        # Add a clear chat button
        if st.button("🗑️ Bersihkan Riwayat Chat"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Aktifkan AI Assistant untuk menggunakan fitur ini.")

# === Tombol Start & Stop ===
if start:
    st.session_state.camera_on = True
    st.session_state.stop_clicks = 0
    send_ubidots(VARIABLE_CAMERA, 1)

    # Log for AI
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    st.session_state.activity_history.append(f"[{timestamp}] Kamera diaktifkan")

if stop:
    st.session_state.stop_clicks += 1
    if st.session_state.stop_clicks >= 2:
        st.session_state.camera_on = False
        st.session_state.stop_clicks = 0
        send_ubidots(VARIABLE_CAMERA, 0)
        send_ubidots(VARIABLE_LIGHT, 0)
        send_ubidots(VARIABLE_COUNT, 0)

        # Log for AI
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        st.session_state.activity_history.append(f"[{timestamp}] Kamera dan lampu dimatikan")
    else:
        st.session_state.camera_on = False
        send_ubidots(VARIABLE_CAMERA, 0)
        send_ubidots(VARIABLE_LIGHT, 0)
        send_ubidots(VARIABLE_COUNT, 0)

        # Log for AI
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        st.session_state.activity_history.append(f"[{timestamp}] Kamera dan lampu dimatikan")

# === Streaming dan Deteksi ===
if st.session_state.camera_on:
    try:
        if st.session_state.camera_option == "ESP32-CAM":
            url = st.session_state.esp32_url.strip()
            if not url:
                st.error("⚠️ URL ESP32-CAM tidak boleh kosong!")
                st.session_state.camera_on = False
            else:
                url = format_esp32_url(url)
                for frame in esp32_stream_generator(url):
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        # Update status
                        status_placeholder.markdown(
                            f"👥 **Jumlah Orang:** `{st.session_state.count}` &nbsp;&nbsp; "
                            f"💡 **Lampu:** `{'ON' if st.session_state.lamp else 'OFF'}`"
                        )
        else:
            # Optimize webcam capture
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer

        last_frame_time = time.time()
        frame_count = 0
        last_count = -1
        last_ubidots_send = time.time()
        skip_frames = 0  # Frame skip counter

        while st.session_state.camera_on:
            try:
                # Skip frames for better performance
                skip_frames += 1
                if skip_frames % 2 != 0:  # Process every other frame
                    continue

                if st.session_state.camera_option != "ESP32-CAM":
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # Regular webcam detection
                    results = model.predict(
                        frame,
                        verbose=False,
                        conf=0.5,
                        iou=0.45
                    )

                # Improved count calculation with filtering
                boxes = results[0].boxes
                people_boxes = [box for box in boxes if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.45]
                count = len(people_boxes)

                # More efficient drawing
                if count > 0:
                    frame = results[0].plot(
                        conf=False,  # Don't show confidence
                        labels=False,  # Don't show labels
                        line_width=2
                    )

                # Update state and UI less frequently
                if count != last_count:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    st.session_state.activity_history.append(f"[{timestamp}] Terdeteksi {count} orang")
                    last_count = count
                    st.session_state.count = count
                    st.session_state.lamp = 1 if count > 0 else 0

                # Optimize Ubidots sending (increased interval)
                now = time.time()
                if now - last_ubidots_send > 3.0:  # Increased to 3 seconds
                    send_ubidots(VARIABLE_LIGHT, st.session_state.lamp)
                    send_ubidots(VARIABLE_COUNT, count)
                    last_ubidots_send = now

                # Optimize frame display
                cv2.putText(frame, f"Jumlah Orang: {count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Update UI elements less frequently
                if frame_count % 10 == 0:
                    status_placeholder.markdown(
                        f"👥 **Jumlah Orang:** `{count}` &nbsp;&nbsp; 💡 **Lampu:** `{'ON' if st.session_state.lamp else 'OFF'}`"
                    )
                    detail_placeholder.markdown(f"""
                    **ℹ️ Detail**  
                    - Terakhir Kirim: `{time.strftime('%H:%M:%S', time.localtime(last_ubidots_send))}`  
                    - Status Kamera: `{'Aktif' if st.session_state.camera_on else 'Nonaktif'}`  
                    - Kamera Dipilih: `{st.session_state.camera_option}`  
                    """)

                frame_count += 1
                if frame_count % 30 == 0:  # Reduced FPS logging frequency
                    fps = 30 / (time.time() - last_frame_time)
                    last_frame_time = time.time()
                    st.session_state.log.append(f"📈 FPS: {fps:.1f}")

                time.sleep(0.01)  # Reduced sleep time

            except Exception as e:
                st.session_state.log.append(f"❌ Error: {str(e)}")
                time.sleep(0.1)

        if st.session_state.camera_option != "ESP32-CAM" and 'cap' in locals():
            cap.release()

    except Exception as e:
        st.session_state.log.append(f"❌ Error pada sistem kamera: {str(e)}")
        st.session_state.camera_on = False

# === Info Saat Kamera Mati ===
if not st.session_state.camera_on:
    detail_placeholder.markdown(f"""
    ### ℹ️ Detail  
    - Terakhir Kirim: `{time.strftime('%H:%M:%S', time.localtime(st.session_state.get("last_sent", time.time())))}`  
    - Status Kamera: `{'Aktif' if st.session_state.camera_on else 'Nonaktif'}`  
    - Kamera Dipilih: `{st.session_state.camera_option}`  
    """)

# Tampilkan log terakhir
with st.expander("Log System", expanded=False):
    for log in st.session_state.log[-10:]:
        st.write(log)
