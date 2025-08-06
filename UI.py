import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import time
import os
import concurrent.futures
from deepface import DeepFace
import numpy as np
import requests
from ultralytics import YOLO

class AlertWindow(tk.Toplevel):
    def __init__(self, parent, message):
        super().__init__(parent)
        self.title("ALERT!")
        self.geometry("400x200")
        self.attributes('-topmost', True)
        self.configure(bg='black')
        
        alert_label = tk.Label(
            self, 
            text=message, 
            font=('Helvetica', 24, 'bold'), 
            fg='red', 
            bg='black',
            wraplength=380
        )
        alert_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        close_btn = tk.Button(
            self, 
            text="OK", 
            command=self.destroy,
            font=('Helvetica', 14),
            bg='red',
            fg='white'
        )
        close_btn.pack(pady=10)
        self.after(5000, self.destroy)

class SecuritySystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Security System")
        self.root.geometry("1400x900")
        
        # Initialize all attributes
        self.face_detection_mode = None
        self.target_image_path = None
        self.video_file_path = None
        self.weapon_alert_sent = False
        self.trespassing_alert_sent = False
        self.last_alert_time = 0
        self.target_alert_time = 0
        self.weapon_alert_time = 0
        self.trespassing_alert_time = 0
        self.alert_cooldown = 5
        self.cap = None
        self.video_capture = None
        self.current_frame = None
        self.weapon_detection_mode = None
        self.trespassing_detection_mode = None
        
        # Variables
        self.running = True
        self.active_module = None
        self.modules = {
            "face_detection": {"active": False, "tab": None},
            "weapon_detection": {"active": False, "tab": None},
            "trespassing_detection": {"active": False, "tab": None}
        }
        
        # Initialize models
        self.initialize_models()
        
        # Setup UI
        self.setup_ui()
        
        # Start the video update thread
        self.update_video()
    
    def initialize_models(self):
        """Initialize all the required models"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.weapon_model = YOLO(r"runs\detect\train\weights\best.pt")
            self.track_model = YOLO(r"runs\segment\track_segmentation_1\weights\best.pt")
            self.person_model = YOLO("yolov8n.pt")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize models: {str(e)}")
            self.root.destroy()
    
    def setup_ui(self):
        """Setup the main UI components"""
        style = ttk.Style()
        style.configure('TNotebook.Tab', font=('Helvetica', 10, 'bold'))
        style.configure('Alert.TLabel', foreground='red', font=('Helvetica', 16, 'bold'))
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.setup_face_detection_tab()
        self.setup_weapon_detection_tab()
        self.setup_trespassing_detection_tab()
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        alert_frame = ttk.LabelFrame(main_frame, text="System Alerts", height=150)
        alert_frame.pack(fill=tk.X, pady=(5,0))
        alert_frame.pack_propagate(False)
        
        self.alert_text = tk.Text(
            alert_frame, 
            height=8, 
            state=tk.DISABLED,
            font=('Helvetica', 10)
        )
        self.alert_text.tag_config('important', foreground='red', font=('Helvetica', 10, 'bold'))
        
        scrollbar = ttk.Scrollbar(alert_frame, command=self.alert_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.alert_text.config(yscrollcommand=scrollbar.set)
        self.alert_text.pack(fill=tk.BOTH, expand=True)
        
        self.current_alert = ttk.Label(
            main_frame, 
            text="", 
            style='Alert.TLabel',
            wraplength=1200
        )
        self.current_alert.pack(fill=tk.X, pady=(5,0))
    
    def setup_face_detection_tab(self):
        """Setup the face detection module tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Missing/Criminal Detection")
        self.modules["face_detection"]["tab"] = tab
        
        video_frame = ttk.LabelFrame(tab, text="Live Feed")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.face_video_label = ttk.Label(video_frame)
        self.face_video_label.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(tab, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)
        
        mode_frame = ttk.LabelFrame(control_frame, text="Detection Mode")
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.normal_btn = ttk.Button(mode_frame, text="Start Normal Detection", 
                                   command=lambda: self.start_face_detection('normal'))
        self.normal_btn.pack(fill=tk.X, pady=5)
        
        self.specific_btn = ttk.Button(mode_frame, text="Start Specific Detection", 
                                     command=lambda: self.start_face_detection('specific'))
        self.specific_btn.pack(fill=tk.X, pady=5)
        
        self.face_stop_btn = ttk.Button(mode_frame, text="Stop Detection", 
                                     command=self.stop_face_detection,
                                     state=tk.DISABLED)
        self.face_stop_btn.pack(fill=tk.X, pady=5)
        
        self.target_frame = ttk.LabelFrame(control_frame, text="Target Image")
        self.target_frame.pack(fill=tk.X, pady=5)
        
        self.select_target_btn = ttk.Button(self.target_frame, text="Select Target Image", 
                                          command=self.select_target_image)
        self.select_target_btn.pack(fill=tk.X, pady=5)
        
        self.target_preview = ttk.Label(self.target_frame)
        self.target_preview.pack(fill=tk.X, pady=5)
    
    def setup_weapon_detection_tab(self):
        """Setup the weapon detection module tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Weapon Detection")
        self.modules["weapon_detection"]["tab"] = tab
        
        video_frame = ttk.LabelFrame(tab, text="Detection Feed")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.weapon_video_label = ttk.Label(video_frame)
        self.weapon_video_label.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(tab, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)
        
        weapon_ctrl_frame = ttk.LabelFrame(control_frame, text="Weapon Detection")
        weapon_ctrl_frame.pack(fill=tk.X, pady=5)
        
        self.select_weapon_file_btn = ttk.Button(weapon_ctrl_frame, text="Select Video File", 
                                              command=self.select_weapon_video_file)
        self.select_weapon_file_btn.pack(fill=tk.X, pady=5)
        
        self.start_weapon_file_btn = ttk.Button(weapon_ctrl_frame, text="Start File Detection", 
                                             command=lambda: self.start_weapon_detection('file'),
                                             state=tk.DISABLED)
        self.start_weapon_file_btn.pack(fill=tk.X, pady=5)
        
        self.start_weapon_realtime_btn = ttk.Button(weapon_ctrl_frame, text="Start Realtime Detection", 
                                                 command=lambda: self.start_weapon_detection('realtime'))
        self.start_weapon_realtime_btn.pack(fill=tk.X, pady=5)
        
        self.stop_weapon_btn = ttk.Button(weapon_ctrl_frame, text="Stop Detection", 
                                        command=self.stop_weapon_detection,
                                        state=tk.DISABLED)
        self.stop_weapon_btn.pack(fill=tk.X, pady=5)
        
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_frame = ttk.LabelFrame(control_frame, text="Confidence Threshold")
        confidence_frame.pack(fill=tk.X, pady=5)
        
        self.confidence_slider = ttk.Scale(confidence_frame, from_=0.1, to=0.9, 
                                         variable=self.confidence_var,
                                         command=lambda v: self.confidence_var.set(round(float(v), 1)))
        self.confidence_slider.pack(fill=tk.X, padx=5, pady=5)
        
        self.confidence_label = ttk.Label(confidence_frame, text=f"Current: {self.confidence_var.get():.1f}")
        self.confidence_label.pack()
        
        self.confidence_var.trace_add("write", lambda *_: self.confidence_label.config(
            text=f"Current: {self.confidence_var.get():.1f}"))
    
    def setup_trespassing_detection_tab(self):
        """Setup the trespassing detection module tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Trespassing Detection")
        self.modules["trespassing_detection"]["tab"] = tab
        
        video_frame = ttk.LabelFrame(tab, text="Detection Feed")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.trespassing_video_label = ttk.Label(video_frame)
        self.trespassing_video_label.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(tab, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)
        
        trespass_ctrl_frame = ttk.LabelFrame(control_frame, text="Trespassing Detection")
        trespass_ctrl_frame.pack(fill=tk.X, pady=5)
        
        self.select_trespassing_file_btn = ttk.Button(trespass_ctrl_frame, text="Select Video File", 
                                                    command=self.select_trespassing_video_file)
        self.select_trespassing_file_btn.pack(fill=tk.X, pady=5)
        
        self.start_trespassing_file_btn = ttk.Button(trespass_ctrl_frame, text="Start File Detection", 
                                                  command=lambda: self.start_trespassing_detection('file'),
                                                  state=tk.DISABLED)
        self.start_trespassing_file_btn.pack(fill=tk.X, pady=5)
        
        self.start_trespassing_realtime_btn = ttk.Button(trespass_ctrl_frame, text="Start Realtime Detection", 
                                                      command=lambda: self.start_trespassing_detection('realtime'))
        self.start_trespassing_realtime_btn.pack(fill=tk.X, pady=5)
        
        self.stop_trespassing_btn = ttk.Button(trespass_ctrl_frame, text="Stop Detection", 
                                             command=self.stop_trespassing_detection,
                                             state=tk.DISABLED)
        self.stop_trespassing_btn.pack(fill=tk.X, pady=5)
    
    def on_tab_changed(self, event):
        """Handle tab changes"""
        selected_tab = self.notebook.tab(self.notebook.select(), "text")
        
        self.stop_all_detections()
        
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        
        if selected_tab == "Missing/Criminal Detection":
            self.modules["face_detection"]["active"] = True
        elif selected_tab == "Weapon Detection":
            self.modules["weapon_detection"]["active"] = True
        elif selected_tab == "Trespassing Detection":
            self.modules["trespassing_detection"]["active"] = True
    
    def stop_all_detections(self):
        """Stop all active detections"""
        if self.modules["face_detection"]["active"]:
            self.stop_face_detection()
        if self.modules["weapon_detection"]["active"]:
            self.stop_weapon_detection()
        if self.modules["trespassing_detection"]["active"]:
            self.stop_trespassing_detection()
    
    def start_face_detection(self, mode):
        """Start face detection in the specified mode"""
        if mode == 'specific' and not hasattr(self, 'target_image_path'):
            messagebox.showwarning("Warning", "Please select a target image first!")
            return
            
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.face_detection_mode = mode
        self.modules["face_detection"]["active"] = True
        self.add_alert(f"Started {mode} face detection")
        
        self.normal_btn.config(state=tk.DISABLED)
        self.specific_btn.config(state=tk.DISABLED)
        self.face_stop_btn.config(state=tk.NORMAL)
        self.select_target_btn.config(state=tk.DISABLED)
    
    def stop_face_detection(self):
        """Stop face detection"""
        self.modules["face_detection"]["active"] = False
        self.face_detection_mode = None
        
        if self.cap is not None and not any(module["active"] for module in self.modules.values()):
            self.cap.release()
            self.cap = None
        
        self.add_alert("Face detection stopped")
        
        self.normal_btn.config(state=tk.NORMAL)
        self.specific_btn.config(state=tk.NORMAL)
        self.face_stop_btn.config(state=tk.DISABLED)
        self.select_target_btn.config(state=tk.NORMAL)
    
    def select_weapon_video_file(self):
        """Select video file for weapon detection"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.weapon_video_path = file_path
            self.add_alert(f"Weapon detection video set: {os.path.basename(file_path)}")
            self.start_weapon_file_btn.config(state=tk.NORMAL)
    
    def start_weapon_detection(self, mode):
        """Start weapon detection in specified mode"""
        self.weapon_detection_mode = mode
        
        if mode == 'file' and not hasattr(self, 'weapon_video_path'):
            messagebox.showwarning("Warning", "Please select a video file first!")
            return
            
        if mode == 'realtime':
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            self.video_capture = cv2.VideoCapture(self.weapon_video_path)
        
        self.modules["weapon_detection"]["active"] = True
        self.add_alert(f"Weapon detection started ({mode} mode)")
        
        self.select_weapon_file_btn.config(state=tk.DISABLED)
        self.start_weapon_file_btn.config(state=tk.DISABLED)
        self.start_weapon_realtime_btn.config(state=tk.DISABLED)
        self.stop_weapon_btn.config(state=tk.NORMAL)
    
    def stop_weapon_detection(self):
        """Stop weapon detection"""
        self.modules["weapon_detection"]["active"] = False
        self.weapon_detection_mode = None
        
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            
        if self.cap is not None and not any(module["active"] for module in self.modules.values()):
            self.cap.release()
            self.cap = None
        
        self.add_alert("Weapon detection stopped")
        
        self.select_weapon_file_btn.config(state=tk.NORMAL)
        self.start_weapon_file_btn.config(state=tk.NORMAL if hasattr(self, 'weapon_video_path') else tk.DISABLED)
        self.start_weapon_realtime_btn.config(state=tk.NORMAL)
        self.stop_weapon_btn.config(state=tk.DISABLED)
    
    def select_trespassing_video_file(self):
        """Select video file for trespassing detection"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.trespassing_video_path = file_path
            self.add_alert(f"Trespassing detection video set: {os.path.basename(file_path)}")
            self.start_trespassing_file_btn.config(state=tk.NORMAL)
    
    def start_trespassing_detection(self, mode):
        """Start trespassing detection in specified mode"""
        self.trespassing_detection_mode = mode
        
        if mode == 'file' and not hasattr(self, 'trespassing_video_path'):
            messagebox.showwarning("Warning", "Please select a video file first!")
            return
            
        if mode == 'realtime':
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            self.video_capture = cv2.VideoCapture(self.trespassing_video_path)
        
        self.modules["trespassing_detection"]["active"] = True
        self.add_alert(f"Trespassing detection started ({mode} mode)")
        
        self.select_trespassing_file_btn.config(state=tk.DISABLED)
        self.start_trespassing_file_btn.config(state=tk.DISABLED)
        self.start_trespassing_realtime_btn.config(state=tk.DISABLED)
        self.stop_trespassing_btn.config(state=tk.NORMAL)
    
    def stop_trespassing_detection(self):
        """Stop trespassing detection"""
        self.modules["trespassing_detection"]["active"] = False
        self.trespassing_detection_mode = None
        
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            
        if self.cap is not None and not any(module["active"] for module in self.modules.values()):
            self.cap.release()
            self.cap = None
        
        self.add_alert("Trespassing detection stopped")
        
        self.select_trespassing_file_btn.config(state=tk.NORMAL)
        self.start_trespassing_file_btn.config(state=tk.NORMAL if hasattr(self, 'trespassing_video_path') else tk.DISABLED)
        self.start_trespassing_realtime_btn.config(state=tk.NORMAL)
        self.stop_trespassing_btn.config(state=tk.DISABLED)
    
    def select_target_image(self):
        """Select target image for face detection"""
        file_path = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.target_image_path = file_path
            self.add_alert(f"Target image set: {os.path.basename(file_path)}")
            
            try:
                img = Image.open(file_path)
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)
                self.target_preview.config(image=photo)
                self.target_preview.image = photo
            except Exception as e:
                self.add_alert(f"Error loading image: {str(e)}")
    
    def add_alert(self, message, is_important=False):
        """Add message to alert log"""
        self.alert_text.config(state=tk.NORMAL)
        if is_important:
            self.alert_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] ", 'important')
            self.alert_text.insert(tk.END, f"{message}\n", 'important')
            self.show_alert_popup(message)
        else:
            self.alert_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        
        self.alert_text.see(tk.END)
        self.alert_text.config(state=tk.DISABLED)
        
        self.current_alert.config(text=message)
        if is_important:
            self.current_alert.config(foreground='red', font=('Helvetica', 16, 'bold'))
        else:
            self.current_alert.config(foreground='black', font=('Helvetica', 12))
    
    def show_alert_popup(self, message):
        """Show popup alert window"""
        AlertWindow(self.root, message)
    
    def process_face_detection(self, frame):
        """Process frame for face detection with target person bounding box and caption"""
        display_frame = frame.copy()
        
        if self.face_detection_mode == 'normal':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    result = DeepFace.find(
                        img_path=face_roi, 
                        db_path="face_data/", 
                        model_name="SFace",
                        enforce_detection=False
                    )
                    
                    if result and any(len(res) > 0 for res in result):
                        for res in result:
                            if len(res) > 0:
                                identity = os.path.splitext(os.path.basename(res.iloc[0]["identity"]))[0]
                                
                                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(display_frame, identity, (x, y - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                
                                current_time = time.time()
                                if current_time - self.last_alert_time > self.alert_cooldown:
                                    alert_message = f"Person identified: {identity}"
                                    self.add_alert(alert_message, is_important=True)
                                    try:
                                        response = requests.post("http://127.0.0.1:8000/alert", 
                                                              json={"names": [identity]})
                                    except Exception as e:
                                        self.add_alert(f"Alert send failed: {e}")
                                    self.last_alert_time = current_time
                except Exception as e:
                    self.add_alert(f"Face recognition error: {e}")
        
        elif self.face_detection_mode == 'specific' and hasattr(self, 'target_image_path'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    result = DeepFace.verify(
                        img1_path=face_roi,
                        img2_path=self.target_image_path,
                        model_name="SFace",
                        enforce_detection=False
                    )
                    
                    if result.get("verified"):
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(display_frame, "Target Person", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        current_time = time.time()
                        if current_time - self.target_alert_time > self.alert_cooldown:
                            alert_message = "TARGET PERSON DETECTED!"
                            self.add_alert(alert_message, is_important=True)
                            try:
                                response = requests.get("http://127.0.0.1:8000/target_face_alert")
                                self.target_alert_time = current_time
                            except Exception as e:
                                self.add_alert(f"Alert send failed: {e}")
                
                except Exception as e:
                    self.add_alert(f"Face verification error: {e}")
        
        return display_frame
    
    def process_weapon_detection(self, frame):
        """Process frame for weapon detection with 5-second alert delay"""
        display_frame = frame.copy()
        weapon_detected = False
        
        results = self.weapon_model(frame, stream=True, conf=self.confidence_var.get())
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                cls = int(box.cls[0])
                label = self.weapon_model.names[cls]
                
                if label.lower() == "weapon":
                    weapon_detected = True
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(display_frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    current_time = time.time()
                    if current_time - self.weapon_alert_time > self.alert_cooldown:
                        alert_message = "🚨 Weapon detected!"
                        self.add_alert(alert_message, is_important=True)
                        try:
                            response = requests.get("http://127.0.0.1:8000/weapon_alert")
                            self.weapon_alert_time = current_time
                        except requests.exceptions.RequestException as e:
                            self.add_alert(f"Error sending alert: {e}")
        
        if not weapon_detected:
            self.weapon_alert_sent = False
        
        return display_frame
    
    def process_trespassing_detection(self, frame):
        """Process frame for trespassing detection with 5-second alert delay"""
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        person_detected_on_track = False
        
        track_results = self.track_model(frame)[0]
        track_mask = None

        if track_results.masks:
            masks = track_results.masks.data.cpu().numpy()
            combined_mask = np.any(masks > 0.5, axis=0).astype(np.uint8)
            mask_resized = cv2.resize(combined_mask, (width, height))
            track_mask = mask_resized

            colored_mask = np.zeros_like(frame)
            colored_mask[track_mask == 1] = (0, 0, 255)
            display_frame = cv2.addWeighted(display_frame, 1.0, colored_mask, 0.5, 0)

        person_results = self.person_model(frame)[0]

        for box in person_results.boxes:
            cls = int(box.cls[0])
            if self.person_model.names[cls] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(display_frame, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if track_mask is not None and track_mask[cy, cx] == 1:
                    person_detected_on_track = True

        current_time = time.time()
        if person_detected_on_track and current_time - self.trespassing_alert_time > self.alert_cooldown:
            alert_message = "🚨 Person detected on railway track!"
            self.add_alert(alert_message, is_important=True)
            try:
                res = requests.get("http://127.0.0.1:8000/track_alert")
                self.trespassing_alert_time = current_time
            except Exception as e:
                self.add_alert(f"Failed to send alert: {e}")
        elif not person_detected_on_track:
            self.trespassing_alert_sent = False
        
        return display_frame
    
    def update_video(self):
        """Main video update loop"""
        if self.running:
            frame = None
            if self.modules["face_detection"]["active"] and self.cap is not None:
                ret, frame = self.cap.read()
                if not ret:
                    frame = None
            elif self.modules["weapon_detection"]["active"]:
                if self.weapon_detection_mode == 'realtime' and self.cap is not None:
                    ret, frame = self.cap.read()
                    if not ret:
                        frame = None
                elif self.video_capture is not None:
                    ret, frame = self.video_capture.read()
                    if not ret:
                        self.stop_weapon_detection()
                        frame = None
            elif self.modules["trespassing_detection"]["active"]:
                if self.trespassing_detection_mode == 'realtime' and self.cap is not None:
                    ret, frame = self.cap.read()
                    if not ret:
                        frame = None
                elif self.video_capture is not None:
                    ret, frame = self.video_capture.read()
                    if not ret:
                        self.stop_trespassing_detection()
                        frame = None
            
            if frame is not None:
                frame = cv2.resize(frame, (640, 480))
                
                if self.modules["face_detection"]["active"]:
                    processed_frame = self.process_face_detection(frame)
                    img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.face_video_label.imgtk = imgtk
                    self.face_video_label.configure(image=imgtk)
                
                if self.modules["weapon_detection"]["active"]:
                    processed_frame = self.process_weapon_detection(frame)
                    img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.weapon_video_label.imgtk = imgtk
                    self.weapon_video_label.configure(image=imgtk)
                
                if self.modules["trespassing_detection"]["active"]:
                    processed_frame = self.process_trespassing_detection(frame)
                    img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.trespassing_video_label.imgtk = imgtk
                    self.trespassing_video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_video)
    
    def exit_app(self):
        """Cleanup and exit application"""
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        if self.video_capture is not None:
            self.video_capture.release()
        
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SecuritySystemApp(root)
    root.mainloop()
