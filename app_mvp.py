import cv2
import numpy as np
import mediapipe as mp
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- Configuration ---
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
HEADER_HEIGHT = 100 
UI_BG_COLOR = (45, 45, 45)

# --- UI Coordinates ---
SPEC_X1, SPEC_Y1 = 20, 15
SPEC_X2, SPEC_Y2 = 400, 85
CURR_CENTER_X, CURR_CENTER_Y = 450, 50
SHAPE_X1, SHAPE_Y1 = 500, 15
SHAPE_X2, SHAPE_Y2 = 620, 85
SLIDER_X1, SLIDER_Y = 660, 50
SLIDER_X2 = 860 
ERASE_X1, ERASE_Y1 = 900, 15
ERASE_X2, ERASE_Y2 = 1020, 85
CLEAR_X1, CLEAR_Y1 = 1060, 15
CLEAR_X2, CLEAR_Y2 = 1210, 85

# --- Helper Functions ---
def create_spectrum_strip(total_width, height):
    rb_width = int(total_width * 0.8)
    bw_width = total_width - rb_width
    hues = np.tile(np.linspace(0, 179, rb_width, dtype=np.uint8), (height, 1))
    sat = np.ones((height, rb_width), dtype=np.uint8) * 255
    val = np.ones((height, rb_width), dtype=np.uint8) * 255
    bgr_rb = cv2.cvtColor(cv2.merge([hues, sat, val]), cv2.COLOR_HSV2BGR)
    bw_val = np.tile(np.linspace(255, 0, bw_width, dtype=np.uint8), (height, 1))
    bgr_bw = cv2.cvtColor(cv2.merge([np.zeros_like(bw_val), np.zeros_like(bw_val), bw_val]), cv2.COLOR_HSV2BGR)
    return np.hstack((bgr_rb, bgr_bw))

def detect_and_draw_shape(points, canvas, color, thickness):
    if len(points) < 10: return
    contour = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    if len(approx) == 3: cv2.polylines(canvas, [approx], True, color, thickness)
    elif len(approx) == 4:
        aspectRatio = float(w) / h
        if 0.90 <= aspectRatio <= 1.10: cv2.rectangle(canvas, (x, y), (x + max(w,h), y + max(w,h)), color, thickness)
        else: cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
    else:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cv2.circle(canvas, (int(cx), int(cy)), int(radius), color, thickness)

# --- The Processor Class ---
class AirCanvasProcessor(VideoTransformerBase):
    def __init__(self):
        # Initialize State
        self.brushThickness = 15
        self.eraserThickness = 50
        self.currColor = (0, 0, 255) # Red
        self.lastBrushColor = (0, 0, 255)
        self.xp, self.yp = 0, 0
        self.imgCanvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        self.shapeMode = False
        self.shapePoints = []
        
        # Initialize MediaPipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)
        self.spectrum_img = create_spectrum_strip(SPEC_X2 - SPEC_X1, SPEC_Y2 - SPEC_Y1)

    def draw_ui(self, img):
        cv2.rectangle(img, (0, 0), (FRAME_WIDTH, HEADER_HEIGHT), UI_BG_COLOR, cv2.FILLED)
        img[SPEC_Y1:SPEC_Y2, SPEC_X1:SPEC_X2] = self.spectrum_img
        cv2.rectangle(img, (SPEC_X1, SPEC_Y1), (SPEC_X2, SPEC_Y2), (200, 200, 200), 2)
        cv2.circle(img, (CURR_CENTER_X, CURR_CENTER_Y), 30, self.currColor, cv2.FILLED)
        
        btn_col = (0, 255, 0) if self.shapeMode else (80, 80, 80)
        cv2.rectangle(img, (SHAPE_X1, SHAPE_Y1), (SHAPE_X2, SHAPE_Y2), btn_col, cv2.FILLED)
        cv2.putText(img, "SHAPE", (SHAPE_X1+10, SHAPE_Y1+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        cv2.line(img, (SLIDER_X1, SLIDER_Y), (SLIDER_X2, SLIDER_Y), (150, 150, 150), 4)
        cv2.rectangle(img, (ERASE_X1, ERASE_Y1), (ERASE_X2, ERASE_Y2), (0,0,0), cv2.FILLED)
        cv2.putText(img, "Erase", (ERASE_X1+10, ERASE_Y1+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.rectangle(img, (CLEAR_X1, CLEAR_Y1), (CLEAR_X2, CLEAR_Y2), (80,80,80), cv2.FILLED)
        cv2.putText(img, "Clear", (CLEAR_X1+10, CLEAR_Y1+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    def transform(self, frame):
        # Convert Streamlit frame to OpenCV
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

        imgSmall = cv2.resize(img, (640, 360))
        results = self.hands.process(cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB))
        
        active_thickness = self.eraserThickness if self.currColor == (0,0,0) else self.brushThickness

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                
                if len(lmList) != 0:
                    x1, y1 = lmList[8][1:]
                    x2, y2 = lmList[12][1:]
                    fingers = []
                    if lmList[4][1] < lmList[3][1]: fingers.append(1) 
                    else: fingers.append(0)
                    for id in [8, 12, 16, 20]:
                        fingers.append(1 if lmList[id][2] < lmList[id-2][2] else 0)
                    
                    fingers_up = fingers.count(1)

                    # Selection
                    if fingers[1] and fingers[2]:
                        if self.shapeMode and len(self.shapePoints) > 0:
                            detect_and_draw_shape(self.shapePoints, self.imgCanvas, self.currColor, active_thickness)
                            self.shapePoints = []
                        
                        self.xp, self.yp = 0, 0
                        cv2.rectangle(img, (x1-15, y1-25), (x2+15, y2+25), (255,255,255), 2)
                        
                        if y1 < HEADER_HEIGHT:
                            if SPEC_X1 < x1 < SPEC_X2:
                                rel_x = max(0, min(x1 - SPEC_X1, (SPEC_X2 - SPEC_X1) - 1))
                                rel_y = max(0, min(y1 - SPEC_Y1, (SPEC_Y2 - SPEC_Y1) - 1))
                                c = self.spectrum_img[rel_y, rel_x]
                                self.currColor = (int(c[0]), int(c[1]), int(c[2]))
                            elif SHAPE_X1 < x1 < SHAPE_X2:
                                self.shapeMode = not self.shapeMode
                                self.shapePoints = []
                            elif ERASE_X1 < x1 < ERASE_X2: self.currColor = (0,0,0)
                            elif CLEAR_X1 < x1 < CLEAR_X2: 
                                self.imgCanvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
                                self.shapePoints = []
                    
                    # Drawing
                    elif fingers[1] and fingers_up == 1:
                        if self.shapeMode:
                            self.shapePoints.append((x1, y1))
                            if len(self.shapePoints) > 2:
                                cv2.line(img, self.shapePoints[-2], self.shapePoints[-1], self.currColor, active_thickness)
                        else:
                            if self.xp == 0 and self.yp == 0: self.xp, self.yp = x1, y1
                            cv2.line(img, (self.xp, self.yp), (x1, y1), self.currColor, active_thickness)
                            cv2.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.currColor, active_thickness)
                            self.xp, self.yp = x1, y1
                    else:
                        if self.shapeMode and len(self.shapePoints) > 0:
                             detect_and_draw_shape(self.shapePoints, self.imgCanvas, self.currColor, active_thickness)
                             self.shapePoints = []
                        self.xp, self.yp = 0, 0

        # Fast Merge
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY)
        img[mask == 255] = self.imgCanvas[mask == 255]
        
        self.draw_ui(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit Layout ---
st.set_page_config(page_title="Air Canvas MVP", layout="wide")
st.title("🎨 Air Canvas: AI Powered Drawing")
st.write("Turn on your webcam and wait for the system to load. Stand back slightly so your hands are visible.")

webrtc_streamer(key="air-canvas", 
                mode=WebRtcMode.SENDRECV, 
                video_transformer_factory=AirCanvasProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True)
