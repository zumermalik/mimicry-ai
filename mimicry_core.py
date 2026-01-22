import cv2
import mediapipe as mp
import numpy as np
import math
import sys

# --- CONFIGURATION (The "Hardware" specs) ---
WINDOW_NAME = "Mimicry.ai // FounderForge Prototype"
ARM_SEGMENT_1_LEN = 160  # Length of shoulder-to-elbow (pixels)
ARM_SEGMENT_2_LEN = 130  # Length of elbow-to-wrist (pixels)
# Origin will be dynamic based on window size

class RobotArmSim:
    """
    Simulates a 2-Link Planar Robotic Arm using Inverse Kinematics.
    This proves we can control hardware math using software vision.
    """
    def __init__(self, l1, l2, origin):
        self.l1 = l1
        self.l2 = l2
        self.origin = origin
        self.theta1 = math.pi / 2 # Initial angle (90 deg)
        self.theta2 = 0.0         # Initial angle
        self.target_reached = True

    def inverse_kinematics(self, target_x, target_y):
        """
        Calculates joint angles (theta1, theta2) to reach (target_x, target_y).
        Uses geometric Law of Cosines.
        """
        # Shift coordinates relative to robot base
        x = target_x - self.origin[0]
        y = target_y - self.origin[1]
        
        # Distance to target
        dist = math.sqrt(x**2 + y**2)
        
        # Safety: Clamp reach if target is beyond arm span
        max_reach = self.l1 + self.l2
        if dist > max_reach:
            scale = max_reach / dist
            x *= scale
            y *= scale
            dist = max_reach
            self.target_reached = False
        else:
            self.target_reached = True

        # Avoid division by zero
        if dist == 0: return

        # Law of Cosines to find angle of elbow (theta2)
        try:
            cos_angle2 = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
            # Clip for float stability
            cos_angle2 = max(-1.0, min(1.0, cos_angle2))
            angle2 = math.acos(cos_angle2) 
            
            # Elbow config: -angle2 for "elbow up"
            self.theta2 = -angle2 
            
            # Find Shoulder angle (theta1)
            k1 = self.l1 + self.l2 * math.cos(self.theta2)
            k2 = self.l2 * math.sin(self.theta2)
            self.theta1 = math.atan2(y, x) - math.atan2(k2, k1)
            
        except ValueError:
            pass # Target unobtainable

    def get_joints(self):
        """Returns (shoulder, elbow, wrist) coordinates for drawing."""
        sx, sy = self.origin
        
        # Elbow calculation
        ex = sx + self.l1 * math.cos(self.theta1)
        ey = sy + self.l1 * math.sin(self.theta1)
        
        # Wrist calculation relative to elbow
        wx = ex + self.l2 * math.cos(self.theta1 + self.theta2)
        wy = ey + self.l2 * math.sin(self.theta1 + self.theta2)
        
        return (int(sx), int(sy)), (int(ex), int(ey)), (int(wx), int(wy))

def run_mimicry():
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam not found.")
        return

    # Dynamic setup based on camera resolution
    ret, frame = cap.read()
    h, w, c = frame.shape
    robot = RobotArmSim(ARM_SEGMENT_1_LEN, ARM_SEGMENT_2_LEN, (w//2, h))
    
    print(">>> SYSTEM BOOT: Mimicry.ai Teleoperation Interface")
    print(">>> CAM: ON")

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        # Mirror and Convert
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        # UI Overlay
        cv2.rectangle(image, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.putText(image, "MIMICRY.AI // TELEOP", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 127), 2)

        target_point = None

        # 1. PERCEPTION
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS)
                idx_finger = hand_lms.landmark[8]
                cx, cy = int(idx_finger.x * w), int(idx_finger.y * h)
                target_point = (cx, cy)
                cv2.circle(image, (cx, cy), 10, (255, 0, 255), -1)

        # 2. LOGIC
        if target_point:
            robot.inverse_kinematics(target_point[0], target_point[1])

        # 3. ACTUATION VISUALIZATION
        shoulder, elbow, wrist = robot.get_joints()
        cv2.line(image, shoulder, elbow, (200, 200, 200), 12)
        cv2.line(image, elbow, wrist, (200, 200, 200), 8)
        cv2.circle(image, shoulder, 15, (50, 50, 255), -1)
        cv2.circle(image, elbow, 12, (50, 50, 255), -1)
        cv2.circle(image, wrist, 8, (0, 255, 0), -1)

        # Status
        status_text = "LOCKED" if robot.target_reached else "MAX RANGE"
        color = (0, 255, 0) if robot.target_reached else (0, 0, 255)
        cv2.putText(image, f"STATUS: {status_text}", (w - 300, 50), 
                    cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

        cv2.imshow(WINDOW_NAME, image)
        if cv2.waitKey(5) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_mimicry()
