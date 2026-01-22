# Mimicry.ai // Zero-Hardware Teleoperation
> **FounderForge Submission** | **Category:** Robotics & AI Integration  
> **Status:** MVP / Prototype

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green) ![MediaPipe](https://img.shields.io/badge/MediaPipe-Skeletal%20Tracking-orange)

## 🚀 The Concept
**Mimicry.ai** democratizes robotic control. 

Traditional teleoperation (controlling robots remotely) requires expensive haptic suits, bulky gloves, or complex joystick arrays. This creates a massive barrier to entry for operators in hazardous environments, telesurgery, or space robotics.

**Mimicry.ai replaces hardware sensors with Computer Vision.**
By turning a standard laptop webcam into a high-precision, low-latency robotic controller, we allow anyone to control complex kinematic chains using natural hand gestures.

---

## 🛠 How It Works
We simulate a **2-Link Planar Robotic Arm** entirely in software to prove the control logic without needing physical servos (Zero-Hardware Validation).

1.  **Perception Layer:** Uses `MediaPipe` to track the human index finger in 3D space (XYZ) via a standard 2D webcam.
2.  **Control Layer:** A custom-written **Inverse Kinematics (IK)** solver calculates the required joint angles ($\theta_1, \theta_2$) to reach the target coordinates.
3.  **Actuation Layer:** The system visualizes the mechanical response in real-time, respecting physical constraints like arm length and joint limits.

---

## ⚡ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/mimicry-ai.git](https://github.com/YOUR_USERNAME/mimicry-ai.git)
cd mimicry-ai

```

### 2. Set Up Environment (Recommended)

It is best practice to run this in a virtual environment to keep your system clean.

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

```

### 3. Install Dependencies

We use lightweight, standard libraries to ensure portability.

```bash
pip install -r requirements.txt

```

---

## 🕹️ Usage

Run the core engine. Ensure your webcam is connected.

```bash
python mimicry_core.py

```

### Controls:

* **Move Hand:** The robotic arm will mimic your index finger's position.
* **Safety Range:** If you move your hand beyond the robot's physical length, the system will lock at max extension and warn **"OUT OF RANGE"**.
* **ESC:** Press `Esc` to exit the simulation.

---

## 🧠 The Math: Inverse Kinematics

Unlike basic motion tracking, Mimicry.ai implements physics-based kinematics. We calculate the joint angles using the **Law of Cosines** to solve the geometric triangle formed by the arm links.

Given a target :

1. **Elbow Angle ():**
$$ \cos(\theta_2) = \frac{x^2 + y^2 - L_1^2 - L_2^2}{2 L_1 L_2} $$
2. **Shoulder Angle ():**
Derived from the arctangent of the target vector minus the offset created by the second link.

This ensures the software can drive **real physical motors** (servos/steppers) simply by sending these calculated angles to a microcontroller (ESP32/Arduino).

---

## 🔮 Roadmap

* **Phase 1 (Current):** Vision-based kinematic simulation (The "FounderForge" MVP).
* **Phase 2:** IoT Integration. Sending angle data via WebSocket/MQTT to an ESP32.
* **Phase 3:** Physical Twin. Controlling a 3D-printed robotic arm in real-time.

---


```

```
