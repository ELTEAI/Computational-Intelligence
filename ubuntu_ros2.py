import cv2
import mediapipe as mp
import math
import socket
import time

# 初始化 MediaPipe 手部模型和绘图工具 Initialize the MediaPipe hand model and drawing tools
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# 定义虚拟机的IP地址和端口 Define the IP address and port of the virtual machine
VM_IP = "192.168.177.129"  # 替换为你的虚拟机的IP地址 Replace with the IP address of your virtual machine
PORT = 9090
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 初始化视频流 Initialize video stream
cap = cv2.VideoCapture(0)

# 用于存储上一帧的拇指位置和上一次动作时间 Used to store the thumb position and last action time of the previous frame
prev_thumb_x = None
last_action_time = time.time()  # 初始化为当前时间 Initialize to current time


def detect_gesture(hand_landmarks, frame_width, frame_height):
    """根据手势轨迹生成控制指令。 Generate control instructions based on gesture trajectory"""
    global prev_thumb_x
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_x = int(thumb_tip.x * frame_width)
    action = None

    # 检测左挥和右挥 Detect left and right swipes
    if prev_thumb_x is not None:
        if thumb_x - prev_thumb_x > 20:
            action = "right"  # 右挥指令 Right Swing Command
        elif prev_thumb_x - thumb_x > 20:
            action = "left"  # 左挥指令 Left Swipe Command

    prev_thumb_x = thumb_x

    # 检测握拳 Detecting clenched fist
    if is_fist(hand_landmarks, frame_width, frame_height):
        action = "stop"  # 握拳指令  Detecting clenched fist

    return action


def is_fist(hand_landmarks, frame_width, frame_height):
    """检测是否握拳，通过判断每根手指是否弯曲。"""
    """Detect whether a fist is made by determining whether each finger is bent."""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    def to_pixel(landmark):
        return int(landmark.x * frame_width), int(landmark.y * frame_height)

    index_tip_pos = to_pixel(index_tip)
    index_pip_pos = to_pixel(index_pip)

    # 判断食指是否弯曲（握拳时食指会弯曲）Determine whether the index finger is bent (the index finger will bend when making a fist)
    threshold = 30  # 弯曲判定距离阈值 Bend determination distance threshold
    return math.hypot(index_tip_pos[0] - index_pip_pos[0], index_tip_pos[1] - index_pip_pos[1]) < threshold


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 翻转图像，使其更符合镜像效果 Flip the image
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # 检测手部 Detecting hands
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    action = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 检测手势并获取相应的控制指令 Detect gestures and obtain corresponding control instructions
            detected_action = detect_gesture(hand_landmarks, frame.shape[1], frame.shape[0])
            if detected_action:
                action = detected_action  # 如果检测到手势，则更新动作指令 If a gesture is detected, update the action instructions
                last_action_time = time.time()  # 更新上一次动作时间 Update last action time

            # 绘制手部关键点和骨架 Draw the hand key points and skeleton
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 如果超过5秒未检测到任何手势，则发送“未行动”指令 If no gesture is detected for more than 5 seconds, a "no action" command is sent
    if action is None and (time.time() - last_action_time > 5):
        action = "no_action"  # 设置未行动指令 Set unactioned command
        last_action_time = time.time()  # 更新上一次动作时间 Update last action time

    # 发送控制指令到虚拟机 Send control instructions to the virtual machine
    if action:
        sock.sendto(action.encode(), (VM_IP, PORT))
        print(f"Sent action: {action}")

    # 显示结果 显示结果
    cv2.imshow('Hand Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
