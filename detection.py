"""
https://github.com/shanren7/real_time_face_recognition
"""

import sys
import time

import cv2

from facenet.face_contrib import *
import sounddevice as sd
import soundfile as sf

def add_overlays(frame, faces, frame_rate, colors, confidence=0.3):
    if faces is not None:
        for idx, face in enumerate(faces):
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), colors[idx], 2)
            # ProbFace - это надежный метод Probabilistic Face Embeddging (PFE)
            if face.name and face.prob:
                if face.prob > confidence:
                    class_name = face.name
                    playsound = 'Vubasang\Vubasang.wav'
                else:
                    class_name = 'Unknown'

                cv2.putText(frame, class_name, (face_bb[0], face_bb[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            colors[idx], thickness=2, lineType=2)
                # cv2.putText(frame, '{:.02f}'.format(face.prob * 100), (face_bb[0], face_bb[3] + 40),
                cv2.putText(frame, '{:.02f}'.format(face.prob), (face_bb[0], face_bb[3] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], thickness=1, lineType=2)
                data, fs = sf.read(playsound, dtype='float32')
                sd.play(data, fs)
                status = sd.wait()

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                thickness=2, lineType=2)

def run(model_checkpoint, classifier, video_file=None, output_file=None):
    frame_interval = 3  # Number of frames after which to run face detection (Количество кадров)
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    if video_file is not None:
        video_capture = cv2.VideoCapture(video_file)
    else:
        # Использовать камеру
        video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    width = frame.shape[1]
    height = frame.shape[0]
    if output_file is not None:
        video_format = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, video_format, 20, (width, height))
    face_recognition = Recognition(model_checkpoint, classifier)
    start_time = time.time()
    colors = np.random.uniform(0, 255, size=(1, 3))
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # a % b = {P}, где {P} - это остаток от деления a на b
        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)
            for i in range(len(colors), len(faces)):
                colors = np.append(colors, np.random.uniform(150, 255, size=(1, 3)), axis=0)
            # Проверьте свой текущий fps
            # fps - Frames Per Second - количество кадров в секунду. (Кадровая частота)
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, frame_rate, colors)

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count += 1
        # cv2.imshow('frame', gray)
        cv2.imshow('Video', frame)
        if output_file is not None:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    if output_file is not None:
        out.release()
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run('models', 'models/your_model.pkl', video_file=None, output_file='taolatao.avi')