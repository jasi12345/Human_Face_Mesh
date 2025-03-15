import cv2
import mediapipe as mp
import numpy as num

mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_face_mesh=mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

video=cv2.VideoCapture(r"C:\Users\Keanu\Downloads\Realtime_face_Mesh\expression.mp4")

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:

    while(video.isOpened()):
        ret,frame= video.read()

        if  not ret:
            print("Frame is empty")
            break
        
        original_frame=frame.copy()

        frame.flags.writeable=False
        frame= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        transparent_frame=num.zeros((frame.shape[0],frame.shape[1],3),dtype=num.uint8)

        results=face_mesh.process(frame)

         
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                
                mp_drawing.draw_landmarks(
                    transparent_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                mp_drawing.draw_landmarks(
                    transparent_frame,
                    face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                
                mp_drawing.draw_landmarks(
                    transparent_frame,
                    face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                
        cv2.imshow("originl",original_frame)
        cv2.imshow("mesh",transparent_frame)

        if cv2.waitKey(1) & 0xFF ==27:
            break

video.release()
cv2.destroyAllWindows()
                
                



