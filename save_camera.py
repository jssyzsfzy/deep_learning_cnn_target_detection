import cv2


def save_camera(dir):
    cap = cv2.VideoCapture(0)
    num = 87
    while True:
        gra, img = cap.read()
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            print(dir+r'/cam_'+str(num)+'.jpg')
            cv2.imwrite(dir+r'/cam_'+str(num)+'.jpg', img)
            num += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(num)
            break


if __name__ == '__main__':
    save_camera(dir='save_camera')
