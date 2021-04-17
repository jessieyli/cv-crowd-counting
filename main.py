import argparse
import cv2
import numpy as np
from yolo3 import YoloV3, load_pretrained_weights, weights_download

# detect people and make box
yolo = YoloV3()
load_pretrained_weights(yolo, 'models/yolov3.weights')


def detect(frame):
    image = frame

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    img = img / 255

    print("running yolo")
    boxes, scores, classes, nums = yolo(img)
    count = 0
    display_boxes = []
    display_scores = []
    for i in range(nums[0]):
        if int(classes[0][i] == 0):
            count += 1
            display_boxes.append(boxes[0][i])
            display_scores.append(scores[0][i])

    # draw output
    wh = np.flip(image.shape[0:2])
    for i in range(len(display_boxes)):
        x1y1 = tuple((np.array(display_boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(display_boxes[i][2:4]) * wh).astype(np.int32))
        image = cv2.rectangle(image, x1y1, x2y2, (255, 0, 0), 2)
        image = cv2.putText(image, '{} {:.2f}'.format("person", display_scores[i]),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    cv2.putText(image, 'Status : Detecting ', (40, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f'Total Persons : {count}', (40, 70),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('output', image)
    return image


def useCamera(writer):
    video = cv2.VideoCapture(0)
    print('Detecting people...')
    while True:
        check, frame = video.read()
        frame = detect(frame)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def useVideo(path, writer):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return
    print('Detecting people...')
    while video.isOpened():
        # check is True if reading was successful
        check, frame = video.read()
        if check:
            # frame = imutils.resize(frame , width=min(800,frame.shape[1]))
            frame = detect(frame)

            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()


def useCamera(writer):
    video = cv2.VideoCapture(0)
    print('Detecting people...')
    while True:
        check, frame = video.read()
        frame = detect(frame)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def useImage(path, output_path):
    image = cv2.imread(path)

    # image = imutils.resize(image, width = min(800, image.shape[1]))

    result_image = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def useImageYolo(path, output_path):
    weights_download()

    yolo = YoloV3()
    load_pretrained_weights(yolo, 'models/yolov3.weights')

    print("reading image")
    image = cv2.imread(path)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    img = img / 255

    print("running yolo")
    boxes, scores, classes, nums = yolo(img)
    count = 0
    display_boxes = []
    display_scores = []
    for i in range(nums[0]):
        if int(classes[0][i] == 0):
            count += 1
            display_boxes.append(boxes[0][i])
            display_scores.append(scores[0][i])

    # draw output
    wh = np.flip(image.shape[0:2])
    for i in range(len(display_boxes)):
        x1y1 = tuple((np.array(display_boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(display_boxes[i][2:4]) * wh).astype(np.int32))
        image = cv2.rectangle(image, x1y1, x2y2, (255, 0, 0), 2)
        image = cv2.putText(image, '{} {:.2f}'.format("person", display_scores[i]),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    cv2.putText(image, 'Status : Detecting ', (40, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f'Total Persons : {count}', (40, 70),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    # cv2.imshow('output', image)

    if output_path is not None:
        cv2.imwrite(output_path, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None,
                           help="video file path")
    arg_parse.add_argument("-i", "--image", default=None,
                           help="image file path")
    arg_parse.add_argument("-w", "--webcam", default=False,
                           help="Set true to use webcam")
    arg_parse.add_argument("-o", "--output", type=str, help="output path")
    arg_parse.add_argument("-y", "--yolo", action="store_true", help="use YOLOv3")
    args = vars(arg_parse.parse_args())

    # TODO: WRITER DOES NOT WORK FOR VIDS
    if args['output'] is not None and args['image'] is None:
        writer = cv2.VideoWriter(
            args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))
    else:
        writer = None


    if args['webcam']:
        print('USING WEBCAM')
        useCamera(writer)
    elif args['video'] is not None:
        print('USING VIDEO')
        useVideo(args['video'], writer)
    elif args['image'] is not None and args['yolo']:
        print('USING IMAGE W YOLO')
        useImageYolo(args['image'], args['output'])
    elif args['image'] is not None:
        print('USING IMAGE')
        useImage(args['image'], args['output'])