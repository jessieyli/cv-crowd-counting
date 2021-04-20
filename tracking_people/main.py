import argparse
import cv2
import numpy as np
from yolo3 import YoloV3, weights_download
from tiny_yolo import YoloV3Tiny, tiny_weights_download
from helpers import load_pretrained_weights
import os
from skimage.feature import hog
from match import match_features
import random

# constants
CONFIDENCE_THRESHOLD = 1.05

# detect people and make box
yolo = YoloV3()
load_pretrained_weights(yolo, 'models/yolov3.weights')
people_count = 0
people_colors = {}

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))



def detect(frame):
    image = frame

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    img = img / 255

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
    return image

# returns image, hog descriptors of people
# frame - image frame
# prevFeatures - array of descriptors
# prevLabels - array of labels corresponding to descriptors
def detectTrack(frame, prevFeatures, prevLabels):
    global people_count
    image = frame.copy()

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    img = img / 255

    boxes, scores, classes, nums = yolo(img)
    count = 0
    display_boxes = []
    display_scores = []
    for i in range(nums[0]):
        if int(classes[0][i] == 0):
            count += 1
            display_boxes.append(boxes[0][i])
            display_scores.append(scores[0][i])
    
    # extract features
    wh = np.flip(image.shape[0:2])
    features = np.zeros((len(display_boxes), 3969))
    labels = []

    # draw output
    for i in range(len(display_boxes)):
        x1, y1 = tuple((np.array(display_boxes[i][0:2]) * wh).astype(np.int32))
        x2, y2 = tuple((np.array(display_boxes[i][2:4]) * wh).astype(np.int32))

        cx = x1 + ((x2 - x1) // 2)
        cy = y1 + ((y2 - y1) // 2)
        
        fx1 = cx - 36
        fy1 = cy - 36
        fx2 = cx + 36
        fy2 = cy + 36

        # draw feature
        image = cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 0, 255), 1)

        # calculate and save feature
        if fy1 < 0 or fy2 >= image.shape[0] or fx1 < 0 or fx2 >= image.shape[1]:
            continue
        feature = hog(image[fy1:fy2, fx1:fx2], feature_vector=True)
        features[i] = feature

        # # draw bounding box
        # image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # image = cv2.putText(image, '{} {:.2f}'.format("person", display_scores[i]),
        #                   (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    if prevFeatures is not None:
        matches, confidences = match_features(features, prevFeatures)
        print(confidences)
        for match in matches:
            index = match[0]
            confidence = confidences[index]
            x1, y1 = tuple((np.array(display_boxes[index][0:2]) * wh).astype(np.int32))
            x2, y2 = tuple((np.array(display_boxes[index][2:4]) * wh).astype(np.int32))

            
            if confidence > CONFIDENCE_THRESHOLD:
                # matched
                matchedIndex = match[1]
                labels.append(prevLabels[matchedIndex])
                image = cv2.rectangle(image, (x1, y1), (x2, y2), people_colors[prevLabels[matchedIndex]], 2)
                image = cv2.putText(image, '{} {:.2f}'.format("person", prevLabels[matchedIndex]),
                                (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            else:
                # not matched
                labels.append(people_count)
                people_colors[people_count] = random_color()
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                image = cv2.putText(image, '{} {:.2f}'.format("person", people_count),
                                (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                people_count += 1
    else:
        for j in range(len(display_boxes)):
            x1, y1 = tuple((np.array(display_boxes[j][0:2]) * wh).astype(np.int32))
            x2, y2 = tuple((np.array(display_boxes[j][2:4]) * wh).astype(np.int32))

            labels.append(people_count)
            people_colors[people_count] = random_color()

            # draw bounding box
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            image = cv2.putText(image, '{} {:.2f}'.format("person", people_count),
                          (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            people_count += 1

    cv2.putText(image, 'Status : Detecting ', (40, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f'Total Persons : {count}', (40, 70),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    # cv2.imshow('output', image)
    return image, features, labels



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
    

    # yolo = YoloV3()
    # weights_download()
    # load_pretrained_weights(yolo, 'models/yolov3.weights')

    yolo = YoloV3Tiny()
    tiny_weights_download()
    load_pretrained_weights(yolo, 'models/yolov3-tiny.weights', tiny=True)

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

def imagesIntoVideoTrack(image_folder, video_name):
    images = [img for img in os.listdir(image_folder)
            if img.endswith(".jpg") or
                img.endswith(".jpeg") or
                img.endswith("png")]

    # setting frame width, height
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 30, (width, height))

    # write images into video
    prevFeatures = None
    prevLabels = None
    for image in images: 
        # process images
        raw = cv2.imread(os.path.join(image_folder, image))
        processed, newFeatures, newLabels = detectTrack(raw, prevFeatures, prevLabels)
        prevFeatures = newFeatures
        prevLabels = newLabels
        video.write(processed)

    # cleanup
    cv2.destroyAllWindows()
    video.release()

def imagesIntoVideo(image_folder, video_name):
    images = [img for img in os.listdir(image_folder)
            if img.endswith(".jpg") or
                img.endswith(".jpeg") or
                img.endswith("png")]

    # setting frame width, height
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 30, (width, height))

    # write images into video
    for image in images: 
        # process images
        raw = cv2.imread(os.path.join(image_folder, image))
        processed = detect(raw)
        video.write(processed)

    # cleanup
    cv2.destroyAllWindows()
    video.release()



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
    arg_parse.add_argument("-if", "--imagefolder", default=None,
                           help="image folder to video")
    arg_parse.add_argument("-t", "--track", action="store_true",
                           help="track people")
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
    elif args['imagefolder'] is not None and args['track']:
        imagesIntoVideoTrack(args['imagefolder'], args['output'])
    elif args['imagefolder'] is not None:
        imagesIntoVideo(args['imagefolder'], args['output'])
        

