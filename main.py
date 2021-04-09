import argparse
import cv2

# detect people and make box
def detect(frame):
    
    # bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    
    # person = 1
    # for x,y,w,h in bounding_box_cordinates:
    #     cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    #     cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #     person += 1
    
    # cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    # cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    
    # TODO: IMPLEMENT PEOPLE DETECTOR
    cv2.imshow('output', frame)

    return frame

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
        #check is True if reading was successful 
        check, frame =  video.read()
        if check:
            # frame = imutils.resize(frame , width=min(800,frame.shape[1]))
            frame = detect(frame)
            
            if writer is not None:
                writer.write(frame)
            
            key = cv2.waitKey(1)
            if key== ord('q'):
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



if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="video file path")
    arg_parse.add_argument("-i", "--image", default=None, help="image file path")
    arg_parse.add_argument("-w", "--webcam", default=False, help="Set true to use webcam")
    arg_parse.add_argument("-o", "--output", type=str, help="output path")
    args = vars(arg_parse.parse_args())

    # TODO: WRITER DOES NOT WORK FOR VIDS
    if args['output'] is not None and args['image'] is None:
        writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))
    else:
      writer = None

    if args['webcam']:
        print('USING WEBCAM')
        useCamera(writer)
    elif args['video'] is not None:
        print('USING VIDEO')
        useVideo(args['video'], writer)
    elif args['image'] is not None:
        print('USING IMAGE')
        useImage(args['image'], args['output'])