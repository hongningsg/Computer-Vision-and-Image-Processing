#reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
import numpy as np
import cv2
import sys

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
colors = [red, green, blue]

def TrackerDefine(t_type = 'KCF'):
    if t_type == 'boosting':
        tracker = cv2.TrackerBoosting_create()
    elif t_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif t_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    else:
        tracker = cv2.TrackerKCF_create()
    return tracker

def initialVideo(filename, t_type = 'KCF'):
    video = cv2.VideoCapture(filename)
    if filename == 0:
        while True:
            ret, frame = video.read()
            cv2.imshow("Press A to start", frame)
            k = cv2.waitKey(1)
            if k == ord('a'):
                cv2.destroyAllWindows()
                break
    success, firstframe = video.read()
    multiTracker = []
    boxes = cv2.selectROIs("Select", firstframe, False, False)

    for box in boxes:
        box = tuple(box)
        tracker = TrackerDefine(t_type)
        tracker.init(firstframe, box)
        multiTracker.append(tracker)
    cv2.destroyAllWindows()
    return video, multiTracker[:3], boxes[:3], firstframe

def drawline(frame, points, color, thickness = 5):
    point = points[0]
    for i in range(1, len(points)):
        newpoint = points[i]
        cv2.line(frame, point, newpoint, color, thickness)
        point = newpoint

def keypointMatch(firstframe, frame, box, POLY = False):
    surf = cv2.xfeatures2d.SURF_create()
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    good_match_set = []
    kps = []
    kp_num = 0
    if len(box) == 0:
        return frame
    for i, b in enumerate(box):
        frame_kp, frame_des = surf.detectAndCompute(frame, None)
        kp, des = surf.detectAndCompute(firstframe[b[1]:b[1]+b[3], b[0]:b[0]+b[2], :], None)
        match = flann.knnMatch(des, frame_des, k=2)
        good_match = []
        for m, n in match:
            if m.distance < 0.8 * n.distance:
                good_match.append(m)
        if POLY and len(good_match) > 10:
            sub_src_pts = np.float32([kp[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
            sub_dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(sub_src_pts, sub_dst_pts, cv2.RANSAC, 5.0)
            h, w, c = firstframe[b[1]:b[1]+b[3], b[0]:b[0]+b[2], :].shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            cv2.polylines(frame, [np.int32(dst)], True, colors[i], 3, cv2.LINE_AA)
        for g in good_match:
            g.queryIdx += kp_num
        kp_num += len(kp)
        good_match_set += good_match
        for points in kp:
            points.pt = (points.pt[0] + b[0], points.pt[1] + b[1])
        kps += kp
    src_pts = np.float32([kps[m.queryIdx].pt for m in good_match_set]).reshape(-1, 1, 2)
    dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_match_set]).reshape(-1, 1, 2)
    homo, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()
    draw_params = dict(matchColor=(30, 0, 150), singlePointColor=None, matchesMask=matches_mask, flags=2)
    concat = cv2.drawMatches(firstframe, kps, frame, frame_kp, good_match_set, None, **draw_params)
    concat = cv2.resize(concat, (int(concat.shape[1]/2), int(concat.shape[0]/2)))
    return concat

def tracking(video, multiTracker, box, firstframe, drawproj = False):
    firstbox = []
    pointset = []
    for b in box:
        firstbox.append(b)
        pointset.append([(int(b[0] + b[2]/2), int(b[1] + b[3]/2))])
    while video.isOpened():
        still, frame = video.read()
        if not still:
            break
        timer = cv2.getTickCount()
        for i, tracker in enumerate(multiTracker):
            success, box = tracker.update(frame)
            if success:
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
                if drawproj:
                    pointset[i].append((int(box[0] + box[2]/2), int(box[1] + box[3]/2)))
                    drawline(frame, pointset[i], colors[i])
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        cv2.putText(frame, "FPS : " + str(int(fps)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        concat = keypointMatch(firstframe, frame, firstbox)
        cv2.imshow('MultiTracker', concat)
        k = cv2.waitKey(1)
        if k == 27:
            sys.exit(1)
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tp = sys.argv[1]
    if tp == "Live":
        file = 0
    elif tp == "File":
        file = sys.argv[2]
    else:
        print("USAGE: Live for camera, File filename for video.")
    video, multiTracker, boxes, firstframe = initialVideo(file)
    tracking(video, multiTracker, boxes, firstframe)
