
USAGE:
Video Tracking: 
$python tracking.py File [filename]
Live Camera:
$python tracking.py Live

Procedure of Tracking:
Select ROI bounding box and store in box list.
Create tracker with the same length of box list (support Boosting, MIL, TLD and KCF tracker).
Boosting method has good accuracy.
MIL suitable for fast scene tracking.
TLD has better performance under occlusion over multiple frames.
KCF has the best overall performance (set as default).
Save the first frame as template frame.
Read frames continuous update trackers.
Boxing out the object of interest.
With procedure of tracking, at each frame, SURF key point matching will be applied. 
Procedure of Key Point Matching
Create a SURF matching detector.
Crop sub-image with bonding boxes.
Detect and compute key points and descriptor.
Use KNN to group match points.
Select out good match points by choosing points within threshold distance.
Regulate position of key points to locate correctly on original frame.
Select out source points and descriptor points beyond minimum matches to create mask.
Draw matches on template frame and video frames.
