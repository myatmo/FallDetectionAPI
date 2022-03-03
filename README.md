# FallDetectionAPI

Intro: <br />
This api is implemented using openpose pose estimation in OpenCV to detect if a person falls. <br />
It uses pre-trained caffee model to simply detect keypoints of a person and alert when a person falls. <br />
Note: the detector is only working for one person now; more work is required to be able to detect for multiple people. <br />
Then, it is combined with Fastapi to stream video to run fall detection algorithm. <br />

To run: <br />
pip install -r requirements.txt <br />
python app/main.py <br />

Quick start: <br />
Go to local host and choose a video to upload (sample videos can be found under app/examples). <br />
Click on Run Fall Detection and the video will start streaming. <br />
Note: the fall detection algorithm is currently very slow to run. <br />
To test running webcam, simply click on Run Webcam. The webcam will not run fall detection algorithm. <br />
Instead it only runs a simple movement detection. <br />