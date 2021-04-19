face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
vs = VideoStream(src=0).start()
start = time.perf_counter() 
data = []
time_value = 0
if args["save"]:    
    out = cv2.VideoWriter(path + "liveoutput.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 10, (450,253))
