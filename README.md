## **Person Counting In region**

# step1- run combined_videos.py file

# step2- run this code in terminal 

python tracking_savetime.py --source "data/output/combined_output.avi" --save-img --view-img  --weights "model/PeopleDetector.pt"


# threading file run

python Multi_threading.py  --source "11.mp4" "12.mp4"  --save-img --view-img  --weights "model/PeopleDetector.pt"


