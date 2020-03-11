
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
 

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--label", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())
 

print("[INFO] encoding labels...")
label = LabelEncoder()
labels = label.fit_transform(data["names"])

print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()
 
f = open(args["label"], "wb")
f.write(pickle.dumps(label))
f.close()
