from joblib import load
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize

model = load('GA SVM.joblib')

url = input('Enter URL of Image :')
img = imread(url)
plt.imshow(img)
plt.show()

img_resize = resize(img,(150,150,3))
l = [img_resize.flatten()]
probability = model.predict_proba(l)

categories = ["glioma", "meningioma", "notumor", "pituitary"]

for ind, val in enumerate(categories):
    print(f'{val} = {probability[0][ind]*100}%')
    print("The predicted image is : " + categories[model.predict(l)[0]])
