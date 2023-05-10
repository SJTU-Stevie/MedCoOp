from PIL import Image
from Medclip import dataset
from Medclip import Medclip
import os
os.chdir('../')

processor = dataset.MedCLIPProcessor()

image = Image.open('./example_data/view1_frontal.jpg')
inputs = processor(
    text=["opacity left costophrenic angle is new since prior exam ___ represent some loculated fluid cavitation unlikely"],
    images=image,
    return_tensors="pt",
    padding=True
)

print(inputs)

model = Medclip.MedCLIPModel(vision_cls=Medclip.MedCLIPVisionModel)
model.from_pretrained()
model.cuda()
outputs = model(**inputs)
print(outputs)