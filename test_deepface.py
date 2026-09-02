from deepface import DeepFace

result = DeepFace.analyze(
    img_path="test.jpg",
    actions=['gender'],
    detector_backend='retinaface',
    enforce_detection=True
)
print(result)