from deepface import DeepFace
import os
import csv

unknown_path = "/scratch-shared/LIMG0038/images/img_final"
known_path = "/scratch-shared/LIMG0038/images/img_original"
# unknown_path = "/scratch-shared/LIMG0038/images/adv_then_ials/test/unknown"
# known_path = "/scratch-shared/LIMG0038/images/adv_then_ials/test/known"
image_names = os.listdir(unknown_path)
metrics = ["cosine", "euclidean", "euclidean_l2"]

csv_data = []
for image_name in image_names:
    verification_result = DeepFace.verify(img1_path = unknown_path + '/' + image_name, 
        img2_path = known_path + '/' + image_name, 
        distance_metric = metrics[1],
        enforce_detection=False
    )

    unknown_attributes = DeepFace.analyze(img_path = unknown_path + '/' + image_name, 
        actions = ['age', 'gender', 'emotion'],
        enforce_detection=False
    )

    known_attributes = DeepFace.analyze(img_path = known_path + '/' + image_name, 
        actions = ['age', 'gender', 'emotion'],
        enforce_detection=False
    )

    csv_data.append([image_name[:-4], verification_result["distance"], 
                    unknown_attributes[0]["age"], 
                    unknown_attributes[0]["dominant_gender"], 
                    unknown_attributes[0]["dominant_emotion"],
                    known_attributes[0]["age"], 
                    known_attributes[0]["dominant_gender"], 
                    known_attributes[0]["dominant_emotion"]                    
                    ])

csv_header = ['unknown_name', 'distance', 'new_age', 'new_gender', 'new_emotion', 'original_age', 'original_gender', 'original_emotion']
with open('/scratch-shared/LIMG0038/csv/test_ials_anonymization.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_data)