import pandas as pd
from collections import Counter

# CSV 파일 경로 설정 (file.csv 파일 경로에 맞게 수정해주세요)
file_path_base = 'outputs/'

vote_dict = {}

for i in range(7):
    file_path = file_path_base + 'output' + str(i) + '.csv'
    print(file_path)

    # CSV 파일 불러오기
    data = pd.read_csv(file_path)

    data_list = data.values.tolist()

    for data in data_list:
        if data[0] not in vote_dict.keys():
            vote_dict[data[0]] = []
        vote_dict[data[0]].append(data[1])

# 주어진 가중치
class_weights = {
    0: 18900 / 2745,  # Class 0
    1: 18900 / 1570,  # Class 1
    2: 18900 / 895,   # Class 2
    3: 18900 / 3660,  # Class 3
    4: 18900 / 3345,  # Class 4
    5: 18900 / 1285,  # Class 5
    6: 18900 / 549,   # Class 6
    7: 18900 / 314,   # Class 7
    8: 18900 / 179,   # Class 8
    9: 18900 / 732,   # Class 9
    10: 18900 / 669,  # Class 10
    11: 18900 / 257,  # Class 11
    12: 18900 / 549,  # Class 12
    13: 18900 / 314,  # Class 13
    14: 18900 / 179,  # Class 14
    15: 18900 / 732,  # Class 15
    16: 18900 / 669,  # Class 16
    17: 18900 / 257,  # Class 17
}
final_predictions = {}
cnt = 0

# 각 이미지에 대해 다수결 투표로 최종 예측값 계산
for image, predictions in vote_dict.items():
    counter = Counter(predictions)  # 각 클래스별 빈도 계산
    most_common = counter.most_common()  # 가장 많이 예측된 클래스 및 빈도수 확인

    # 동률이 발생했는지 확인
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        print()
        print(f"동률 발생! 이미지: {image}, 예측값 후보: {most_common}")
        print("가중치를 적용하여 선택합니다.")

        weighted_classes = {}  # 각 클래스별 가중합을 저장할 딕셔너리
        for class_pred, count in most_common:
            weighted_classes[class_pred] = sum([class_weights[class_pred] for _ in range(count)])
        print(weighted_classes)

        # 가장 높은 가중치를 가진 클래스 선택
        final_prediction = max(weighted_classes, key=weighted_classes.get)
        final_predictions[image] = final_prediction
        print('선택된 값: ', final_prediction)
        # final_predictions[image] = most_common[0][0]
        cnt += 1
    else:
        # 동률이 아닌 경우 가장 빈도가 높은 클래스 선택
        final_predictions[image] = most_common[0][0]

# 최종 예측값 출력
print()
print('동률 데이터 수: ',cnt)

# DataFrame 생성
data_df = pd.DataFrame(list(final_predictions.items()), columns=['ImageID', 'ans'])

# CSV 파일로 저장
data_df.to_csv('predictions.csv', index=False)