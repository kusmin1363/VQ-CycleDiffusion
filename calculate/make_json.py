import json


for epoch in range(5, 301, 5):
    epoch = str(epoch)
    model_name = 'last_train_dec_2speakers_cycle6'
    # last_train_dec_2speakers_cycle6
    # last_train_dec_2speakers_cycle8
    # last_train_dec_2speakers_cycle_linear
    seenORunseen = 'seen'
    output_last_dir =  seenORunseen + '_' + model_name + '_' + epoch

    filename = f"mcd_{output_last_dir}.json"
    data = {
        "mean_mcd": "0",
        "std_mcd": "0"
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)  # JSON 형식으로 데이터를 파일에 씁니다.
