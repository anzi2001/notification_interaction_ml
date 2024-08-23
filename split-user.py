from data import *
from sklearn.model_selection import train_test_split
import json
import sys

def split_user(split_num):
    users = import_files()
    device = filter_device(users)
    for index, notifications in enumerate(device):
        user_index = index % split_num
        train_data, test_data = train_test_split(notifications, test_size=0.3)
        with open(f"users/{user_index}_train.json", "w") as f:
            json.dump([notification.to_json() for notification in train_data], f)
        with open(f"users/{user_index}_test.json", "w") as f:
            json.dump([notification.to_json() for notification in test_data], f)


if __name__ == "__main__":
    #get number of users to split
    if len(sys.argv) > 1:
        num_users = int(sys.argv[1])
    else:
        num_users = 12
    

    split_user(num_users)
    print("Done!")