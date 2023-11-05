import os
import pandas as pd
# read parameters from cli
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, default='../amlt/smoothing/cifar100_r152gn1_n025_lr01e100/cifar100/cifar100_r152gn1_n025_lr01e100')

path = parser.parse_args().path

files = ['certify_sigma0.25_test_ra.csv',
         'certify_sigma0.5_test_ra.csv',
         'certify_sigma1.0_test_ra.csv',]

# merge three csv files into one, using the same head
df = pd.DataFrame()
for idx, file in enumerate(files):
    file_path = os.path.join(path, file)
    print(file)
    # read csv file
    df_temp = pd.read_csv(file_path)
    # print(df_temp)
    if idx == 0:
        df = df_temp
    else:
    # ignore the first column
        df = pd.concat([df, df_temp.iloc[:, 1:]], axis=0)
print(df)


# l0 = []
# l025 = []
# l05 = []
# l075 = []
# l1 = []

# for file in files:
#     file_path = os.path.join(path, file)
#     print(file)
#     # read csv file
#     df = pd.read_csv(file_path)

#     l0.append(df['0.0'].values.tolist()[0])
#     l025.append(df['0.25'].values.tolist()[0])
#     l05.append(df['0.5'].values.tolist()[0])
#     l075.append(df['0.75'].values.tolist()[0])
#     l1.append(df['1.0'].values.tolist()[0])

# # print(l0)
# # print(l025)
# # print(l05)
# # print(l075)
# # print(l1)

# print('radius 0.0: ', max(l0))
# print('radius 0.25: ', max(l025))
# print('radius 0.5: ', max(l05))
# print('radius 0.75: ', max(l075))
# print('radius 1.0: ', max(l1))

            
            