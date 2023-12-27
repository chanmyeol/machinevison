import csv

# 输入和输出文件路径
input_file_path = 'train_ground_truth.txt'
output_csv_path = 'train_ground_truth.csv'

# 打开文本文件以及 CSV 文件进行读写
with open(input_file_path, 'r') as txt_file, open(output_csv_path, 'w', newline='') as csv_file:
    # 创建 CSV 写入器
    csv_writer = csv.writer(csv_file)

    # 遍历文本文件的每一行
    for line in txt_file:
        # 使用逗号分隔行中的数据，并写入 CSV 文件
        csv_writer.writerow(line.strip().split(','))

print(f'转换完成，CSV 文件保存在 {output_csv_path}')
