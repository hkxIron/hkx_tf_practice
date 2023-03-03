import pandas as pd
import numpy as np
import os
def get_cols(df):
    return [col for col in df.columns]

def merge_data(data_path):
    print("data path:" + data_path)
    dfs = []
    count = 0
    for root_dir, sub_dir, files in os.walk(data_path):
        for file in files:
            if file.endswith(".xls"):
                file_name = os.path.join(root_dir, file)
                print("merge df:"+ file_name)
                df = pd.read_excel(file_name)
                print("old columns:"+ ",".join(get_cols(df)))
                df.rename(columns={'说明': '举例', '公布时间':'答复时间'}, inplace=True)
                print("new columns:"+ ",".join(get_cols(df)))
                print("data shape:{}".format(df.shape))
                dfs.append(df)
                count += df.shape[0]

    df_conated = pd.concat(dfs)
    df_conated.sort_values(by='涉及报表编号', inplace=True, ascending=True)
    df_conated["新序号"] = np.arange(1, df_conated.shape[0]+1)
    out_file= "merged.xlsx"
    df_conated.to_excel(out_file, sheet_name="sheet1", index=None)
    print("output file:"+ out_file + " shape:"+ str(df_conated.shape) + " expect count:" + str(count))

if __name__ == "__main__":
    merge_data("C:\\Users\\kexin\\Desktop\\data\\")