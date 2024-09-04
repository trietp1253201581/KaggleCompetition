import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def view_feature_importances(
    df: pd.DataFrame,
    feature: str,
    target: str,
    is_numeric: bool=False,
    bins: int|list|None=None,
    labels: list|None=None
) -> pd.DataFrame:
    """
    Xem xét sự phụ thuộc của `df[target]` vào `df[feature]`.
    
    Parameters
    ----------
    df: DataFrame
        DataFrame để tham chiếu mối quan hệ
    feature: str
        Tên tính năng để xem xét mối quan hệ với cột đích.
    target: str
        Tên cột đích (cột cần dự đoán), mang dữ liệu phân loại
    is_numeric: bool, default=False
        Lựa chọn xem tính năng là numeric hay categorical:
        - Nếu là `False`, hàm sẽ đếm tỉ lệ phần trăm của mỗi nhãn đích
        trong mỗi nhãn của `feature`.
        - Nếu là `True`, hàm sẽ đem feature này phân ra tùy theo
        `bins` rồi chuyển về như làm với dữ liệu categorical.
    split_numeric_method: str or None, default=None
    bins: int or list or None, default=None
        Số lượng bins để phân chia tính năng thành các dữ liệu categorical:
        - Nếu là `int` >= 2, hàm sẽ phân chia chính xác cột này thành `bins` labels.
        - Nếu là `list` với len()>=3, hàm sẽ chia cột này thành các phần có các đầu mút là
        các phần tử của `list`.
        - Nếu là `None`: Nếu feature này là numeric thì chia thành 10 bin, còn 
        nếu feature này là categorical thì số bin chính là số label. 
    labels: list or None, default=None
        Nếu là list, các label này được đánh lần lượt cho các bin, nếu là None
        thì label sẽ được đánh tự động theo thứ tự bin.
        
    Returns
    -------
    out_df: DataFrame
    Một DataFrame miêu tả mối quan hệ phụ thuộc
    """
    df = df[[feature, target]].copy()
    if isinstance(bins, int) and bins>=2:
        if labels is None:
            labels = [i for i in range(bins)]
        else:
            if len(labels) > bins:
                labels = labels[:bins]
            elif len(labels) < bins:
                add_labels = [i for i in range(len(labels), bins)]
                labels += add_labels
        df[f'cut_of_{feature}'] = pd.cut(df[feature], bins=bins, labels=labels)
    elif isinstance(bins, list) and len(bins)>=3:
        num_bins = len(bins) - 1
        if labels is None:
            labels = [i for i in range(num_bins)]
        else:
            if len(labels) > num_bins:
                labels = labels[:num_bins]
            elif len(labels) < num_bins:
                add_labels = [i for i in range(len(labels), num_bins)]
                labels += add_labels
        df[f'cut_of_{feature}'] = pd.cut(df[feature], bins=bins, labels=labels)
    else:
        if is_numeric:
            num_bins = 10
            if labels is None:
                labels = [i for i in range(num_bins)]
            else:
                if len(labels) > num_bins:
                    labels = labels[:num_bins]
                elif len(labels) < num_bins:
                    add_labels = [i for i in range(len(labels), num_bins)]
                    labels += add_labels
            df[f'cut_of_{feature}'] = pd.cut(df[feature], bins=bins, labels=labels)
        else:
            df[f'cut_of_{feature}'] = df[feature]
    
    feature_dict = dict()
    total = df[f'cut_of_{feature}'].value_counts()
    view_idx = df[target].value_counts().index.tolist()
    for idx in view_idx:
        label = df[df[target]==idx][f'cut_of_{feature}'].value_counts()
        feature_dict[idx] = label/total*100
    
    out_df = pd.DataFrame(feature_dict, columns=view_idx)
    return out_df

def draw_numeric_features(
    df: pd.DataFrame, 
    feature: str, 
    target: str, 
    overlap: list|None = None
) -> None:
    """
    Xem xét sự phụ thuộc của `df[target]` vào `df[feature]` qua biểu đồ line.
    
    Parameters
    ----------
    df: DataFrame
        DataFrame để tham chiếu mối quan hệ
    feature: str
        Tên tính năng để xem xét mối quan hệ với cột đích.
    target: str
        Tên cột đích (cột cần dự đoán), mang dữ liệu phân loại
    overlap: default=None
        Nếu `overlap` không phải `None` thì sẽ vẽ chồng các biểu đồ
        với từng phần từ trong overlap
    """
    plt.figure(figsize=(30,10))
    plt.grid()
    sns.lineplot(data=df, x=feature, y=target)
    if overlap is not None:
        for plot in overlap:
            sns.lineplot(data=plot[0], x=plot[1], y=plot[2])
            
def read_csv_data(
    data_dir: str = 'data/',
    train_file_name: str = 'train.csv',
    test_file_name: str = 'test.csv',
    submission_file_name: str = 'sample_submission.csv'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Đọc file csv để có các data frame
    
    Parameters
    ----------
    data_dir: str, default='data'
        Đường dẫn đến thư mục
    train_file_name: str, default='train.csv'
        Tên file train
    test_file_name: str, default='test.csv'
        Tên file test
    submission_file_name: str, default='sample_submission.csv'
        Tên file submit
        
    Returns
    -------
    (train_df, test_df, sub_df): Một tuple chứa 3 dataframe được đọc
    """
    train_df = pd.read_csv(data_dir + train_file_name)
    test_df = pd.read_csv(data_dir + test_file_name)
    sub_df = pd.read_csv(data_dir + submission_file_name)
    return train_df, test_df, sub_df

def get_target_interact_feature(
    features: str|list[str],
    target: str,
    method: str = 'mean',
    
):
