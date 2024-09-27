import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def view_cat_target_feature_corr(
    df: pd.DataFrame,
    feature: str,
    target: str,
    is_numeric: bool=False,
    bins: int|list|None=None,
    labels: list|None=None
) -> pd.DataFrame:
    """
    Xem xét sự phụ thuộc của `df[target]` vào `df[feature]`, trong đó
    target là một category, còn feature có thể là dữ liệu numeric hoặc category.
    
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
    df = df[df[target].isna()==False]
    if is_numeric:
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

def view_numeric_target_feature_corr(
    df: pd.DataFrame,
    feature: str,
    target: str,
    strategy: str|list[str]='mean',
    is_numeric: bool=False,
    bins: int|list|None=None,
    labels: list|None=None
) -> pd.DataFrame:
    """
    Xem xét sự phụ thuộc của `df[target]` vào `df[feature]`, trong đó
    target là một category, còn feature có thể là dữ liệu numeric hoặc category.
    
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
    strategy: str, default='mean'
        Cách để kết số liệu target của một phân nhóm trong feature, ví dụ như `mean`,
        `std`, `count`, `min`, etc.
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
    df = df[df[target].isna()==False]
    if is_numeric:
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

    if isinstance(strategy, str):
        strategys = [strategy]
    else:
        strategys = strategy
    return df.groupby(f'cut_of_{feature}')[target].aggregate(strategys)

def plot_numeric_target_numeric_feature(
    df: pd.DataFrame, 
    feature: str, 
    target: str,    
    overlap: list|None = None
) -> None:
    """
    Xem xét sự phụ thuộc của `df[target]` vào `df[feature]` qua biểu đồ line,
    trong đó target và feature đều là dữ liệu numeric
    
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

def cross_validation(
    model,
    cv,
    train_inputs: pd.DataFrame,
    train_targets: pd.Series|pd.DataFrame,
    num_classes: int|None=None,
    test_inputs: pd.DataFrame|None=None,
    need_print: bool=False
):
    train_scores = []
    val_scores = []
    if test_inputs is not None:
        if num_classes is 
    else:
        test_preds = np.zeros((len(inputs), num_classes))
    for fold, (train_idx, val_idx) in enumerate(cv.split(inputs, targets)):
        X_train = inputs.iloc[train_idx].reset_index(drop=True)
        y_train = targets.iloc[train_idx].reset_index(drop=True)
        X_val = inputs.iloc[val_idx].reset_index(drop=True)
        y_val = targets.iloc[val_idx].reset_index(drop=True)

        clf.fit(X_train, y_train)

        train_pred = clf.predict_proba(X_train)
        train_acc = log_loss(y_train, train_pred)
        train_accs.append(train_acc)
        val_pred = clf.predict_proba(X_val)
        val_acc = log_loss(y_val, val_pred)
        val_accs.append(val_acc)

        if need_print:
            print(f'Fold {fold}: train_acc = {train_acc:.5f}, val_acc = {val_acc:.5f}')

        if test_inputs is not None:
            test_pred = clf.predict_proba(test_inputs)
            test_preds += test_pred/cv.get_n_splits()
            
    m_train_acc = np.mean(train_accs)
    s_train_acc = np.std(train_accs)
    m_val_acc = np.mean(val_accs)
    s_val_acc = np.std(val_accs)
    
    msg = f'{m_val_acc:.7f} ± {s_val_acc:.7f}'
    
    if need_print:
        print(f'Train acc: {m_train_acc:.7f} ± {s_train_acc:.7f} | Val acc: {m_val_acc:.7f} ± {s_val_acc:.7f}')
    return test_preds, msg