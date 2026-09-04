# /// script
# dependencies = ["wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/train_val_test_split_with_tabular_data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{splitting-tabular-data} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{splitting-tabular-data} -->

    # Tabular Data Versioning and Deduplication with Weights & Biases

    <img src="http://wandb.me/mini-diagram" width="600" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction
    This walkthrough focuses on using W&B to version control and iterate on tabular data. We will use [Artifacts](https://docs.wandb.ai/guides/artifacts) and [Tables](https://docs.wandb.ai/guides/data-vis/tables-quickstart) to load in a dataset and split it into train, validation, and test subsets. Thanks to the versioning capability of Artifacts, we will use minimal storage space and have persistent version labels to easily share dataset iterations with colleagues.

    For this project, we will be working with tabular medical data that has great potential for predicting outcomes of heart attack patients. If you'd rather learn about similar features applied to classification tasks on image data, see this [other example](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA).

    You can also find an overview [report on this topic here](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Dataset-Version-Control-and-Deduplication-with-Tabular-Data-with-W-B-Artifacts-and-Tables--VmlldzoxNDIzOTA1).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### About the data
    Our dataset is a collection of measurements and survey answers from hospital patients that have heart attack-related symptoms. It's not a stretch to say datasets like these -- and the models built from them -- can improve patient outcomes and save lives. On a purely machine learning level, this particular dataset is interesting because it has:
    * **Mixed types**: entries can be binary, ordinal, numeric, or categorical
    * **Missing data**: almost all features have some fraction of missing data
    * **Real-world complexity**: feature values are not uniformly distributed and have outliers
    * **Limited size**: collecting data is difficult and so the dataset is small
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper functions
    Below we will keep all of our imports and helper functions. Reading through them is optional, and only recommended after you have looked over the rest of the report.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install --upgrade wandb -qqq
    import wandb

    import random
    from collections import OrderedDict
    import json
    import requests
    import csv
    import os

    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    DEVICE = 'cpu'
    PROJECT_NAME = 'splitting-tabular-data'

    # Set the random seeds to improve reproducibility by removing stochasticity
    def set_seeds(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False # Force cuDNN to use a consistent convolution algorithm
        torch.backends.cudnn.deterministic = True # Force cuDNN to use deterministic algorithms if available
        torch.use_deterministic_algorithms(True) # Force torch to use deterministic algorithms if available

    set_seeds(0)
    return (
        DEVICE,
        DataLoader,
        Dataset,
        OrderedDict,
        PROJECT_NAME,
        csv,
        json,
        np,
        os,
        requests,
        torch,
        transforms,
        wandb,
    )


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _(
    DataLoader,
    Dataset,
    OrderedDict,
    PROJECT_NAME,
    json,
    np,
    os,
    torch,
    transforms,
    wandb,
):
    def make_split_artifact(run, raw_data_table, train_rows, val_rows, test_rows):
        """
        Creates a w&b artifact that contains a singular reference table (aka a ForeignIndex table).
        The ForeignIndex table has a single column that we are naming 'source'.
        It contains references to the original table (raw_data_table) for each of the splits.
        Arguments:
            run (wandb run) returned from wandb.init()
            raw_data_table (wandb Table) that contains your original tabular data
            train_rows (list of ints) indices that reference the training rows in the raw_data_table
            val_rows (list of ints) indices that reference the validation rows in the raw_data_table
            test_rows (list of ints) indices that reference the test rows in the raw_data_table
        """
        split_artifact = wandb.Artifact('data-splits', type='dataset', description='Train, validation, test dataset splits')
        data_table_pointer = raw_data_table.get_index()
        split_artifact.add(wandb.Table(columns=['source'], data=[[data_table_pointer[i]] for i in train_rows]), 'train-data')
        split_artifact.add(wandb.Table(columns=['source'], data=[[data_table_pointer[i]] for i in val_rows]), 'val-data')
        split_artifact.add(wandb.Table(columns=['source'], data=[[data_table_pointer[i]] for i in test_rows]), 'test-data')  # Our data split artifact will only store index references to the original dataset table to save space
        _run.log_artifact(split_artifact)  # ForeignIndex automatically references the source table

    def make_loaders(config):
        """
        Makes data loaders using a artifact containing the dataset splits (created using the make_split_artifact() function)
        The function assumes that you have created a data-splits artifact and a data-transforms artifact
        Arguments:
            config [dict] containing keys:
                data_columns (list of ints) referencing which columns are to be treated as data
                label_columns (list of ints) referencing which columns are to be treated as labels
                num_classes (int) number of possible label classes in the dataset
                batch_size (int) amount of rows (i.e. data instances) to be delivered in a single batch
        Returns:
            train_loader (PyTorch DataLoader) containing the training data
            val_loader (PyTorch DataLoader) containing the validation data
            test_loader (PyTorch DataLoader) containing the test data
        """
        with wandb.init(project=PROJECT_NAME, job_type='package-data', config=config) as _run:
            transform_dir = _run.use_artifact('data-transforms:latest').download()
            transform_dict = json.load(open(os.path.join(transform_dir, 'transforms.txt')), object_pairs_hook=OrderedDict)
            composed_transforms = get_transforms(transform_dict)
            split_artifact = _run.use_artifact('data-splits:latest')
            train_loader = DataLoader(MyocardialInfarctionDataset(split_artifact.get('train-data'), config['data_columns'], config['label_columns'], config['num_classes'], composed_transforms), batch_size=config['batch_size'], drop_last=True, shuffle=True, num_workers=0)
            val_loader = DataLoader(MyocardialInfarctionDataset(split_artifact.get('val-data'), config['data_columns'], config['label_columns'], config['num_classes'], composed_transforms), batch_size=config['batch_size'], batch_sampler=None, shuffle=False, num_workers=0)
            test_loader = DataLoader(MyocardialInfarctionDataset(split_artifact.get('test-data'), config['data_columns'], config['label_columns'], config['num_classes'], composed_transforms), batch_size=config['batch_size'], batch_sampler=None, shuffle=False, num_workers=0)
        return (train_loader, val_loader, test_loader)

    def get_table_row(table, ndx):
        """
        Given a table and index, return the corresponding row  # Load the transforms
        Arguments:
            table (wandb.Table) can be a standard table of data or a pointer to a reference table
            ndx (int) row index to slice
        Returns:
            ref_row (list) of data entries for the row referenced by ndx  # Reformat data to (inputs, labels)
        """
        linked_table = np.all([type(value) is wandb.data_types._ForeignIndexType for value in table._column_types.params['type_map'].values()])
        if linked_table:
            ref_table = table.get_column(table.columns[0])
            if type(ndx) is list:
                ref_row = [list(ref_table[i].get_row().values()) for i in ndx]
            elif type(ndx) is int:
                ref_row = list(ref_table[ndx].get_row().values())
            else:
                raise ValueError(f'Input argument ndx must be of type int or list, not {type(ndx)}')
            return ref_row
        else:
            return table.data[ndx]

    class MyocardialInfarctionDataset(Dataset):
        """
        Myocardial Infarction Dataset
        In general columns 2-112 can be used as input data for prediction.
        Possible complications (outputs) are listed in columns 113-124.
   
        There  are  four  possible  time  moments  for  complication  prediction:  on  base  of  the information known at
        1. The time of admission to hospital: all input columns (2-112) except 93, 94, 95, 100, 101, 102, 103, 104, 105 can be used for prediction;
        2. The end of the first day (24 hours after admission to the hospital): all input columns (2-112) except 94, 95, 101, 102, 104, 105 can be used for prediction;
        3. The end of the second day (48 hours after admission to the hospital) all input columns (2-112) except 95, 102, 105 can be used for prediction;
        4. The end of the third day (72 hours after admission to the hospital) all input columns (2-112) can be used for prediction.

        All of the above column numbers are 1-indexed.
        """

        def __init__(self, table, data_columns, label_columns, num_classes, transform=None):
            """
            Args:
                table (wandb.Table): table containing the dataset
                data_columns (list): list of column indices corresponding to the data (X)
                label_columns (list): list of column indices corresponding to the labels (Y)
                num_classes (int): number of possible output classes (for one-hot encoding)
                transform (function): receives (data, label) tuple as input and produces transformed (data, label) tuple as output
            """
            super(MyocardialInfarctionDataset, self).__init__()
            self.table = table
            self.data_columns = data_columns  # Check if the table's contents are pointers to another table or not
            self.label_columns = label_columns
            self.num_classes = num_classes
            self.transform = transform

        def __len__(self):  # The table entries reference another table
            return len(self.table.data)  # There should only be one reference column
      # The pointers are dereferenced using the get_row() function
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            label_row = np.array(get_table_row(self.table, idx), dtype=np.float32).take(self.label_columns)
            data_row = np.array(get_table_row(self.table, idx), dtype=np.float32).take(self.data_columns)
            if self.transform:
                data_row, label_row = self.transform((data_row, label_row))
            return (data_row, label_row)

    class NoneToVal(object):  # Standard w&b Table containing the data
        """Convert None or NaN entries to usable values
        """

        def __init__(self, fill_value):
            self.fill_value = fill_value

        def __call__(self, data_tuple):
            data, label = data_tuple
            data = np.ma.masked_invalid(data).filled(fill_value=self.fill_value)
            return (data, label)

    class ToTensor(object):
        """Convert numpy arrays to tensor arrays
        """

        def __init__(self, device=None):
            if device is None:
                device = 'cpu'
            self.device = device

        def __call__(self, data_tuple):
            data, labels = data_tuple
            return (torch.from_numpy(data).to(self.device), torch.from_numpy(labels).to(self.device))

    class OneHot(object):
        """Convert input tensor to one-hot array
        """

        def __init__(self, num_classes):
            self.num_classes = int(num_classes)

        def __call__(self, data_tuple):
            data, labels = data_tuple
            device = labels.device
            dtype = labels.dtype
            num_datapoints = int(labels.ndim)
            labels_one_hot = torch.zeros((num_datapoints, self.num_classes), dtype=dtype).to(device)
            labels_one_hot[:, labels.long()] = 1
            return (data, labels_one_hot.squeeze())

    def get_transforms(transform_dict):
        """
        Given a dictionary of transform parameters, return a list of class instances for each transform
        Arguments:
            transform_dict (OrderedDict) with optional keys:
                NoneToVal (dict) if present, requires the 'value' key that None/nan will be replaced with
                ToTensor (dict) if present, requires the 'device' key that indicates the PyTorch device
                OneHot (dict) if present, requires the 'num_classes' key that has an int value for the number of possible data labels
        Returns:
            composed_transforms (PyTorch composed transform class) containing the requested transform steps in order
        """
        transform_functions = []
        for key in transform_dict.keys():
            if key == 'NoneToVal':
                transform_functions.append(NoneToVal(transform_dict[key]['value']))
            elif key == 'ToTensor':
                transform_functions.append(ToTensor(transform_dict[key]['device']))
            elif key == 'OneHot':
                transform_functions.append(OneHot(transform_dict[key]['num_classes']))
        composed_transforms = transforms.Compose(transform_functions)
        return composed_transforms  # Replace None/nan entries with a given value or array of values  # Convert array to a PyTorch Tensor  # Convert class labels to a one-hot representation

    return make_loaders, make_split_artifact


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following list contains the data types for each feature in the Myocardial Infarction dataset.

    More detailed information can be found via this [pdf file](https://s3-eu-west-1.amazonaws.com/pstorage-leicester-213265548798/22803695/Descriptivestatistics.pdf)
    """)
    return


@app.cell
def _():
    data_types = [
        int,   # 001. ID; numeric, [0, --]
        int,   # 002. AGE; numeric, [26, 92]
        int,   # 003. SEX; binary
        int,   # 004. INF_ANAM; ordinal, 4 levels
        int,   # 005. STENOK_AN; ordinal, 7 levels
        int,   # 006. FK_STENOK; ordinal, 5 levels
        int,   # 007. IBS_POST; ordinal, 3 levels
        int,   # 008. IBS_NASL; binary
        int,   # 009. GB; ordinal, 4 levels
        int,   # 010. SIM_GIPERT; binary
        int,   # 011. DLT_AG; ordinal, 8 levels
        int,   # 012. ZSN_A; partially ordered, 5 levels
        int,   # 013. nr11; binary
        int,   # 014. nr01; binary
        int,   # 015. nr02; binary
        int,   # 016. nr03; binary
        int,   # 017. nr04; binary
        int,   # 018. nr07; binary
        int,   # 019. nr08; binary
        int,   # 020. np01; binary
        int,   # 021. np04; binary
        int,   # 022. np05; binary
        int,   # 023. np07; binary
        int,   # 024. np08; binary
        int,   # 025. np09; binary
        int,   # 026. np10; binary
        int,   # 027. endocr_01; binary
        int,   # 028. endocr_02; binary
        int,   # 029. endocr_03; binary
        int,   # 030. zab_leg_01; binary
        int,   # 031. zab_leg_02; binary
        int,   # 032. zab_leg_03; binary
        int,   # 033. zab_leg_04; binary
        int,   # 034. zab_leg_06; binary
        float, # 035. S_AD_KBRIG; numeric, [0, 260] mmHg
        float, # 036. D_AD_KBRIG; numeric, [0, 190] mmHg
        float, # 037. S_AD_ORIT; numeric, [0, 260] mmHg
        float, # 038. D_AD_ORIT; numeric, [0, 190] mmHg
        int,   # 039. O_L_POST; binary
        int,   # 040. K_SH_POST; binary
        int,   # 041. MP_TP_POST; binary
        int,   # 042. SVT_POST; binary
        int,   # 043. GT_POST;  binary
        int,   # 044. FIB_G_POST; binary
        int,   # 045. ant_im; ordinal, 5 levels
        int,   # 046. lat_im; ordinal, 5 levels
        int,   # 047. inf_im; ordinal, 5 levels
        int,   # 048. post_im; ordinal, 5 levels
        int,   # 049. IM_PG_P; binary
        int,   # 050. ritm_ecg_p_01; binary
        int,   # 051. ritm_ecg_p_02; binary
        int,   # 052. ritm_ecg_p_04; binary
        int,   # 053. ritm_ecg_p_06; binary
        int,   # 054. ritm_ecg_p_07; binary
        int,   # 055. ritm_ecg_p_08; binary
        int,   # 056. n_r_ecg_p_01; binary
        int,   # 057. n_r_ecg_p_02; binary
        int,   # 058. n_r_ecg_p_03; binary
        int,   # 059. n_r_ecg_p_04; binary
        int,   # 060. n_r_ecg_p_05; binary
        int,   # 061. n_r_ecg_p_06; binary
        int,   # 062. n_r_ecg_p_08; binary
        int,   # 063. n_r_ecg_p_09; binary
        int,   # 064. n_r_ecg_p_10; binary
        int,   # 065. n_p_ecg_p_01; binary
        int,   # 066. n_p_ecg_p_03; binary
        int,   # 067. n_p_ecg_p_04; binary
        int,   # 068. n_p_ecg_p_05; binary
        int,   # 069. n_p_ecg_p_06; binary
        int,   # 070. n_p_ecg_p_07; binary
        int,   # 071. n_p_ecg_p_08; binary
        int,   # 072. n_p_ecg_p_09; binary
        int,   # 073. n_p_ecg_p_10; binary
        int,   # 074. n_p_ecg_p_11; binary
        int,   # 075. n_p_ecg_p_12; binary
        int,   # 076. fibr_ter_01; binary
        int,   # 077. fibr_ter_02; binary
        int,   # 078. fibr_ter_03; binary
        int,   # 079. fibr_ter_05; binary
        int,   # 080. fibr_ter_06; binary
        int,   # 081. fibr_ter_07; binary
        int,   # 082. fibr_ter_08; binary
        int,   # 083. GIPO_K; binary
        float, # 084. K_BLOOD; numeric, [2.3, 8.2] mmol/L
        int,   # 085. GIPER_Na; binary
        float, # 086. Na_BLOOD; numeric, [117, 169] mmol/L
        float, # 087. ALT_BLOOD; numeric, [0.03, 0.48] IU/L
        float, # 088. AST_BLOOD; numeric, [0.04, 2.15] IU/L
        float, # 089. KFK_BLOOD; numeric, [1.2, 3.6] IU/L
        float, # 090. L_BLOOD; numeric, [2, 27.9] billions per liter
        float, # 091. ROE; numeric, [1, 140] mm
        int,   # 092. TIME_B_S; ordinal, 10 levels
        int,   # 093. R_AB_1_n; ordinal, 4 levels
        int,   # 094. R_AB_2_n; ordinal, 4 levels
        int,   # 095. R_AB_3_n; ordinal, 4 levels
        int,   # 096. NA_KB; binary
        int,   # 097. NOT_NA_KB; binary 
        int,   # 098. LID_KB; binary
        int,   # 099. NITR_S; binary
        int,   # 100. NA_R_1_n; ordinal, 5 levels
        int,   # 101. NA_R_2_n; ordinal, 4 levels
        int,   # 102. NA_R_3_n; ordinal, 3 levels
        int,   # 103. NOT_NA_1_n; ordinal, 5 levels
        int,   # 104. NOT_NA_2_n; ordinal, 4 levels
        int,   # 105. NOT_NA_3_n; ordinal, 3 levels
        int,   # 106. LID_S_n; binary
        int,   # 107. B_BLOCK_S_n; binary
        int,   # 108. ANT_CA_S_n; binary
        int,   # 109. GEPAR_S_n; binary
        int,   # 110. ASP_S_n; binary
        int,   # 111. TIKL_S_n; binary
        int,   # 112. TRENT_S_n; binary
        int,   # 113. FIBR_PREDS; binary
        int,   # 114. PREDS_TAH; binary
        int,   # 115. JELUD_TAH; binary
        int,   # 116. FIBR_JELUD; binary
        int,   # 117. A_V_BLOCK; binary
        int,   # 118. OTEK_LANC; binary
        int,   # 119. RAZRIV; binary
        int,   # 120. DRESSLER, binary
        int,   # 121. ZSN; binary
        int,   # 122. REC_IM; binary
        int,   # 123. P_IM_STEN; binary
        int,   # 124. LET_IS; categorical, 8 categories
    ]
    return (data_types,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the dataset into an artifact
    Our first step is to load in the dataset from a CSV file, which we accomplish with Artifacts and Tables. The wandb Artifact has two very useful features for our application: 1) it supports versioning, which will allow us to track changes we make to the original datset and 2) it supports deduplication, which will minimize the amount of storage space we use when generating modified versions of the dataset.

    In the code below we use the python `requests` and `csv` libararies to load in each line of the CSV file into a wandb Table. Then we store the table in a wandb Artifact.
    """)
    return


@app.cell
def _(PROJECT_NAME, csv, data_types, np, requests, wandb):
    # Load raw dataset into a table and store it as an artifact
    with wandb.init(project=PROJECT_NAME, job_type='load-data') as _run:
        dataset_url = 'https://s3-eu-west-1.amazonaws.com/pstorage-leicester-213265548798/23581310/MyocardialinfarctioncomplicationsDatabase.csv'  # Load data row-by-row & add the rows to the table (Note: the whole table will be stored in memory)
        with requests.get(dataset_url, stream=True) as r:
            lines = (line.decode('utf-8') for line in r.iter_lines())
            column_headings = next(csv.reader(lines))  # Load each line into the table
            data_table = wandb.Table(columns=column_headings)
            for index, row in enumerate(csv.reader(lines)):  # This assumes that the first CSV line contains the column headings
                row = [data_types[entry_index](entry) if entry != '' else np.nan for entry_index, entry in enumerate(row)]  # Initialize the table
                if len(row) == len(column_headings):  # Starting at the second row
                    data_table.add_data(*row)
        dataset_artifact = wandb.Artifact('data-library', type='dataset', description='Table containing the CSV dataset', metadata={'MD5_checksum': 'd409a89bd7e566da4b82232c3956f576', 'filename': 'MyocardialinfarctioncomplicationsDatabase.csv', 'filesize': '427.31 kB', 'dataset_host': 'University of Leicester', 'dataset_url': dataset_url, 'project_url': 'https://doi.org/10.25392/leicester.data.12045261.v3', 'reference_doi': '10.25392/leicester.data.12045261.v3'})
        dataset_artifact.add(data_table, 'data-table')
        _run.log_artifact(dataset_artifact)  # Create an artifact for our dataset  # Add the table to the artifact & log the artifact
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Splitting the data into training, validation, and test sets
    As is typical with most machine learning applications, we want to grab a majority of our data for training and use a smaller subset for validation and testing. The validation set will be used to help us tune our parameters and modify the preprocessing steps.

    To avoid saving multiple copies of the dataset, we will only store corresponding indices for the train/val/test splits. This is also version controlled in case you decide you need more training data or you want to redo the shuffles.
    """)
    return


@app.cell
def _(PROJECT_NAME, make_split_artifact, np, wandb):
    config = {'train_val_test_split': [0.8, 0.1, 0.1], 'data_columns': [i for i in range(1, 112)], 'label_columns': [123], 'num_classes': 8, 'batch_size': 20}
    with wandb.init(project=PROJECT_NAME, job_type='split-data', config=config) as _run:  # These must sum to 1.0
        raw_data_table = _run.use_artifact('data-library:latest').get('data-table')  # All possible training features
        num_samples = len(raw_data_table.data)  # Lethal outcome
        shuffled_rows = np.random.choice(np.arange(num_samples), num_samples, replace=False)  # Num classes after preprocessing
        train_rows, val_rows, test_rows = np.split(shuffled_rows, np.cumsum([num_samples * split for split in config['train_val_test_split'][:-1]], dtype=int))  # Num samples to average over for gradient updates
        make_split_artifact(_run, raw_data_table, train_rows, val_rows, test_rows)
    test_slicing = True
    # Split data into train, val, test tables
    if test_slicing:
        print('num total: ', num_samples)  # Define the data splits
        print('num train:', len(train_rows))
        print('num val:', len(val_rows))  # One of many methods to extract random train/val/test rows
        print('num test:', len(test_rows))
        print('shuffle duplicates: ', len(set(shuffled_rows)) != len(shuffled_rows))
        print('val in train:', np.any([row in train_rows for row in val_rows]))
        print('test in train:', np.any([row in train_rows for row in test_rows]))
        print('val in test:', np.any([row in val_rows for row in test_rows]))  # Construct a new artifact for the data splits
        print('train dupliates: ', len(set(train_rows)) != len(train_rows))
        print('val dupliates: ', len(set(val_rows)) != len(val_rows))
    # Quick test to make sure the slicing worked properly
        print('test dupliates: ', len(set(test_rows)) != len(test_rows))
        print('all samples accounted for in the shuffled set:', len(shuffled_rows) == num_samples)
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Packaging the data into PyTorch data loaders
    Next we will package the data into a PyTorch DataLoader to make it easier to work with. The DataLoader includes a list of preprocessing steps that are to be performed on the data. We want to be able to iterate and version control preprocessing pipeline, so we also have to write some code to store it as an artifact.
    """)
    return


@app.cell
def _(DEVICE, OrderedDict, PROJECT_NAME, config, json, make_loaders, wandb):
    with wandb.init(project=PROJECT_NAME, job_type='define-transforms', config=config) as _run:
        transform_dict = OrderedDict()  # Define an initial set of transforms that we think will be useful
        transform_dict['NoneToVal'] = {'value': 0}
        transform_dict['ToTensor'] = {'device': DEVICE}
        transform_dict['OneHot'] = {'num_classes': config['num_classes']}  # for the first pass we will replace missing values with 0
        for key_idx, key in enumerate(transform_dict.keys()):
            transform_dict[key]['order'] = key_idx
        data_transform_artifact = wandb.Artifact('data-transforms', type='parameters', description='Data preprocessing functions and parameters.', metadata=transform_dict)
        with data_transform_artifact.new_file('transforms.txt') as f:
            f.write(json.dumps(transform_dict, indent=4))
        _run.log_artifact(data_transform_artifact)
    config.update(transform_dict)
    # Now we can make the data loaders with the preprocessing pipeline
    train_loader, val_loader, test_loader = make_loaders(config)  # Include an operational index to verify the order  # Create an artifact for logging the transforms  # Optional for viewing on the web app; the data is also stored in the txt file below  # Log the transforms in JSON format  # Log the transforms in the config so that we can sweep over them in future iterations
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Done! Time to train

    That concludes this part of the tutorial. In a future tutorial we will use this same data to train and iterate on our model. This will also use Artifacts to version control iterations on the preprocessing pipeline, parameters, and model architecture.
    """)
    return


if __name__ == "__main__":
    app.run()
