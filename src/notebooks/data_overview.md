# GTSRB Dataset Overview

Notebook này trình bày tổng quan về bộ dữ liệu **German Traffic Sign Recognition Benchmark – GTSRB**.

Nội dung gồm:

- Đọc thông tin tập train và test.
- Thống kê số lượng ảnh và số lớp.
- Phân tích phân bố lớp.
- Phân tích kích thước ảnh.
- Hiển thị một số ảnh mẫu.


```python
!mkdir -p /content/data

!wget -nc -P /content/data \
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip

!wget -nc -P /content/data \
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip

!wget -nc -P /content/data \
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip

!unzip -q -n /content/data/GTSRB_Final_Training_Images.zip -d /content/data
!unzip -q -n /content/data/GTSRB_Final_Test_Images.zip -d /content/data
!unzip -q -n /content/data/GTSRB_Final_Test_GT.zip -d /content/data

print("Dataset downloaded and extracted successfully.")
```

    --2026-07-13 02:45:02--  https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
    Resolving sid.erda.dk (sid.erda.dk)... 130.225.104.13
    Connecting to sid.erda.dk (sid.erda.dk)|130.225.104.13|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 276294756 (263M) [application/zip]
    Saving to: ‘/content/data/GTSRB_Final_Training_Images.zip’
    
    GTSRB_Final_Trainin 100%[===================>] 263.50M  14.6MB/s    in 25s     
    
    2026-07-13 02:45:28 (10.6 MB/s) - ‘/content/data/GTSRB_Final_Training_Images.zip’ saved [276294756/276294756]
    
    --2026-07-13 02:45:28--  https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
    Resolving sid.erda.dk (sid.erda.dk)... 130.225.104.13
    Connecting to sid.erda.dk (sid.erda.dk)|130.225.104.13|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 88978620 (85M) [application/zip]
    Saving to: ‘/content/data/GTSRB_Final_Test_Images.zip’
    
    GTSRB_Final_Test_Im 100%[===================>]  84.86M  13.9MB/s    in 7.2s    
    
    2026-07-13 02:45:36 (11.8 MB/s) - ‘/content/data/GTSRB_Final_Test_Images.zip’ saved [88978620/88978620]
    
    --2026-07-13 02:45:36--  https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip
    Resolving sid.erda.dk (sid.erda.dk)... 130.225.104.13
    Connecting to sid.erda.dk (sid.erda.dk)|130.225.104.13|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 99620 (97K) [application/zip]
    Saving to: ‘/content/data/GTSRB_Final_Test_GT.zip’
    
    GTSRB_Final_Test_GT 100%[===================>]  97.29K   278KB/s    in 0.4s    
    
    2026-07-13 02:45:37 (278 KB/s) - ‘/content/data/GTSRB_Final_Test_GT.zip’ saved [99620/99620]
    
    Dataset downloaded and extracted successfully.


## 1. Load Dataset Annotations

GTSRB có 43 lớp biển báo giao thông.

Training set được chia thành 43 thư mục. Mỗi thư mục chứa ảnh và một file CSV annotation.

Test set sử dụng một file annotation chung có tên `GT-final_test.csv`.


```python
import os
import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt

TRAIN_DIR = "/content/data/GTSRB/Final_Training/Images"
TEST_DIR = "/content/data/GTSRB/Final_Test/Images"
```


```python
# Đọc annotation của training set
train_csv_files = sorted(
    glob.glob(os.path.join(TRAIN_DIR, "*", "GT-*.csv"))
)

train_frames = []

for csv_path in train_csv_files:
    class_folder = os.path.dirname(csv_path)

    class_df = pd.read_csv(csv_path, sep=";")

    class_df["ImagePath"] = class_df["Filename"].apply(
        lambda filename: os.path.join(class_folder, filename)
    )

    train_frames.append(class_df)

train_df = pd.concat(train_frames, ignore_index=True)

print("Training shape:", train_df.shape)
train_df.head()
```

    Training shape: (39209, 9)






  <div id="df-c3e584e7-217e-470a-9993-59b229c7e0db" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filename</th>
      <th>Width</th>
      <th>Height</th>
      <th>Roi.X1</th>
      <th>Roi.Y1</th>
      <th>Roi.X2</th>
      <th>Roi.Y2</th>
      <th>ClassId</th>
      <th>ImagePath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00000_00000.ppm</td>
      <td>29</td>
      <td>30</td>
      <td>5</td>
      <td>6</td>
      <td>24</td>
      <td>25</td>
      <td>0</td>
      <td>/content/data/GTSRB/Final_Training/Images/0000...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00000_00001.ppm</td>
      <td>30</td>
      <td>30</td>
      <td>5</td>
      <td>5</td>
      <td>25</td>
      <td>25</td>
      <td>0</td>
      <td>/content/data/GTSRB/Final_Training/Images/0000...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000_00002.ppm</td>
      <td>30</td>
      <td>30</td>
      <td>5</td>
      <td>5</td>
      <td>25</td>
      <td>25</td>
      <td>0</td>
      <td>/content/data/GTSRB/Final_Training/Images/0000...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00000_00003.ppm</td>
      <td>31</td>
      <td>31</td>
      <td>5</td>
      <td>5</td>
      <td>26</td>
      <td>26</td>
      <td>0</td>
      <td>/content/data/GTSRB/Final_Training/Images/0000...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00000_00004.ppm</td>
      <td>30</td>
      <td>32</td>
      <td>5</td>
      <td>6</td>
      <td>25</td>
      <td>26</td>
      <td>0</td>
      <td>/content/data/GTSRB/Final_Training/Images/0000...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c3e584e7-217e-470a-9993-59b229c7e0db')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c3e584e7-217e-470a-9993-59b229c7e0db button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c3e584e7-217e-470a-9993-59b229c7e0db');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
# Đọc annotation của test set
test_gt_path = glob.glob(
    "/content/data/**/GT-final_test.csv",
    recursive=True
)[0]

test_df = pd.read_csv(test_gt_path, sep=";")

test_df["ImagePath"] = test_df["Filename"].apply(
    lambda filename: os.path.join(TEST_DIR, filename)
)

print("Test shape:", test_df.shape)
test_df.head()
```

    Test shape: (12630, 9)






  <div id="df-1b71aa7f-f910-4c13-a31c-e17d8e931c34" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filename</th>
      <th>Width</th>
      <th>Height</th>
      <th>Roi.X1</th>
      <th>Roi.Y1</th>
      <th>Roi.X2</th>
      <th>Roi.Y2</th>
      <th>ClassId</th>
      <th>ImagePath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00000.ppm</td>
      <td>53</td>
      <td>54</td>
      <td>6</td>
      <td>5</td>
      <td>48</td>
      <td>49</td>
      <td>16</td>
      <td>/content/data/GTSRB/Final_Test/Images/00000.ppm</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00001.ppm</td>
      <td>42</td>
      <td>45</td>
      <td>5</td>
      <td>5</td>
      <td>36</td>
      <td>40</td>
      <td>1</td>
      <td>/content/data/GTSRB/Final_Test/Images/00001.ppm</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00002.ppm</td>
      <td>48</td>
      <td>52</td>
      <td>6</td>
      <td>6</td>
      <td>43</td>
      <td>47</td>
      <td>38</td>
      <td>/content/data/GTSRB/Final_Test/Images/00002.ppm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00003.ppm</td>
      <td>27</td>
      <td>29</td>
      <td>5</td>
      <td>5</td>
      <td>22</td>
      <td>24</td>
      <td>33</td>
      <td>/content/data/GTSRB/Final_Test/Images/00003.ppm</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00004.ppm</td>
      <td>60</td>
      <td>57</td>
      <td>5</td>
      <td>5</td>
      <td>55</td>
      <td>52</td>
      <td>11</td>
      <td>/content/data/GTSRB/Final_Test/Images/00004.ppm</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1b71aa7f-f910-4c13-a31c-e17d8e931c34')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1b71aa7f-f910-4c13-a31c-e17d8e931c34 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1b71aa7f-f910-4c13-a31c-e17d8e931c34');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




## 2. Dataset Summary

Phần này thống kê số lượng ảnh và số lớp của training set, test set và toàn bộ dataset.


```python
dataset_summary = pd.DataFrame({
    "Dataset": [
        "Training set",
        "Test set",
        "Complete dataset"
    ],
    "Number of images": [
        len(train_df),
        len(test_df),
        len(train_df) + len(test_df)
    ],
    "Number of classes": [
        train_df["ClassId"].nunique(),
        test_df["ClassId"].nunique(),
        train_df["ClassId"].nunique()
    ]
})

dataset_summary
```





  <div id="df-c387d845-cae4-4a03-b139-edb3c0934885" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataset</th>
      <th>Number of images</th>
      <th>Number of classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Training set</td>
      <td>39209</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Test set</td>
      <td>12630</td>
      <td>43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Complete dataset</td>
      <td>51839</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c387d845-cae4-4a03-b139-edb3c0934885')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c387d845-cae4-4a03-b139-edb3c0934885 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c387d845-cae4-4a03-b139-edb3c0934885');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_bb5e9bec-5c65-49dc-a4cf-fadb11001b62">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('dataset_summary')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_bb5e9bec-5c65-49dc-a4cf-fadb11001b62 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('dataset_summary');
      }
      })();
    </script>
  </div>

    </div>
  </div>




## 3. Class Distribution

GTSRB có 43 lớp nhưng số lượng ảnh giữa các lớp không bằng nhau.

Biểu đồ phân bố lớp giúp xác định mức độ mất cân bằng dữ liệu. Những lớp có nhiều ảnh hơn có thể ảnh hưởng mạnh hơn đến quá trình huấn luyện mô hình.


```python
class_distribution = (
    train_df["ClassId"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(16, 6))

plt.bar(
    class_distribution.index,
    class_distribution.values
)

plt.xlabel("Class ID")
plt.ylabel("Number of training images")
plt.title("Distribution of Training Images Across 43 Classes")

plt.xticks(range(43))
plt.grid(axis="y", alpha=0.3)

plt.show()
```


    
![png](data_overview_files/data_overview_9_0.png)
    



```python
print("Class with the fewest images:")
print("Class ID:", class_distribution.idxmin())
print("Number of images:", class_distribution.min())

print("\nClass with the most images:")
print("Class ID:", class_distribution.idxmax())
print("Number of images:", class_distribution.max())

print(
    "\nImbalance ratio:",
    round(
        class_distribution.max()
        / class_distribution.min(),
        2
    )
)
```

    Class with the fewest images:
    Class ID: 0
    Number of images: 210
    
    Class with the most images:
    Class ID: 2
    Number of images: 2250
    
    Imbalance ratio: 10.71


## 4. Image Dimension Analysis

Các ảnh trong GTSRB có chiều rộng và chiều cao khác nhau.

Điều này xảy ra vì biển báo được chụp ở nhiều khoảng cách, góc nhìn và điều kiện ánh sáng khác nhau.

Do kích thước ảnh không cố định, ảnh cần được resize về cùng kích thước trong bước tiền xử lý.


```python
train_df[["Width", "Height"]].describe()
```





  <div id="df-6ba93784-495e-4eca-8190-d24bf28d1a94" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Width</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>39209.000000</td>
      <td>39209.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>50.835880</td>
      <td>50.328930</td>
    </tr>
    <tr>
      <th>std</th>
      <td>24.306933</td>
      <td>23.115423</td>
    </tr>
    <tr>
      <th>min</th>
      <td>25.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35.000000</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>43.000000</td>
      <td>43.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>58.000000</td>
      <td>58.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>243.000000</td>
      <td>225.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6ba93784-495e-4eca-8190-d24bf28d1a94')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6ba93784-495e-4eca-8190-d24bf28d1a94 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6ba93784-495e-4eca-8190-d24bf28d1a94');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
plt.figure(figsize=(10, 5))

plt.hist(
    train_df["Width"],
    bins=40,
    alpha=0.65,
    label="Width"
)

plt.hist(
    train_df["Height"],
    bins=40,
    alpha=0.65,
    label="Height"
)

plt.xlabel("Number of pixels")
plt.ylabel("Number of images")
plt.title("Distribution of Original Image Dimensions")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.show()
```


    
![png](data_overview_files/data_overview_13_0.png)
    


## 5. Sample Images

Phần này hiển thị một số ảnh ngẫu nhiên trong training set.

Các ảnh cho thấy sự khác nhau về loại biển báo, kích thước, độ sáng, góc nhìn và chất lượng hình ảnh.


```python
sample_df = train_df.sample(
    n=12,
    random_state=42
)

plt.figure(figsize=(16, 10))

for position, (_, row) in enumerate(
    sample_df.iterrows(),
    start=1
):
    image = cv2.imread(row["ImagePath"])

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Crop vùng chứa biển báo để hiển thị rõ hơn
    x1 = int(row["Roi.X1"])
    y1 = int(row["Roi.Y1"])
    x2 = int(row["Roi.X2"])
    y2 = int(row["Roi.Y2"])

    roi_image = image[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    plt.subplot(3, 4, position)

    plt.imshow(roi_image)

    plt.title(
        f"Class ID: {row['ClassId']}"
    )

    plt.axis("off")

plt.suptitle(
    "Sample Images from the GTSRB Dataset",
    fontsize=16
)

plt.tight_layout()
plt.show()
```


    
![png](data_overview_files/data_overview_15_0.png)
    


## 6. Dataset Overview Conclusion

The GTSRB dataset contains 51,839 traffic sign images belonging to 43 classes.

The training set contains 39,209 images, while the official test set contains 12,630 images.

The exploratory analysis shows that:

- The dataset contains 43 traffic sign categories.
- The classes do not contain equal numbers of images.
- The original images have different widths and heights.
- Traffic signs were captured under different viewing and lighting conditions.
- The dataset therefore requires image preprocessing before feature extraction and model training.

In the next stage, the Region of Interest will be cropped, resized to a fixed size, converted to grayscale, and used for HOG feature extraction.
