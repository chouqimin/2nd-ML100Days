# **Edison**的第二屆機器學習百日馬拉松之旅(2019.4~2019.8)
### 目錄
* ## [**1. 資料清理數據前處理**](#資料清理數據前處理)
* ## [**2. 資料科學特徵工程技術**](#資料科學特徵工程技術)
* ## [**3. 機器學習基礎模型建立**](#機器學習基礎模型建立)
* ## [**4. 機器學習調整參數**](#機器學習調整參數)
* ## [**Kaggle 第一次期中考**](#Kaggle第一次期中考)
* ## [**5. 非監督式機器學習**](#非監督式機器學習)
* ## [**6. 深度學習理論與實作**](#深度學習理論與實作)
* ## [**7. 初探深度學習使用Keras**](#初探深度學習使用Keras)
* ## [**8. 深度學習應用卷積神經網路**](#深度學習應用卷積神經網路)
* ## [**Kaggle 期末考**](#Kaggle期末考)
* ## [**9. Bonus 進階補充**](#Bonus進階補充)
---
# **資料清理數據前處理**
##### *以滾動方式進行資料清理與探索性分析*
## [D1 : 資料介紹與評估資料](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859638/1582798284512/__PDF__?t=1582798284498 "D1")
###### *挑戰是什麼?動手分析前請三思*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_001_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859638/Day_001_HW.ipynb?t=1582798284498)
>* [Day_001_example_of_metrics.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859638/Day_001_example_of_metrics.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_001_HW.ipynb)

>範例解答:
>* [Day_001_example_of_metrics_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859638/Day_001_example_of_metrics_Ans.ipynb?t=1582798284498)
---
## [D2 : EDA-1/讀取資料EDA: Data summary](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859639/1582798284513/__PDF__?t=1582798284498 "D2")
###### *如何讀取資料以及萃取出想要了解的信息*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_002_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859639/Day_002_HW.ipynb?t=1582798284498)
>* [Day_002_first_EDA.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859639/Day_002_first_EDA.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_002_HW.ipynb)

>範例解答:
>* [Day_002_first_EDA_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859639/Day_002_first_EDA_Ans.ipynb?t=1582798284498)
---
## [D3 : 3-1如何新建一個 dataframe?3-2 如何讀取其他資料? (非 csv 的資料)](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859640/1582798284514/__PDF__?t=1582798284498 "D3")
###### *1. 從頭建立一個 dataframe 2. 如何讀取不同形式的資料 (如圖檔、純文字檔、json 等)*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_003-1_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859640/Day_003-1_HW.ipynb?t=1582798284498)
>* [Day_003-1_build_dataframe_from_scratch.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859640/Day_003-1_build_dataframe_from_scratch.ipynb?t=1582798284498)
>* [Day_003-2_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859640/Day_003-2_HW.ipynb?t=1582798284498)
>* [Day_003-2_read_and_write_files.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859640/Day_003-2_read_and_write_files.ipynb?t=1582798284498)
>* [Day_003-3_read_and_write_files.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859640/Day_003-3_read_and_write_files.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_003-2_HW.ipynb)

>範例解答:
>* [Day_003-1_build_dataframe_from_scratch_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859640/Day_003-1_build_dataframe_from_scratch_Ans.ipynb?t=1582798284498)
>* [Day_003-2_read_and_write_files_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859640/Day_003-2_read_and_write_files_Ans.ipynb?t=1582798284498)
---
## [D4 : EDA: 欄位的資料類型介紹及處理](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859641/1582798284515/__PDF__?t=1582798284498 "D4")
###### *了解資料在 pandas 中可以表示的類型*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_004_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859641/Day_004_HW.ipynb?t=1582798284498)
>* [Day_004_column_data_type.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859641/Day_004_column_data_type.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_004_HW.ipynb)

>範例解答:
>* [Day_004_column_data_type_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859641/Day_004_column_data_type_Ans.ipynb?t=1582798284498)
---
## [D5 : EDA資料分佈](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859642/1582798284516/__PDF__?t=1582798284498 "D5")
###### *用統計方式描述資料*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_005_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859642/Day_005_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_005_HW.ipynb)

>範例解答:
>* [Day_005_distribution_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859642/Day_005_distribution_Ans.ipynb?t=1582798284498)
---
## [D6 : EDA: Outlier 及處理](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859643/1582798284517/__PDF__?t=1582798284498 "D6")
###### *偵測與處理例外數值點：1. 透過常用的偵測方法找到例外 2. 判斷例外是否正常 (推測可能的發生原因)*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_006_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859643/Day_006_HW.ipynb?t=1582798284498)
>* [Day_006_outliers_detection.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859643/Day_006_outliers_detection.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_006_HW.ipynb)

>範例解答:
>* [Day_006_outliers_detection_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859643/Day_006_outliers_detection_Ans.ipynb?t=1582798284498)
---
## [D7 : 常用的數值取代：中位數與分位數連續數值標準化](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859644/1582798284518/__PDF__?t=1582798284498 "D7")
###### *偵測與處理例外數值 1. 缺值或例外取代 2. 數據標準化*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_007_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859644/Day_007_HW.ipynb?t=1582798284498)
>* [Day_007_handle_outliers.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859644/Day_007_handle_outliers.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_007_HW.ipynb)

>範例解答:
>* [Day_007_handle_outliers_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859644/Day_007_handle_outliers_Ans.ipynb?t=1582798284498)
---
## [D8 : DataFrame operationData frame merge/常用的 DataFrame 操作](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859645/1582798284519/__PDF__?t=1582798284498 "D8")
###### *1. 常見的資料操作方法 2. 資料表串接*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_008_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859645/Day_008_HW.ipynb?t=1582798284498)
>* [Day_008_dataFrame_operation.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859645/Day_008_dataFrame_operation.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_008_HW.ipynb)

>範例解答:
>* [Day_008_dataFrame_operation_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859645/Day_008_dataFrame_operation_Ans.ipynb?t=1582798284498)
---
## [D9 : 程式實作 EDA: correlation/相關係數簡介](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859646/1582798284520/__PDF__?t=1582798284498 "D9")
###### *1. 了解相關係數 2. 利用相關係數直觀地理解對欄位與預測目標之間的關係*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_009_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859646/Day_009_HW.ipynb?t=1582798284498)
>* [Day_009_correlation_example.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859646/Day_009_correlation_example.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_009_HW.ipynb)

>範例解答:
>* [Day_009_correlation_example_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859646/Day_009_correlation_example_Ans.ipynb?t=1582798284498)
---
## [D10 : EDA from Correlation](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859647/1582798284521/__PDF__?t=1582798284498 "D10")
###### *深入了解資料，從 correlation 的結果下手*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_010-supplementary_correlation_and_plot_with_different_range.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859647/Day_010-supplementary_correlation_and_plot_with_different_range.ipynb?t=1582798284498)
>* [Day_010_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859647/Day_010_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_010_HW.ipynb)

>範例解答:
>* [Day_010_correlation_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859647/Day_010_correlation_Ans.ipynb?t=1582798284498)
---
## [D11 : EDA: 不同數值範圍間的特徵如何檢視/繪圖與樣式Kernel Density Estimation (KDE)](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859648/1582798284522/__PDF__?t=1582798284498 "D11")
###### *1. 如何調整視覺化方式檢視數值範圍 2. 美圖修修 - 轉換繪圖樣式*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_011_EDA_KDEplots.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859648/Day_011_EDA_KDEplots.ipynb?t=1582798284498)
>* [Day_011_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859648/Day_011_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_011_HW.ipynb)

>範例解答:
>* [Day_011_EDA_KDEplots_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859648/Day_011_EDA_KDEplots_Ans.ipynb?t=1582798284498)
---
## [D12 : EDA: 把連續型變數離散化](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859649/1582798284523/__PDF__?t=1582798284498 "D12")
###### *簡化連續性變數*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_012_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859649/Day_012_HW.ipynb?t=1582798284498)
>* [Day_012_discretizing.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859649/Day_012_discretizing.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_012_HW.ipynb)

>範例解答:
>* [Day_012_discretizing_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859649/Day_012_discretizing_Ans.ipynb?t=1582798284498)
---
## [D13 : 程式實作 把連續型變數離散化](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859650/1582798284524/__PDF__?t=1582798284498 "D13")
###### *深入了解資料，從簡化後的離散變數下手*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_013_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859650/Day_013_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_013_HW.ipynb)

>範例解答:
>* [Day_013_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859650/Day_013_Ans.ipynb?t=1582798284498)
---
## [D14 : Subplots](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859651/1582798284525/__PDF__?t=1582798284498 "D14")
###### *探索性資料分析 - 資料視覺化 - 多圖檢視 1. 將數據分組一次呈現 2. 把同一組資料相關的數據一次攤在面前*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_014_EDA_subplots.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859651/Day_014_EDA_subplots.ipynb?t=1582798284498)
>* [Day_014_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859651/Day_014_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_014_HW.ipynb)

>範例解答:
>* [Day_014_EDA_subplots_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859651/Day_014_EDA_subplots_Ans.ipynb?t=1582798284498)
---
## [D15 : Heatmap & Grid-plot](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859652/1582798284526/__PDF__?t=1582798284498 "D15")
###### *探索性資料分析 - 資料視覺化 - 熱像圖 / 格狀圖 1. 熱圖：以直觀的方式檢視變數間的相關性 2. 格圖：繪製變數間的散佈圖及分布*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_015_EDA_heatmap.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859652/Day_015_EDA_heatmap.ipynb?t=1582798284498)
>* [Day_015_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859652/Day_015_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_015_HW.ipynb)

>範例解答:
>* [Day_015_EDA_heatmap_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859652/Day_015_EDA_heatmap_Ans.ipynb?t=1582798284498)
---
## [D16 : 模型初體驗 Logistic Regression](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859653/1582798284527/__PDF__?t=1582798284498 "D16")
###### *在我們開始使用任何複雜的模型之前，有一個最簡單的模型當作 baseline 是一個好習慣*
>Data:
>* [HomeCredit_columns_description.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/HomeCredit_columns_description.csv?t=1582796969499)
>* [POS_CASH_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/POS_CASH_balance.csv?t=1582796969499)
>* [application_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_test.csv?t=1582796969499)
>* [application_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/application_train.csv?t=1582796969499)
>* [bureau.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau.csv?t=1582796969499)
>* [bureau_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/bureau_balance.csv?t=1582796969499)
>* [credit_card_balance.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/credit_card_balance.csv?t=1582796969499)
>* [example.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.jpg?t=1582796969499)
>* [example.mat](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.mat?t=1582796969499)
>* [example.npy](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.npy?t=1582796969499)
>* [example.pkl](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.pkl?t=1582796969499)
>* [example.txt](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example.txt?t=1582796969499)
>* [example01.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example01.json?t=1582796969499)
>* [example02.json](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/example02.json?t=1582796969499)
>* [installments_payments.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/installments_payments.csv?t=1582796969499)
>* [previous_application.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/previous_application.csv?t=1582796969499)
>* [sample_submission.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795763215/sample_submission.csv?t=1582796969499)

>作業/範例:
>* [Day_016_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859653/Day_016_HW.ipynb?t=1582798284498)
>* [Day_016_first_model.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859653/Day_016_first_model.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_016_Home_Credit_Default_Risk.jpg)

>範例解答:
>* [Day_016_first_model_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859653/Day_016_first_model_Ans.ipynb?t=1582798284498)
---
---
# **資料科學特徵工程技術**
##### *使用統計或領域知識，以各種組合調整方式，生成新特徵以提升模型預測力。*
## [D17 : 特徵工程簡介](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859654/1582798284528/__PDF__?t=1582798284498 "D17")
###### *介紹機器學習完整步驟中，特徵工程的位置以及流程架構*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_017_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859654/Day_017_HW.ipynb?t=1582798284498)
>* [Day_017_Introduction_of_Feature Engineering.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859654/Day_017_Introduction_of_Feature%20Engineering.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_017_HW.ipynb)

>範例解答:
>* [Day_017_Introduction_of_Feature Engineering_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859654/Day_017_Introduction_of_Feature%20Engineering_Ans.ipynb?t=1582798284498)
---
## [D18 : 特徵類型](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859655/1582798284529/__PDF__?t=1582798284498 "D18")
###### *特徵工程依照特徵類型，做法不同，大致可分為數值/類別/時間型三類特徵*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_018_Feature_Types.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859655/Day_018_Feature_Types.ipynb?t=1582798284498)
>* [Day_018_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859655/Day_018_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_018_HW.ipynb)

>範例解答:
>* [Day_018_Feature_Types_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859655/Day_018_Feature_Types_Ans.ipynb?t=1582798284498)
---
## [D19 : 數值型特徵-補缺失值與標準化](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859656/1582798284530/__PDF__?t=1582798284498 "D19")
###### *數值型特徵首先必須填補缺值與標準化，在此複習並展示對預測結果的差異*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_019_Fill_NaN_and_Scalers.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859656/Day_019_Fill_NaN_and_Scalers.ipynb?t=1582798284498)
>* [Day_019_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859656/Day_019_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_019_HW.ipynb)

>範例解答:
>* [Day_019_Fill_NaN_and_Scalers_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859656/Day_019_Fill_NaN_and_Scalers_Ans.ipynb?t=1582798284498)
---
## [D20 : 數值型特徵 -  去除離群值](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859657/1582798284531/__PDF__?t=1582798284498 "D20")
###### *數值型特徵若出現少量的離群值，則需要去除以保持其餘數據不被影響*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_020_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859657/Day_020_HW.ipynb?t=1582798284498)
>* [Day_020_Outliers.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859657/Day_020_Outliers.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_020_HW.ipynb)

>範例解答:
>* [Day_020_Outliers_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859657/Day_020_Outliers_Ans.ipynb?t=1582798284498)
---
## [D21 : 數值型特徵 -  去除偏態](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859658/1582798284532/__PDF__?t=1582798284498 "D21")
###### *數值型特徵若分布明顯偏一邊，則需去除偏態以消除預測的偏差*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_021_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859658/Day_021_HW.ipynb?t=1582798284498)
>* [Day_021_Reduce_Skewness.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859658/Day_021_Reduce_Skewness.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_021_HW.ipynb)

>範例解答:
>* [Day_021_Reduce_Skewness_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859658/Day_021_Reduce_Skewness_Ans.ipynb?t=1582798284498)
---
## [D22 : 類別型特徵 - 基礎處理](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859659/1582798284533/__PDF__?t=1582798284498 "D22")
###### *介紹類別型特徵最基礎的作法 : 標籤編碼與獨熱編碼*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_022_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859659/Day_022_HW.ipynb?t=1582798284498)
>* [Day_022_LabelEncoder_and_OneHotEncoder.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859659/Day_022_LabelEncoder_and_OneHotEncoder.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_022_HW.ipynb)

>範例解答:
>* [Day_022_LabelEncoder_and_OneHotEncoder_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859659/Day_022_LabelEncoder_and_OneHotEncoder_Ans.ipynb?t=1582798284498)
---
## [D23 : 類別型特徵 - 均值編碼](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859660/1582798284534/__PDF__?t=1582798284498 "D23")
###### *類別型特徵最重要的編碼 : 均值編碼，將標籤以目標均值取代*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_023_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859660/Day_023_HW.ipynb?t=1582798284498)
>* [Day_023_Mean_Encoder.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859660/Day_023_Mean_Encoder.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_023_HW.ipynb)

>範例解答:
>* [Day_023_Mean_Encoder_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859660/Day_023_Mean_Encoder_Ans.ipynb?t=1582798284498)
---
## [D24 : 類別型特徵 - 其他進階處理](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859661/1582798284535/__PDF__?t=1582798284498 "D24")
###### *類別型特徵的其他常見編碼 : 計數編碼對應出現頻率相關的特徵，雜湊編碼對應眾多類別而無法排序的特徵*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_024_CountEncoder_and_FeatureHash.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859661/Day_024_CountEncoder_and_FeatureHash.ipynb?t=1582798284498)
>* [Day_024_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859661/Day_024_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_024_HW.ipynb)

>範例解答:
>* [Day_024_CountEncoder_and_FeatureHash_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859661/Day_024_CountEncoder_and_FeatureHash_Ans.ipynb?t=1582798284498)
---
## [D25 : 時間型特徵](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859662/1582798284536/__PDF__?t=1582798284498 "D25")
###### *時間型特徵可抽取出多個子特徵，或周期化，或取出連續時段內的次數*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_025_DayTime_Features.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859662/Day_025_DayTime_Features.ipynb?t=1582798284498)
>* [Day_025_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859662/Day_025_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_025_HW.ipynb)

>範例解答:
>* [Day_025_DayTime_Features_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859662/Day_025_DayTime_Features_Ans.ipynb?t=1582798284498)
---
## [D26 : 特徵組合 - 數值與數值組合](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859663/1582798284537/__PDF__?t=1582798284498 "D26")
###### *特徵組合的基礎 : 以四則運算的各種方式，組合成更具預測力的特徵*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_026_Feature_Combination.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859663/Day_026_Feature_Combination.ipynb?t=1582798284498)
>* [Day_026_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859663/Day_026_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_026_HW.ipynb)

>範例解答:
>* [Day_026_Feature_Combination_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859663/Day_026_Feature_Combination_Ans.ipynb?t=1582798284498)
---
## [D27 : 特徵組合 - 類別與數值組合](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859664/1582798284538/__PDF__?t=1582798284498 "D27")
###### *類別型對數值型特徵可以做群聚編碼，與目標均值編碼類似，但用途不同*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_027_GroupBy_Encoder.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859664/Day_027_GroupBy_Encoder.ipynb?t=1582798284498)
>* [Day_027_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859664/Day_027_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_027_HW.ipynb)

>範例解答:
>* [Day_027_GroupBy_Encoder_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859664/Day_027_GroupBy_Encoder_Ans.ipynb?t=1582798284498)
---
## [D28 : 特徵選擇](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859665/1582798284539/__PDF__?t=1582798284498 "D28")
###### *介紹常見的幾種特徵篩選方式*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_028_Feature_Selection.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859665/Day_028_Feature_Selection.ipynb?t=1582798284498)
>* [Day_028_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859665/Day_028_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_028_HW.ipynb)

>範例解答:
>* [Day_028_Feature_Selection_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859665/Day_028_Feature_Selection_Ans.ipynb?t=1582798284498)
---
## [D29 : 特徵評估](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859666/1582798284540/__PDF__?t=1582798284498 "D29")
###### *介紹並比較兩種重要的特徵評估方式，協助檢測特徵的重要性*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_029_Feature_Importance.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859666/Day_029_Feature_Importance.ipynb?t=1582798284498)
>* [Day_029_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859666/Day_029_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_029_HW.ipynb)

>範例解答:
>* [Day_029_Feature_Importance_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859666/Day_029_Feature_Importance_Ans.ipynb?t=1582798284498)
---
## [D30 : 分類型特徵優化 - 葉編碼](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859667/1582798284541/__PDF__?t=1582798284498 "D30")
###### *葉編碼 : 適用於分類問題的樹狀預估模型改良*
>Data:
>* [house_test.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_test.csv.gz?t=1582795810907)
>* [house_train.csv.gz](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/house_train.csv.gz?t=1582795811951)
>* [taxi_data1.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data1.csv?t=1582795813338)
>* [taxi_data2.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/taxi_data2.csv?t=1582795814535)
>* [titanic_test.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_test.csv?t=1582795815962)
>* [titanic_train.csv](https://ai100-fileentity.cupoy.com/ml100/homework/data/1582795766653/titanic_train.csv?t=1582795819472)

>作業/範例:
>* [Day_030_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859667/Day_030_HW.ipynb?t=1582798284498)
>* [Day_030_Leaf_Encoding.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859667/Day_030_Leaf_Encoding.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_030_HW.ipynb)

>範例解答:
>* [Day_030_Leaf_Encoding_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859667/Day_030_Leaf_Encoding_Ans.ipynb?t=1582798284498)
---
---
# **機器學習基礎模型建立**
##### *學習透過Scikit-learn等套件，建立機器學習模型並進行訓練！*
## [D31 : 機器學習概論](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859668/1582798284542/__PDF__?t=1582798284498 "D31")
###### *機器學習、深度學習與人工智慧差別是甚麼? 機器學習又有甚麼主題應用?*
>作業/範例:
>* [Day_031_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859668/Day_031_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_031_HW.ipynb)

>範例解答:
>* [Day_031_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859668/Day_031_Ans.ipynb?t=1582798284498)
---
## [D32 : 機器學習-流程與步驟](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859669/1582798284543/__PDF__?t=1582798284498 "D32")
###### *資料前處理 -> 訓練/測試集切分 ->選定目標與評估基準 ->  建立模型 -> 調整參數。熟悉整個 ML 的流程*
>作業/範例:
>* [Day_032_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859669/Day_032_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_032_HW.ipynb)

>範例解答:
>* [Day_032_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859669/Day_032_Ans.ipynb?t=1582798284498)
---
## [D33 : 機器如何學習?](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859670/1582798284544/__PDF__?t=1582798284498 "D33")
###### *了解機器學習的定義，過擬合 (Overfit) 是甚麼，該如何解決*
>作業/範例:
>* [Day_033_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859670/Day_033_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_033_HW.ipynb)

>範例解答:
>* [Day_033_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859670/Day_033_Ans.ipynb?t=1582798284498)
---
## [D34 : 訓練/測試集切分的概念](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859671/1582798284545/__PDF__?t=1582798284498 "D34")
###### *為何要做訓練/測試集切分？有什麼切分的方法？*
>作業/範例:
>* [Day_034_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859671/Day_034_HW.ipynb?t=1582798284498)
>* [Day_034_train_test_split.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859671/Day_034_train_test_split.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_034_HW.ipynb)

>範例解答:
>* [Day_034_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859671/Day_034_Ans.ipynb?t=1582798284498)
---
## [D35 : regression vs. classification](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859672/1582798284546/__PDF__?t=1582798284498 "D35")
###### *回歸問題與分類問題的區別？如何定義專案的目標*
>作業/範例:
>* [Day_035_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859672/Day_035_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_035_HW.ipynb)

>範例解答:
>* [Day_035_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859672/Day_035_Ans.ipynb?t=1582798284498)
---
## [D36 : 評估指標選定/evaluation metrics](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859673/1582798284547/__PDF__?t=1582798284498 "D36")
###### *專案該如何選擇評估指標？常用指標有哪些？*
>作業/範例:
>* [Day_036_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859673/Day_036_HW.ipynb?t=1582798284498)
>* [Day_036_evaluation_metrics.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859673/Day_036_evaluation_metrics.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_036_HW.ipynb)

>範例解答:
>* [Day_036_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859673/Day_036_Ans.ipynb?t=1582798284498)
---
## [D37 : regression model 介紹 - 線性迴歸/羅吉斯回歸](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859674/1582798284548/__PDF__?t=1582798284498 "D37")
###### *線性迴歸/羅吉斯回歸模型的理論基礎與使用時的注意事項*
>作業/範例:
>* [Day_037_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859674/Day_037_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_037_HW.ipynb)

>範例解答:
>* [Day_037_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859674/Day_037_Ans.ipynb?t=1582798284498)
---
## [D38 : regression model 程式碼撰寫](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859675/1582798284549/__PDF__?t=1582798284498 "D38")
###### *如何使用 Scikit-learn 撰寫線性迴歸/羅吉斯回歸模型的程式碼*
>作業/範例:
>* [Day_038_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859675/Day_038_HW.ipynb?t=1582798284498)
>* [Day_038_regression_model.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859675/Day_038_regression_model.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_038_HW.ipynb)

>範例解答:
>* [Day_038_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859675/Day_038_Ans.ipynb?t=1582798284498)
---
## [D39 : regression model 介紹 - LASSO 回歸/ Ridge 回歸](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859676/1582798284550/__PDF__?t=1582798284498 "D39")
###### *LASSO 回歸/ Ridge 回歸的理論基礎與與使用時的注意事項*
>作業/範例:
>* [Day_039_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859676/Day_039_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_039_HW.ipynb)

>範例解答:
>* [Day_039_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859676/Day_039_Ans.ipynb?t=1582798284498)
---
## [D40 : regression model 程式碼撰寫](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859677/1582798284551/__PDF__?t=1582798284498 "D40")
###### *使用 Scikit-learn 撰寫 LASSO 回歸/ Ridge 回歸模型的程式碼*
>作業/範例:
>* [Day_040_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859677/Day_040_HW.ipynb?t=1582798284498)
>* [Day_040_lasso_ridge_regression.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859677/Day_040_lasso_ridge_regression.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_040_HW.ipynb)

>範例解答:
>* [Day_040_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859677/Day_040_Ans.ipynb?t=1582798284498)
---
## [D41 : tree based model - 決策樹 (Decision Tree) 模型介紹](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859678/1582798284552/__PDF__?t=1582798284498 "D41")
###### *決策樹 (Decision Tree) 模型的理論基礎與使用時的注意事項*
>作業/範例:
>* [Day_041_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859678/Day_041_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_041_HW.ipynb)

>範例解答:
>* [Day_041_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859678/Day_041_Ans.ipynb?t=1582798284498)
---
## [D42 : tree based model - 決策樹程式碼撰寫](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859679/1582798284553/__PDF__?t=1582798284498 "D42")
###### *使用 Scikit-learn 撰寫決策樹 (Decision Tree) 模型的程式碼*
>作業/範例:
>* [Day_042_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859679/Day_042_HW.ipynb?t=1582798284498)
>* [Day_042_decision_tree.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859679/Day_042_decision_tree.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_042_HW.ipynb)

>範例解答:
>* [Day_042_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859679/Day_042_Ans.ipynb?t=1582798284498)
---
## [D43 : tree based model - 隨機森林 (Random Forest) 介紹](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859680/1582798284554/__PDF__?t=1582798284498 "D43")
###### *隨機森林 (Random Forest)模型的理論基礎與使用時的注意事項*
>作業/範例:
>* [Day_043_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859680/Day_043_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_043_HW.ipynb)

>範例解答:
>* [Day_043_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859680/Day_043_Ans.ipynb?t=1582798284498)
---
## [D44 : tree based model - 隨機森林程式碼撰寫](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859681/1582798284555/__PDF__?t=1582798284498 "D44")
###### *使用 Scikit-learn 撰寫隨機森林 (Random Forest) 模型的程式碼*
>作業/範例:
>* [Day_044_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859681/Day_044_HW.ipynb?t=1582798284498)
>* [Day_044_random_forest.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859681/Day_044_random_forest.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_044_HW.ipynb)

>範例解答:
>* [Day_044_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859681/Day_044_Ans.ipynb?t=1582798284498)
---
## [D45 : tree based model - 梯度提升機 (Gradient Boosting Machine) 介紹](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859682/1582798284556/__PDF__?t=1582798284498 "D45")
###### *梯度提升機 (Gradient Boosting Machine) 模型的理論基礎與使用時的注意事項*
---
## [D46 : tree based model - 梯度提升機程式碼撰寫](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859683/1582798284557/__PDF__?t=1582798284498 "D46")
###### *使用 Scikit-learn 撰寫梯度提升機 (Gradient Boosting Machine) 模型的程式碼*
>作業/範例:
>* [Day_046_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859683/Day_046_HW.ipynb?t=1582798284498)
>* [Day_046_gradient_boosting_machine.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859683/Day_046_gradient_boosting_machine.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_046_HW.ipynb)

>範例解答:
>* [Day_046_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859683/Day_046_Ans.ipynb?t=1582798284498)
---
---
# **機器學習調整參數**
##### *了解模型內的參數意義，學習如何根據模型訓練情形來調整參數*
## [D47 : 超參數調整與優化](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859684/1582798284558/__PDF__?t=1582798284498 "D47")
###### *什麼是超參數 (Hyper-paramter) ? 如何正確的調整超參數？常用的調參方法為何？*
>作業/範例:
>* [Day_047_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859684/Day_047_HW.ipynb?t=1596089038126)
>* [Day_047_hyper_parameter_tunning.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859684/Day_047_hyper_parameter_tunning.ipynb?t=1596089038311)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_047_HW.ipynb)

>範例解答:
>* [Day_047_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859684/Day_047_Ans.ipynb?t=1596089038848)
---
## [D48 : Kaggle 競賽平台介紹](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859685/1582798284559/__PDF__?t=1582798284498 "D48")
###### *介紹全球最大的資料科學競賽網站。如何參加競賽？*
>作業/範例:
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_048_Kaggle%20Data%20Science%20London%20%2B%20Scikit-learn.jpg)
---
## [D49 : 集成方法 : 混合泛化(Blending)](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859686/1582798284560/__PDF__?t=1582798284498 "D49")
###### *什麼是集成? 集成方法有哪些? Blending 的寫作方法與效果為何?*
>作業/範例:
>* [Day_049_Blending.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859686/Day_049_Blending.ipynb?t=1582798284498)
>* [Day_049_Blending_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859686/Day_049_Blending_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_049_Blending.JPG)

>範例解答:
>* [Day_049_Blending_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859686/Day_049_Blending_Ans.ipynb?t=1582798284498)
---
## [D50 : 集成方法 : 堆疊泛化(Stacking)](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859687/1582798284561/__PDF__?t=1582798284498 "D50")
###### *Stacking 的設計方向與主要用途是什麼? 通常會使用什麼套件實作?*
>作業/範例:
>* [Day_050_Stacking.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859687/Day_050_Stacking.ipynb?t=1582798284498)
>* [Day_050_Stacking_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859687/Day_050_Stacking_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_050_Stacking.JPG)

>範例解答:
>* [Day_050_Stacking_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859687/Day_050_Stacking_Ans.ipynb?t=1582798284498)
---
---
# **Kaggle第一次期中考**
## [D51 : Kaggle期中考](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859688/1582798284562/__PDF__?t=1582798284498 "D51")
###### *優惠券使用預測*
>作業/範例:
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_051-053_Midterm%20exam.jpg)
---
---
# **非監督式機器學習**
##### *利用分群與降維方法探索資料模式*
## [D54 : clustering 1 非監督式機器學習簡介](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859689/1582798284563/__PDF__?t=1582798284498 "D54")
###### *非監督式學習簡介、應用場景*
>作業/範例:
>* [Day_054_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859689/Day_054_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_054_HW.ipynb)

>範例解答:
>* [Day_054_Clustering_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859689/Day_054_Clustering_Ans.ipynb?t=1582798284498)
---
## [D55 : clustering 2 聚類算法](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859690/1582798284564/__PDF__?t=1582798284498 "D55")
###### *K-means*
>作業/範例:
>* [Day_055_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859690/Day_055_HW.ipynb?t=1582798284498)
>* [Day_055_kmean_sample.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859690/Day_055_kmean_sample.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_055_HW.ipynb)

>範例解答:
>* [Day_055_kmean_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859690/Day_055_kmean_Ans.ipynb?t=1582798284498)
---
## [D56 : K-mean 觀察 : 使用輪廓分析](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859691/1582798284565/__PDF__?t=1582798284498 "D56")
###### *非監督模型要以特殊評估方法(而非評估函數)來衡量,  今日介紹大家了解並使用其中一種方法 : 輪廓分析*
>作業/範例:
>* [Day_056_kmean.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859691/Day_056_kmean.ipynb?t=1582798284498)
>* [Day_056_kmean_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859691/Day_056_kmean_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_056_kmean_HW.ipynb)

>範例解答:
>* [Day_056_kmean_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859691/Day_056_kmean_Ans.ipynb?t=1582798284498)
---
## [D57 : clustering 3 階層分群算法](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859692/1582798284566/__PDF__?t=1582798284498 "D57")
###### *hierarchical clustering*
>作業/範例:
>* [Day_057_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859692/Day_057_HW.ipynb?t=1582798284498)
>* [Day_057_hierarchical_clustering_sample.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859692/Day_057_hierarchical_clustering_sample.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_057_HW.ipynb)

>範例解答:
>* [Day_057_hierarchical_clustering_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859692/Day_057_hierarchical_clustering_Ans.ipynb?t=1582798284498)
---
## [D58 : 階層分群法 觀察 : 使用 2D 樣版資料集](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859693/1582798284567/__PDF__?t=1582798284498 "D58")
###### *非監督評估方法 : 2D樣版資料集是什麼? 如何生成與使用?*
>作業/範例:
>* [Day_058_hierarchical_clustering.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859693/Day_058_hierarchical_clustering.ipynb?t=1582798284498)
>* [Day_058_hierarchical_clustering_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859693/Day_058_hierarchical_clustering_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_058_hierarchical_clustering_HW.ipynb)

>範例解答:
>* [Day_058_hierarchical_clustering_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859693/Day_058_hierarchical_clustering_Ans.ipynb?t=1582798284498)
---
## [D59 : dimension reduction 1 降維方法-主成份分析](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859694/1589441319698/__PDF__?t=1589447404198 "D59")
###### *PCA*
>作業/範例:
>* [Day_059_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859694/Day_059_HW.ipynb?t=1582798284498)
>* [Day_059_PCA_sample.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859694/Day_059_PCA_sample.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_059_HW.ipynb)

>範例解答:
>* [Day_059_PCA_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859694/Day_059_PCA_Ans.ipynb?t=1582798284498)
---
## [D60 : PCA 觀察 : 使用手寫辨識資料集](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859695/1582798284569/__PDF__?t=1582798284498 "D60")
###### *以較複雜的範例 : sklearn版手寫辨識資料集, 展示PCA的降維與資料解釋能力*
>作業/範例:
>* [Day_060_PCA.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859695/Day_060_PCA.ipynb?t=1582798284498)
>* [Day_060_PCA_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859695/Day_060_PCA_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_060_PCA_HW.ipynb)

>範例解答:
>* [Day_060_PCA_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859695/Day_060_PCA_Ans.ipynb?t=1582798284498)
---
## [D61 : dimension reduction 2 降維方法-T-SNE](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859696/1589441319692/__PDF__?t=1589441381759 "D61")
###### *TSNE*
>作業/範例:
>* [Day_061_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859696/Day_061_HW.ipynb?t=1582798284498)
>* [Day_061_tsne_sample.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859696/Day_061_tsne_sample.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_061_HW.ipynb)

>範例解答:
>* [Day_061_tsne_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859696/Day_061_tsne_Ans.ipynb?t=1582798284498)
---
## [D62 : t-sne 觀察 : 分群與流形還原](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859697/1582798284571/__PDF__?t=1582798284498 "D62")
###### *什麼是流形還原? 除了 t-sne 之外還有那些常見的流形還原方法?*
>作業/範例:
>* [Day_062_tsne.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859697/Day_062_tsne.ipynb?t=1582798284498)
>* [Day_062_tsne_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859697/Day_062_tsne_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_062_tsne_HW.ipynb)

>範例解答:
>* [Day_062_tsne_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859697/Day_062_tsne_Ans.ipynb?t=1582798284498)
---
---
# **深度學習理論與實作**
##### *神經網路的運用*
## [D63 : 神經網路介紹](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859698/1589441319708/__PDF__?t=1589798104744 "D63")
###### *Neural Network 簡介*
>作業/範例:
>* [Day_063_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859698/Day_063_HW.ipynb?t=1582798284498)
>* [深度學習補充教材.pdf](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859698/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E8%A3%9C%E5%85%85%E6%95%99%E6%9D%90.pdf?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_063_HW.ipynb)

>範例解答:
>* [Day_063_Intro_of_DNN_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859698/Day_063_Intro_of_DNN_Ans.ipynb?t=1582798284498)
---
## [D64 : 深度學習體驗 : 模型調整與學習曲線](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859699/1582798284573/__PDF__?t=1582798284498 "D64")
###### *介紹體驗平台 TensorFlow PlayGround，並初步了解模型的調整*
>作業/範例:
>* [Day_064_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859699/Day_064_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_064_HW.ipynb)

>範例解答:
>* [Day_064_Experience_of_DNN1_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859699/Day_064_Experience_of_DNN1_Ans.ipynb?t=1582798284498)
>* [TF_result1.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859699/TF_result1.jpg?t=1582798284498)
---
## [D65 : 深度學習體驗 : 啟動函數與正規化](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859700/1582798284574/__PDF__?t=1582798284498 "D65")
###### *在 TF PlayGround 上，體驗進階版的深度學習參數調整*
>作業/範例:
>* [Day_065_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859700/Day_065_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_065_HW.ipynb)

>範例解答:
>* [Day_065_Experience_of_DNN2_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859700/Day_065_Experience_of_DNN2_Ans.ipynb?t=1582798284498)
>* [TF_result2.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859700/TF_result2.jpg?t=1582798284498)
---
---
# **初探深度學習使用Keras**
##### *學習機器學習(ML)與深度學習( DL) 的好幫手*
## [D66 : Keras 安裝與介紹](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859701/1582798284575/__PDF__?t=1582798284498 "D66")
###### *如何安裝 Keras 套件*
>作業/範例:
>* [Day66-Keras_Introduction.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/Day66-Keras_Introduction.ipynb?t=1582798284498)
>* [Day66-Keras_Introduction_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/Day66-Keras_Introduction_HW.ipynb?t=1582798284498)
>* [Day66-Win10 安裝 TensorFlow-gpu _ Keras.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/Day66-Win10%20%E5%AE%89%E8%A3%9D%20TensorFlow-gpu%20_%20Keras.ipynb?t=1596453261167)
>* [Day66-Win10_InstallTensorFlow_gpu&Keras.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/Day66-Win10_InstallTensorFlow_gpu%26Keras.ipynb?t=1582798284498)
>* [LINUX_CPU.png](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/LINUX_CPU.png?t=1582798284498)
>* [Linux_GPU.png](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/Linux_GPU.png?t=1582798284498)
>* [Step2.png](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/Step2.png?t=1582798284498)
>* [Step3.png](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/Step3.png?t=1582798284498)
>* [Step5.png](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/Step5.png?t=1582798284498)
>* [Step6.png](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/Step6.png?t=1582798284498)
>* [Windows10 CPU_GPU.png](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/Windows10%20CPU_GPU.png?t=1582798284498)
>* [macGPU.png](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/macGPU.png?t=1582798284498)
>* [macOS CPU.png](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859701/macOS%20CPU.png?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_066_Keras_Introduction_HW.ipynb)

>範例解答:
>* [Day66-Keras_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859701/Day66-Keras_Ans.ipynb?t=1582798284498)
---
## [D67 : Keras Dataset](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859702/1582798284576/__PDF__?t=1582798284498 "D67")
###### *Keras embedded dataset的介紹與應用*
>作業/範例:
>* [Day67-Keras_Dataset_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859702/Day67-Keras_Dataset_HW.ipynb?t=1582798284498)
>* [Day67-Keras_Dataset_Introduce.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859702/Day67-Keras_Dataset_Introduce.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_067_Keras_Dataset_HW.ipynb)

>範例解答:
>* [Day67-Keras_Dataset_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859702/Day67-Keras_Dataset_Ans.ipynb?t=1582798284498)
---
## [D68 : Keras Sequential API](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859703/1582798284577/__PDF__?t=1582798284498 "D68")
###### *序列模型搭建網路*
>作業/範例:
>* [Day68-Keras_Sequential_Model.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859703/Day68-Keras_Sequential_Model.ipynb?t=1582798284498)
>* [Day68-Keras_Sequential_Model_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859703/Day68-Keras_Sequential_Model_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_068-Keras_Sequential_Model_HW.ipynb)

>範例解答:
>* [Day68-Keras_Sequential_Model_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859703/Day68-Keras_Sequential_Model_Ans.ipynb?t=1582798284498)
---
## [D69 : Keras Module API](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859704/1582798284578/__PDF__?t=1582798284498 "D69")
###### *Keras Module API的介紹與應用*
>作業/範例:
>* [Day69-keras_Module_API.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859704/Day69-keras_Module_API.ipynb?t=1582798284498)
>* [Day69-keras_Module_API_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859704/Day69-keras_Module_API_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_069-keras_Module_API_HW.ipynb)

>範例解答:
>* [Day69-keras_Module_API_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859704/Day69-keras_Module_API_Ans.ipynb?t=1582798284498)
---
## [D70 : Multi-layer Perception多層感知](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859705/1582798284579/__PDF__?t=1582798284498 "D70")
###### *MLP簡介*
>作業/範例:
>* [Day70-Keras_Mnist_MLP_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859705/Day70-Keras_Mnist_MLP_HW.ipynb?t=1582798284498)
>* [Day70-Keras_Mnist_MLP_Sample.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859705/Day70-Keras_Mnist_MLP_Sample.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_070-Keras_Mnist_MLP_HW.ipynb)

>範例解答:
>* [Day70-Keras_Mnist_MLP_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859705/Day70-Keras_Mnist_MLP_Ans.ipynb?t=1582798284498)
---
## [D71 : 損失函數](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859706/1582798284580/__PDF__?t=1582798284498 "D71")
###### *損失函數的介紹與應用*
>作業/範例:
>* [Day71-使用損失函數.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859706/Day71-%E4%BD%BF%E7%94%A8%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8.ipynb?t=1582798284498)
>* [Day71-使用損失函數_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859706/Day71-%E4%BD%BF%E7%94%A8%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_071-loss_function_HW.ipynb)

>範例解答:
>* [Day71-使用損失函數_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859706/Day71-%E4%BD%BF%E7%94%A8%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8_Ans.ipynb?t=1582798284498)
---
## [D72 : 啟動函數](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859707/1582798284581/__PDF__?t=1582798284498 "D72")
###### *啟動函數的介紹與應用*
>作業/範例:
>* [Day72-Activation_function.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859707/Day72-Activation_function.ipynb?t=1582798284498)
>* [Day72-Activation_function_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859707/Day72-Activation_function_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_072-Activation_function_HW.ipynb)

>範例解答:
>* [Day72-Activation_function_Ans .ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859707/Day72-Activation_function_Ans%20.ipynb?t=1582798284498)
---
## [D73 : 梯度下降Gradient Descent](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859708/1582798284582/__PDF__?t=1582798284498 "D73")
###### *梯度下降Gradient Descent簡介*
>作業/範例:
>* [Day73_Gradient Descent.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859708/Day73_Gradient%20Descent.ipynb?t=1582798284498)
>* [Day73_Gradient_Descent_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859708/Day73_Gradient_Descent_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_073_Gradient_Descent_HW.ipynb)

>範例解答:
>* [Day73-Gradient_Descent_Ans.ipynb.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859708/Day73-Gradient_Descent_Ans.ipynb.ipynb?t=1582798284498)
---
## [D74 : Gradient Descent 數學原理](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859709/1582798284583/__PDF__?t=1582798284498 "D74")
###### *介紹梯度下降的基礎數學原理*
>作業/範例:
>* [Day74-Gradient Descent_Math.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859709/Day74-Gradient%20Descent_Math.ipynb?t=1582798284498)
>* [Day74-Gradient Descent_數學式說明.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859709/Day74-Gradient%20Descent_%E6%95%B8%E5%AD%B8%E5%BC%8F%E8%AA%AA%E6%98%8E.ipynb?t=1582798284498)
>* [Day74-Gradient_Descent_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859709/Day74-Gradient_Descent_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_074-Gradient_Descent_HW.ipynb)

>範例解答:
>* [Day74-Gradient_Descent_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859709/Day74-Gradient_Descent_Ans.ipynb?t=1582798284498)
---
## [D75 : BackPropagation](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859710/1582798284584/__PDF__?t=1582798284498 "D75")
###### *反向式傳播簡介*
>作業/範例:
>* [Day75-Back_Propagation.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859710/Day75-Back_Propagation.ipynb?t=1582798284498)
>* [Day75-Back_Propagation_Advanced.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859710/Day75-Back_Propagation_Advanced.ipynb?t=1582798284498)
>* [Day75-Back_Propagation_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859710/Day75-Back_Propagation_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_075-Back_Propagation_HW.ipynb)

>範例解答:
>* [Day75-Back_Propagation_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859710/Day75-Back_Propagation_Ans.ipynb?t=1582798284498)
---
## [D76 : 優化器optimizers](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859711/1582798284585/__PDF__?t=1582798284498 "D76")
###### *優化器optimizers簡介*
>作業/範例:
>* [D76-Optimizers_進階.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859711/D76-Optimizers_%E9%80%B2%E9%9A%8E.ipynb?t=1582798284498)
>* [D76-optimizer_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859711/D76-optimizer_HW.ipynb?t=1582798284498)
>* [D76-optimizer_example.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859711/D76-optimizer_example.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_076-Optimizer_HW.ipynb)

>範例解答:
>* [D76-optimizer_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859711/D76-optimizer_Ans.ipynb?t=1582798284498)
---
## [D77 : 訓練神經網路的細節與技巧 - Validation and overfit](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859712/1582798284586/__PDF__?t=1582798284498 "D77")
###### *檢視並了解 overfit 現象*
>作業/範例:
>* [Day077_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859712/Day077_HW.ipynb?t=1582798284498)
>* [Day077_overfitting.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859712/Day077_overfitting.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_077_HW.ipynb)

>範例解答:
>* [Day077_overfitting_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859712/Day077_overfitting_Ans.ipynb?t=1582798284498)
---
## [D78 : 訓練神經網路前的注意事項](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859713/1582798284587/__PDF__?t=1582798284498 "D78")
###### *資料是否經過妥善的處理？運算資源為何？超參數的設置是否正確？*
>作業/範例:
>* [Day078_CheckBeforeTrain.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859713/Day078_CheckBeforeTrain.ipynb?t=1582798284498)
>* [Day078_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859713/Day078_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_078_HW.ipynb)

>範例解答:
>* [Day078_CheckBeforeTrain_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859713/Day078_CheckBeforeTrain_Ans.ipynb?t=1582798284498)
---
## [D79 : 訓練神經網路的細節與技巧 - Learning rate effect](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859714/1582798284588/__PDF__?t=1582798284498 "D79")
###### *比較不同 Learning rate 對訓練過程及結果的差異*
>作業/範例:
>* [Day079_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859714/Day079_HW.ipynb?t=1582798284498)
>* [Day079_LearningRateEffect.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859714/Day079_LearningRateEffect.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_079_HW.ipynb)

>範例解答:
>* [Day079_LearningRateEffect_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859714/Day079_LearningRateEffect_Ans.ipynb?t=1582798284498)
---
## [D80 : [練習 Day] 優化器與學習率的組合與比較](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859715/1582798284589/__PDF__?t=1582798284498 "D80")
###### *練習時間：搭配不同的優化器與學習率進行神經網路訓練*
>作業/範例:
>* [Day080_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859715/Day080_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_080_HW.ipynb)

>範例解答:
>* [Day080_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859715/Day080_Ans.ipynb?t=1582798284498)
---
## [D81 : 訓練神經網路的細節與技巧 - Regularization](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859716/1582798284590/__PDF__?t=1582798284498 "D81")
###### *因應 overfit 的方法概述 - 正規化 (Regularization)*
>作業/範例:
>* [Day081_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859716/Day081_HW.ipynb?t=1582798284498)
>* [Day081_Regulization.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859716/Day081_Regulization.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_081_HW.ipynb)

>範例解答:
>* [Day081_Regulization_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859716/Day081_Regulization_Ans.ipynb?t=1582798284498)
---
## [D82 : 訓練神經網路的細節與技巧 - Dropout](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859717/1582798284591/__PDF__?t=1582798284498 "D82")
###### *因應 overfit 的方法概述 - 隨機缺失 (Dropout)*
>作業/範例:
>* [Day082_Dropout.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859717/Day082_Dropout.ipynb?t=1582798284498)
>* [Day082_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859717/Day082_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_082_HW.ipynb)

>範例解答:
>* [Day082_Dropout_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859717/Day082_Dropout_Ans.ipynb?t=1582798284498)
---
## [D83 : 訓練神經網路的細節與技巧 - Batch normalization](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859718/1582798284592/__PDF__?t=1582798284498 "D83")
###### *因應 overfit 的方法概述 - 批次正規化 (Batch Normalization)*
>作業/範例:
>* [Day083_BatchNorm.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859718/Day083_BatchNorm.ipynb?t=1582798284498)
>* [Day083_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859718/Day083_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_083_HW.ipynb)

>範例解答:
>* [Day083_BatchNorm_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859718/Day083_BatchNorm_Ans.ipynb?t=1582798284498)
---
## [D84 : [練習 Day] 正規化/機移除/批次標準化的 組合與比較](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859719/1582798284593/__PDF__?t=1582798284498 "D84")
###### *練習時間：Hyper-parameters 大雜燴*
>作業/範例:
>* [Day084_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859719/Day084_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_084_HW.ipynb)

>範例解答:
>* [Day084_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859719/Day084_Ans.ipynb?t=1582798284498)
---
## [D85 : 訓練神經網路的細節與技巧 - 使用 callbacks 函數做 earlystop](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859720/1582798284594/__PDF__?t=1582798284498 "D85")
###### *因應 overfit 的方法概述 - 悔不當初的煞車機制 (EarlyStopping)*
>作業/範例:
>* [Day085_CB_EarlyStop.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859720/Day085_CB_EarlyStop.ipynb?t=1582798284498)
>* [Day085_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859720/Day085_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_085_HW.ipynb)

>範例解答:
>* [Day085_CB_EarlyStop_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859720/Day085_CB_EarlyStop_Ans.ipynb?t=1582798284498)
---
## [D86 : 訓練神經網路的細節與技巧 - 使用 callbacks 函數儲存 model](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859721/1582798284595/__PDF__?t=1582798284498 "D86")
###### *使用 Keras 內建的 callback 函數儲存訓練完的模型*
>作業/範例:
>* [Day086_CB_ModelCheckPoint.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859721/Day086_CB_ModelCheckPoint.ipynb?t=1582798284498)
>* [Day086_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859721/Day086_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_086_HW.ipynb)

>範例解答:
>* [Day086_CB_ModelCheckPoint_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859721/Day086_CB_ModelCheckPoint_Ans.ipynb?t=1582798284498)
---
## [D87 : 訓練神經網路的細節與技巧 - 使用 callbacks 函數做 reduce learning rate](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859722/1582798284596/__PDF__?t=1582798284498 "D87")
###### *使用 Keras 內建的 callback 函數做學習率遞減*
>作業/範例:
>* [Day087HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859722/Day087HW.ipynb?t=1582798284498)
>* [Day087_CB_ReduceLR.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859722/Day087_CB_ReduceLR.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_087_HW.ipynb)

>範例解答:
>* [Day087_CB_ReduceLR_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859722/Day087_CB_ReduceLR_Ans.ipynb?t=1582798284498)
---
## [D88 : 訓練神經網路的細節與技巧 - 撰寫自己的 callbacks 函數](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859723/1582798284597/__PDF__?t=1582798284498 "D88")
###### **
>作業/範例:
>* [Day088_CB_CustomizedCallbacks.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859723/Day088_CB_CustomizedCallbacks.ipynb?t=1582798284498)
>* [Day088_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859723/Day088_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_088_HW.ipynb)

>範例解答:
>* [Day088_CB_CustomizedCallbacks_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859723/Day088_CB_CustomizedCallbacks_Ans.ipynb?t=1582798284498)
---
## [D89 : 訓練神經網路的細節與技巧 - 撰寫自己的 Loss function](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859724/1582798284598/__PDF__?t=1582798284498 "D89")
###### *瞭解如何撰寫客製化的損失函數，並用在模型訓練上*
>作業/範例:
>* [Day089_CustomizedLoss.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859724/Day089_CustomizedLoss.ipynb?t=1582798284498)
>* [Day089_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859724/Day089_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_089_HW.ipynb)

>範例解答:
>* [Day089_CustomizedLoss_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859724/Day089_CustomizedLoss_Ans.ipynb?t=1582798284498)
---
## [D90 : 使用傳統電腦視覺與機器學習進行影像辨識](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859725/1582798284599/__PDF__?t=1582798284498 "D90")
###### *了解在神經網路發展前，如何使用傳統機器學習演算法處理影像辨識*
>作業/範例:
>* [Day090_color_histogram.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859725/Day090_color_histogram.ipynb?t=1582798284498)
>* [Day090_color_histogram_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859725/Day090_color_histogram_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_090_color_histogram_HW.ipynb)

>範例解答:
>* [Day090_color_histogram_ Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859725/Day090_color_histogram_%20Ans.ipynb?t=1582798284498)
---
## [D91 : [練習 Day] 使用傳統電腦視覺與機器學習進行影像辨識](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859726/1582798284600/__PDF__?t=1582798284498 "D91")
###### *應用傳統電腦視覺方法＋機器學習進行 CIFAR-10 分類*
>作業/範例:
>* [Day091_classification_with_cv.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859726/Day091_classification_with_cv.ipynb?t=1582798284498)
>* [Day091_classification_with_cv_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859726/Day091_classification_with_cv_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_091_classification_with_cv_HW.ipynb)

>範例解答:
>* [Day091_classification_with_cv_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859726/Day091_classification_with_cv_Ans.ipynb?t=1582798284498)
---
---
# **深度學習應用卷積神經網路**
##### *卷積神經網路(CNN)常用於影像辨識的各種應用，譬如醫療影像與晶片瑕疵檢測*
## [D92 : 卷積神經網路 (Convolution Neural Network, CNN) 簡介](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859727/1582798284601/__PDF__?t=1582798284498 "D92")
###### *了解CNN的重要性, 以及CNN的組成結構*
>作業/範例:
>* [Day092_CNN_theory.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859727/Day092_CNN_theory.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_092_CNN_theory_HW.ipynb)

>範例解答:
>* [Day092_CNN_theory_solution.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859727/Day092_CNN_theory_solution.ipynb?t=1582798284498)
---
## [D93 : 卷積神經網路架構細節](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859728/1582798284602/__PDF__?t=1582798284498 "D93")
###### *為什麼比DNN更適合處理影像問題, 以及Keras上如何實作CNN*
>作業/範例:
>* [Day93-CNN_Brief.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859728/Day93-CNN_Brief.ipynb?t=1582798284498)
>* [Day93-CNN_Brief_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859728/Day93-CNN_Brief_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_093_CNN_Brief_HW.ipynb)

>範例解答:
>* [Day93-CNN_Brief_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859728/Day93-CNN_Brief_Ans.ipynb?t=1582798284498)
---
## [D94 : 卷積神經網路 - 卷積(Convolution)層與參數調整](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859729/1582798284603/__PDF__?t=1582798284498 "D94")
###### *卷積層原理與參數說明*
>作業/範例:
>* [Day94-CNN_Convolution .ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859729/Day94-CNN_Convolution%20.ipynb?t=1582798284498)
>* [Day94-CNN_Convolution_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859729/Day94-CNN_Convolution_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_094_CNN_Convolution_HW.ipynb)

>範例解答:
>* [Day94-CNN_convolution_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859729/Day94-CNN_convolution_Ans.ipynb?t=1582798284498)
---
## [D95 : 卷積神經網路 - 池化(Pooling)層與參數調整](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859730/1582798284604/__PDF__?t=1582798284498 "D95")
###### *池化層原理與參數說明*
>作業/範例:
>* [Day95-CNN_Pooling_Padding.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859730/Day95-CNN_Pooling_Padding.ipynb?t=1582798284498)
>* [Day95-CNN_Pooling_Padding_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859730/Day95-CNN_Pooling_Padding_HW.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_095_CNN_Pooling_Padding_HW.ipynb)

>範例解答:
>* [Day95-CNN_Pooling_Padding_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859730/Day95-CNN_Pooling_Padding_Ans.ipynb?t=1582798284498)
---
## [D96 : Keras 中的 CNN layers](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859731/1582798284605/__PDF__?t=1582798284498 "D96")
###### *介紹 Keras 中常用的 CNN layers*
>作業/範例:
>* [Day096_Keras_CNN_layers.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859731/Day096_Keras_CNN_layers.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_096_Keras_CNN_layers_HW.ipynb)

>範例解答:
>* [Day096_Keras_CNN_layers_solution.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859731/Day096_Keras_CNN_layers_solution.ipynb?t=1582798284498)
---
## [D97 : 使用 CNN 完成 CIFAR-10 資料集](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859732/1582798284606/__PDF__?t=1582798284498 "D97")
###### *透過 CNN 訓練 CIFAR-10 並比較其與 DNN 的差異*
>作業/範例:
>* [Day097_Keras_CNN_vs_DNN.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859732/Day097_Keras_CNN_vs_DNN.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_097_Keras_CNN_vs_DNN_HW.ipynb)

>範例解答:
>* [Day097_Keras_CNN_vs_DNN_solution.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859732/Day097_Keras_CNN_vs_DNN_solution.ipynb?t=1582798284498)
---
## [D98 : 訓練卷積神經網路的細節與技巧 - 處理大量數據](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859733/1582798284607/__PDF__?t=1582798284498 "D98")
###### *資料無法放進記憶體該如何解決？如何使用 Python 的生成器 generator?*
>作業/範例:
>* [Day098_Python_generator.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859733/Day098_Python_generator.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_098_Python_generator_HW.ipynb)

>範例解答:
>* [Day098_Python_generator_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859733/Day098_Python_generator_Ans.ipynb?t=1582798284498)
---
## [D99 : 訓練卷積神經網路的細節與技巧 - 處理小量數據](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859734/1582798284608/__PDF__?t=1582798284498 "D99")
###### *資料太少準確率不高怎麼辦？如何使用資料增強提升準確率？*
>作業/範例:
>* [Day099_data_augmentation.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859734/Day099_data_augmentation.ipynb?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_099_data_augmentation_HW.ipynb)

>範例解答:
>* [Day099_data_augmentation_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859734/Day099_data_augmentation_Ans.ipynb?t=1582798284498)
---
## [D100 : 訓練卷積神經網路的細節與技巧 - 轉移學習 (Transfer learning)](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859735/1582798284609/__PDF__?t=1582798284498 "D100")
###### *何謂轉移學習 Transfer learning？該如何使用？*
>作業/範例:
>* [Day100_transfer_learning.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859735/Day100_transfer_learning.ipynb?t=1582798284498)
>* [Day100_transfer_learning_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859735/Day100_transfer_learning_HW.ipynb?t=1582798284498)
>* [resnet_builder.py](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859735/resnet_builder.py?t=1582798284498)
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_100_transfer_learning_HW.ipynb)

>範例解答:
>* [Day100_transfer_learning_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/answer/1582797859735/Day100_transfer_learning_Ans.ipynb?t=1582798284498)
---
---
# **Kaggle期末考**
## [D101 : Kaggle期末考](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859736/1582798284610/__PDF__?t=1582798284498 "D101")
###### *透過 CNN 進行貓狗影像分類，結合學習到的深度學習模型應用技巧，完成自己的第一個貓狗分類模型。*
>作業/範例:
>* [我的作業](https://github.com/chouqimin/2nd-ML100Days/blob/master/data/homework/Day_101-103_Final%20exam.jpg)
---
---
# **Bonus進階補充**
##### *電腦視覺實務延伸*
## [D104 : 互動式網頁神經網路視覺化](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859737/1582798284611/__PDF__?t=1582798284498 "D104")
###### *利用Standford互動式網頁, 介紹 parameter 微調（fine-tuning)*
>作業/範例:
>* [Day104-ConvNetJS.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859737/Day104-ConvNetJS.ipynb?t=1582798284498)
>* [Day104-ConvNetJS_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859737/Day104-ConvNetJS_Ans.ipynb?t=1582798284498)
>* [Day104-ConvNetJS_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859737/Day104-ConvNetJS_HW.ipynb?t=1582798284498)
---
## [D105 : CNN卷積網路回顧](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859738/1582798284612/__PDF__?t=1582798284498 "D105")
###### *了解卷積網路的有趣應用*
>作業/範例:
>* [Day105-CNN_Ans.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859738/Day105-CNN_Ans.ipynb?t=1582798284498)
>* [Day105-CNN_HW.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859738/Day105-CNN_HW.ipynb?t=1582798284498)
>* [Day105-CNN_sample.ipynb](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859738/Day105-CNN_sample.ipynb?t=1582798284498)
>* [elephant.jpg](https://ai100-fileentity.cupoy.com/ml100/homework/example/1582797859738/elephant.jpg?t=1582798284498)
---
## [D106 : 常見影像資料集介紹 (Cifar-10, ImageNet, COCO)](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859739/1582798284613/__PDF__?t=1582798284498 "D106")
###### *沒有自己的資料集嗎？使用影像競賽中常用的影像資料集*
---
## [D107 : 電腦視覺應用介紹 - 影像分類, 影像分割, 物件偵測](http://ai100-fileentity.cupoy.com/ml100/dailytask/1582797859740/1582798284614/__PDF__?t=1582798284498 "D107")
###### *卷積神經網路在實際生活中的應用案例*
---
