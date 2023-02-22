## 关于DataFrame的大小问题
* 设置对应的dtype，可以有效减少DataFrame的大小，但是当写入csv文件时，dtype数据将会丢失。
* `df.to_pickle(df.pickle)`pickle文件读写比csv更快，且文件更小，并且会保存所有的dtype信息。
* `df.to_feather(df.feather)`feather文件读写比csv更快，文件更小。更适合于短期储存，而`parquet` (需要install)更适合长期储存。
* size: csv > pickle > feather > parquet
* speed: csv > parquet > feather > pickle