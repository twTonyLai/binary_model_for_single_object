# binary_model_for_single_object
由於爬蟲時，部分網頁會將單筆家具圖以及風格布置圖做區分
檢測爬蟲下來的圖片是否屬於'單張'家具圖，將回傳YES
如下：


<img src="https://github.com/yponpon/binary_model_for_single_object/blob/master/pictures/ikea002.528.80(0).jpg" width = 300 alt = '範例1'>
<img src="https://github.com/yponpon/binary_model_for_single_object/blob/master/pictures/ikea003.526.29(0).jpg" width = 300 alt = '範例2'>

若非屬於上述圖片，如風格圖或其他室內圖將回傳NOT:


<img src="https://github.com/yponpon/binary_model_for_single_object/blob/master/pictures/ikea003.526.29(1).jpg" width = 300 alt = '範例3'>
<img src="https://github.com/yponpon/binary_model_for_single_object/blob/master/pictures/ikea003.526.29(4).jpg" width = 300 alt = '範例4'>


check_inception_v3_model.py 
