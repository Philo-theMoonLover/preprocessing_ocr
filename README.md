# preprocessing_ocr

## Overview

File test_full.py thực hiện đầy  luồng Preprocessing, bao gồm:

* Đọc ảnh
* Resize (nếu cần)
* Phân loại ảnh: Card, administrative_document, face
* Nếu ảnh thuộc loại Card: thực hiện crop các card và xoay (nếu cần) trong ảnh đó
* Lưu ảnh vào các folder trong thư mục results

## Folders

* results\results_crop: Lưu các thẻ (card) được phát hiện và crop
* results\gray_image: Lưu các thẻ (card) có dạng grayscale được phát hiện và crop
* results\results_False: Lưu các thẻ (card) được phát hiện và crop nhưng được phân loại sai (không phải card)
* results\administrative_document: Lưu ảnh tài liệu
* results\face: Lưu ảnh face
