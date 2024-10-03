# preprocessing_ocr

## Overview

File test_full.py thực hiện đầy  luồng Preprocessing, bao gồm:

* Đọc ảnh
* Resize (nếu cần)
* Phân loại ảnh: Card, administrative_document, face
* Nếu ảnh thuộc loại Card: thực hiện crop các card và xoay (nếu cần) trong ảnh đó
* Lưu ảnh vào các folder trong thư mục results

## Folders

* image_test: ảnh test
* pyiqa: source code đánh giá ảnh
* Đoạn code sẽ tự tạo folder _file_storage_ nếu chưa có

## link models: 

* https://drive.google.com/file/d/18hJnmtMUhnCm8mwAGhxSUrxJLmfpl_Ho/view?usp=sharing
* https://drive.google.com/file/d/1HvqeA_qZQZQjtj1cVx_tPo1yhY-0S45y/view?usp=sharing
* https://drive.google.com/file/d/1vURfqstglaQwilebJ94D6J-JvQLW6rK0/view?usp=sharing