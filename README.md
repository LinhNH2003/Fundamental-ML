# Mini-Project for Fundamentals of Machine Learning Course
![background](./materials/ai_wp.jpg)
This repository contains the code and data for a mini-project on facial expression recognition using machine learning algorithms.

## 📑 Project Policy
- Team: group should consist of 3-4 students.

    |No.| Student Name    | Student ID |
    | --------| -------- | ------- |
    |1|Nguyễn Hoài Linh|21280097|
    |2|Trần Trịnh Mai Vy|21280122|
    |3|Trần Thị Bích Tuyền|21280059|
    |4|Nguyễn Thị Yến Như|21280082|

- The submission deadline is strict: **11:59 PM** on **June 22nd, 2024**. Commits pushed after this deadline will not be considered.

## 📦 Project Structure

The repository is organized into the following directories:

- **/data**: This directory contains the facial expression dataset. You'll need to download the dataset and place it here before running the notebooks. (Download link provided below)
- **/notebooks**: This directory contains the Jupyter notebook ```EDA.ipynb```. This notebook guides you through exploratory data analysis (EDA) and classification tasks.

## ⚙️ Usage

This project is designed to be completed in the following steps:

1. **Fork the Project**: Click on the ```Fork``` button on the top right corner of this repository, this will create a copy of the repository in your own GitHub account. Complete the table at the top by entering your team member names.

2. **Download the Dataset**: Download the facial expression dataset from the following [link](https://mega.nz/file/foM2wDaa#GPGyspdUB2WV-fATL-ZvYj3i4FqgbVKyct413gxg3rE) and place it in the **/data** directory:

3. **Complete the Tasks**: Open the ```notebooks/EDA.ipynb``` notebook in your Jupyter Notebook environment. The notebook is designed to guide you through various tasks, including:
    
    1. Prerequisite
    3. Principle Component Analysis
    4. Image Classification
    5. Evaluating Classification Performance 

    Make sure to run all the code cells in the ```EDA.ipynb``` notebook and ensure they produce output before committing and pushing your changes.

5. **Commit and Push Your Changes**: Once you've completed the tasks outlined in the notebook, commit your changes to your local repository and push them to your forked repository on GitHub.

## MINI PROJECT
### Tổng quan về đồ án
- Project nhận dạng biểu cảm khuôn mặt được thực hiện trên các hình ảnh khuôn mặt từ bộ dữ liệu
- Tập dữ liệu được tổng hợp từ internet, được thiết kế để phân loại biểu hiện khuôn mặt. Dữ liệu bao gồm các hình ảnh thang độ xám của khuôn mặt, mỗi hình ảnh có kích thước 48x48 pixel. Các khuôn mặt đã được tự động căn chỉnh để gần như ở giữa và chiếm một khu vực tương tự trong mỗi hình ảnh.
- Mục tiêu là sử dụng những thuật toán Machine Learning và Deep Learning để phân loại từng khuôn mặt dựa trên cảm xúc được thể hiện, gán nó vào một trong bảy loại cảm xúc (0=Tức giận, 1=Ghê tởm, 2=Sợ hãi, 3=Vui vẻ, 4=Buồn, 5=Bất ngờ, 6=Trung lập)

### 1.Prerequisite
- Thông tin về dữ liệu:
- Kết quả thu được từ data: tập dữ liệu bao gồm 35,887 dòng và 2 cột: emotion và pixels. Cột emotion chứa các giá trị số nguyên đại diện cho các loại cảm xúc, và cột pixels chứa các chuỗi ký tự đại diện cho các giá trị pixel của hình ảnh.
- Thực hiện các xử lí ban đầu:
  - 1.1 Kiểm tra giá trị thiếu và dữ liệu trùng lặp.
    Khi đó ta thấy rằng dữ liệu được cung cấp không có giá trị thiếu và có **1793** giá trị trùng lặp
    => cần drop giá trị trùng lặp





