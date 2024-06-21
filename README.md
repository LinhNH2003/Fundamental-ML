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
### I.Tổng quan về đồ án
- Project nhận dạng biểu cảm khuôn mặt được thực hiện trên các hình ảnh khuôn mặt từ bộ dữ liệu
- Tập dữ liệu được tổng hợp từ internet, được thiết kế để phân loại biểu hiện khuôn mặt. Dữ liệu bao gồm các hình ảnh thang độ xám của khuôn mặt, mỗi hình ảnh có kích thước 48x48 pixel. Các khuôn mặt đã được tự động căn chỉnh để gần như ở giữa và chiếm một khu vực tương tự trong mỗi hình ảnh.
- Mục tiêu là sử dụng những thuật toán Machine Learning và Deep Learning để phân loại từng khuôn mặt dựa trên cảm xúc được thể hiện, gán nó vào một trong bảy loại cảm xúc (0=Tức giận, 1=Ghê tởm, 2=Sợ hãi, 3=Vui vẻ, 4=Buồn, 5=Bất ngờ, 6=Trung lập).
### 1. Prerequisite
#### **1.1 Thông tin về dữ liệu:**
- Kết quả thu được từ data: tập dữ liệu bao gồm 35,887 dòng và 2 cột: emotion và pixels. Cột emotion chứa các giá trị số nguyên đại diện cho các loại cảm xúc, và cột pixels chứa các chuỗi ký tự đại diện cho các giá trị pixel của hình ảnh.
  ![background](./materials/view.png)
- **Các nhãn dữ liệu:**
   + Nhãn Angry:
     ![background](./materials/Angry.png)
   + Nhãn Disgust:
     ![background](./materials/Disgust.png)
   + Nhãn Fear:
      ![background](./materials/Fear.png)
   + Nhãn Happy:
      ![background](./materials/Happy.png)
   + Nhãn Sad:
      ![background](./materials/Sad.png)
   + Nhãn Surprise:
      ![background](./materials/Surpise.png)
   + Nhãn Neutral:
     ![background](./materials/Neutral.png)
#### **1.2 Thực hiện các xử lí ban đầu:**
- Kiểm tra giá trị thiếu và dữ liệu trùng lặp
- Khi đó ta thấy rằng dữ liệu được cung cấp không có giá trị thiếu và có **1793** giá trị trùng lặp => cần drop giá trị trùng lặp.
#### **1.3 Trực quan dữ liệu ban đầu:**
  - Tổng quan phần trăm các nhãn trong dữ liệu:
     ![background](./materials/PB_emotion.png) 
  - Nhận xét về phân phối nhãn dữ liệu
    ![background](./materials/phanphoidulieu.png) 
    - **Train Labels**: ta nhận thấy sự mất cân bằng rõ rệt:
        - Nhãn "Happy" xuất hiện nhiều nhất (~8000).
        - Nhãn "Disgust" xuất hiện ít nhất (gần 0).
        - Nhãn "Fear", "Sad", và "Neutral" trung bình (~5000-7000).
        - Nhãn "Angry" và "Surprise" thấp hơn.
    - **Random Labels** có phân phối đồng đều: mọi nhãn có tần suất xuất hiện tương đương (~5000).
    - **So sánh chung**
        - **Train Labels**: Mất cân bằng giữa các nhãn.
        - **Random Labels**: Phân phối đều đặn.
    => **Kết luận**: Dữ liệu mất cân bằng trong tập huấn luyện bởi vì nhãn "Happy" quá phổ biến có thể gây thiên vị cho mô hình.
### 2. Principal Components Analysis - PCA
- Một trong những ứng dụng phổ biến nhất của biến đổi dữ liệu không giám sát là giảm chiều dữ liệu. Quá trình này giảm số lượng đặc trưng (chiều) trong dữ liệu. Khi dữ liệu có số lượng đặc trưng lớn, việc phân tích có thể tốn kém về mặt tính toán và khó khăn. Các kỹ thuật giảm chiều dữ liệu giúp khắc phục những thách thức này.

- Phân Tích Thành Phần Chính (PCA) là một kỹ thuật phổ biến cho việc giảm chiều dữ liệu. Nó biến đổi dữ liệu thành một tập hợp mới của các đặc trưng gọi là các thành phần chính (PCs). Những PCs này được sắp xếp theo thứ tự quan trọng, nắm bắt các biến thể quan trọng nhất trong dữ liệu. Bằng cách chọn một tập hợp con của những PCs thông tin nhất, chúng ta có thể đạt được sự giảm kích thước dữ liệu đáng kể trong khi vẫn giữ lại thông tin cốt yếu cho việc phân tích.

#### **Câu hỏi 1: Can you visualize the data projected onto two principal components?**
- Hình ảnh khi trực quan dữ liệu về 2 chiều:
![background](./materials/pca.png)
**Nhận xét**: Từ hình ảnh trên có thể thấy chúng ta không thể trực quan dữ liệu trên không gian hai chiều(n_componets = 2) vì không mang lại ý nghĩa nào về mặt trực quan đối với bộ dữ liệu.

#### **Câu hỏi 2:How to determine the optimal number of principal components using pca.explained_variance_? Explain your selection process.**
- Để xác định số lượng thành phần chính (principal components - PCs) tối ưu, ta cần xem xét tỷ lệ phương sai được giải thích (explained_variance_ratio_) bởi các PCs. Mục tiêu là chọn số lượng PCs sao cho giữ lại được phần lớn phương sai trong dữ liệu ban đầu mà không cần sử dụng quá nhiều PCs, giúp giảm chiều dữ liệu và cải thiện hiệu quả tính toán.
- Các bước thực hiện:
      - 1. **Khởi tạo PCA và tính toán:** Khởi tạo đối tượng PCA và áp dụng nó lên dữ liệu để tính toán các PCs.
      - 2. **Tính toán phương sai tích lũy**: Sử dụng ```np.cumsum(pca.explained_variance_ratio_)``` để tính toán tỷ lệ phương sai tích lũy được giải thích bởi các PCs.
      - 3. **Vẽ Scree Plot**: Tạo biểu đồ Scree Plot để trực quan hóa tỷ lệ phương sai tích lũy theo số lượng PCs.
      - 4. **Xác định điểm khuỷu (elbow point)**: Sử dụng logic để tìm điểm khuỷu, là điểm mà sau đó tỷ lệ tăng của phương sai giải thích giảm đi đáng kể. Đây thường là dấu hiệu để chọn số lượng PCs tối ưu.

  ![background](./materials/ellow.png)
  **Nhận xét**: Từ hình ảnh có thể thấy component tối ưu được chọn là **104**. Điều này có nghĩa là 104 PCs đầu tiên giải thích được 90% phương sai của dữ liệu ban đầu.
### Model 









