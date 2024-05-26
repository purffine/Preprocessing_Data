# Preprocessing_Data

**Đề tài:** _**Loại bỏ nhiễu từ các đánh giá của Shopee (và phân tích và đánh giá số sao của bình luận sản phẩm trên Shopee)**_

**Mô tả:**
Dự án này tập trung vào việc phân tích và đánh giá cảm xúc của bình luận sản phẩm trên Shopee. Mục tiêu là xác định các bình luận spam, trích xuất từ khóa quan trọng liên quan đến chất lượng sản phẩm/dịch vụ, và xây dựng mô hình để dự đoán đánh giá tích cực/tiêu cực từ bình luận mới.

**Cách thức hoạt động:**

Tiền xử lý dữ liệu (quan trọng): Làm sạch dữ liệu thô (loại bỏ ký tự đặc biệt, dấu câu không cần thiết,...) và chuẩn bị dữ liệu cho quá trình phân tích.
Xử lý dữ liệu và đánh giá spam: Tách bình luận thành các trường thông tin (tên sản phẩm, danh mục, đánh giá,...), lọc bỏ các bình luận nhiễu, và sử dụng mô hình PhoBERT để đánh giá mức độ liên quan giữa bình luận và sản phẩm nhằm xác định spam.
Trích xuất từ khóa: Sử dụng kỹ thuật TF-IDF và bộ từ điển từ khóa liên quan đến chất lượng để xác định các từ khóa quan trọng nhất trong các bình luận.
Huấn luyện mô hình Naive Bayes: Xây dựng mô hình phân loại Naive Bayes dựa trên các bình luận đã được lọc và đánh giá trước đó.
Dự đoán đánh giá (chức năng thêm): Sử dụng mô hình đã huấn luyện để dự đoán đánh giá tích cực/tiêu cực cho các bình luận mới.

**Yêu cầu:**
Python 3.x
Các thư viện: underthesea, sklearn, torch, transformers, chardet
Mô hình PhoBERT tiếng Việt (vinai/phobert-base)

**Cài đặt:**

Clone repository:
git clone https://github.com/your-username/your-repository.git

Cài đặt các thư viện cần thiết:
pip install -r requirements.txt

**Sử dụng:**

Đặt dữ liệu shopee_reviews.txt (các dữ liệu được scrape từ website của trang Shopee cần phân tích) và new_reviews.txt ( vào thư mục D:\Preprocessing\ (hoặc cập nhật đường dẫn data_dir trong code).

Chạy file main.py:
python main.py

**Dữ liệu:**

**shopee_reviews.txt**: Dữ liệu bình luận sản phẩm trên Shopee (định dạng txt).
**new_reviews.txt**: Bình luận mới cần dự đoán đánh giá (mỗi bình luận trên một dòng).
**vietnamese-stopwords.txt**: Danh sách stopwords tiếng Việt.
**keyword.txt**: Các từ khóa quan trọng được trích xuất.
**filtered.txt**: Dữ liệu bình luận đã lọc.
**spam.txt**: Dữ liệu bình luận được xác định là spam.

Ví dụ kết quả:

Các từ khóa được trích xuất: ['chất lượng', 'giao hàng', 'nhanh', ...]
Bài đánh giá: Sản phẩm đẹp, giao hàng nhanh chóng!
Dự đoán đánh giá: 1 sao 
----
