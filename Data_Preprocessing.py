import os
import re
import torch
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import chardet
from transformers import AutoTokenizer, AutoModel

# Đường dẫn đến thư mục chứa dữ liệu
data_dir = "D:\\Pythontest\\"
input_file = os.path.join(data_dir, 'shopee_reviews.txt')
cleaned_file = os.path.join(data_dir, 'cleaned.txt')
filtered_file = os.path.join(data_dir, 'filtered.txt')
spam_file = os.path.join(data_dir, 'spam.txt')
new_reviews_file = os.path.join(data_dir, 'new_reviews.txt')
stopwords_file = os.path.join(data_dir, 'vietnamese-stopwords.txt')

# Xác định encoding của file input và đọc file
def read_file(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    with open(file_path, 'r', encoding=encoding) as f:
        content = f.read()
    return content

# Đọc stopwords từ file
def read_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = [word.strip() for word in f]
    return stopwords


# Load mô hình PhoBERT tiếng Việt
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

# Danh sách từ vựng liên quan đến chất lượng sản phẩm và dịch vụ
quality_keywords = ["ngon", "chất lượng", "tốt", "tuyệt vời", "xuất sắc", "hương vị", "thơm ngon",
                    "giao hàng", "nhanh", "chậm", "đóng gói", "dịch vụ", "chăm sóc", "tư vấn"]

# Bước 1: Làm sạch dữ liệu và lưu vào file mới
def clean_and_save(input_file, output_file):
    content = read_file(input_file)
    # Loại bỏ một số ký tự đặc biệt cơ bản (ngoại trừ dấu phẩy và dấu chấm)
    cleaned_content = re.sub(r'[^\w\s.,áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', '', content)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(cleaned_content)

# Bước 2: Đọc dữ liệu đã làm sạch, tách thành các trường, loại bỏ nhiễu và đánh giá spam
def process_data(cleaned_file):
    data = []
    spam_data = []
    with open(cleaned_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Loại bỏ dòng trống và dòng chỉ chứa số
            if not line.strip() or line.strip().isdigit():
                continue

            parts = line.strip().split(',')
            num_parts = len(parts)

            if num_parts >= 4:  # Đảm bảo có ít nhất 4 trường
                review_words = word_tokenize(parts[3])
                product_name = parts[0]
                category = parts[1]
                rating = parts[2]
            else:
                continue

            if review_words and (len(review_words) > 1 or not review_words[0].isdigit()):
                review_vector_bert = model(**tokenizer(review_words, padding=True, truncation=True, return_tensors="pt"))[0][0].mean(dim=0)
                product_vector_bert = model(**tokenizer(product_name, padding=True, truncation=True, return_tensors="pt"))[0][0].mean(dim=0)
                similarity_bert = torch.cosine_similarity(review_vector_bert, product_vector_bert, dim=0)

                if similarity_bert > 0.2:
                    data.append({
                        'product': product_name,
                        'category': category,
                        'rating': rating,
                        'review': parts[num_parts-1]
                    })
                else:
                    spam_data.append({
                        'product': product_name,
                        'category': category,
                        'rating': rating,
                        'review': parts[num_parts-1]
                    })

    keywords = extract_keywords(data, top_n=20)
    return data, spam_data, keywords

# Trích xuất từ khóa từ dữ liệu
def extract_keywords(data, top_n=20):
    stop_words = read_stopwords(stopwords_file)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform([d['review'] for d in data])

    feature_names = tfidf_vectorizer.get_feature_names_out()
    keywords = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i, :].toarray()
        top_n_indices = row.argsort()[0, -top_n:]
        keywords_this_review = [feature_names[j] for j in top_n_indices]

        # Tính độ tương đồng với tên sản phẩm và lọc theo quality_keywords
        product_name = data[i]['product']
        filtered_keywords = [
            kw for kw in keywords_this_review
            if kw in quality_keywords and calculate_similarity(kw, product_name) > 0.3
        ]
        keywords.extend(filtered_keywords)

    return list(set(keywords))

def calculate_similarity(text1, text2):
    with torch.no_grad():
        inputs1 = tokenizer(text1, padding=True, truncation=True, return_tensors="pt")
        inputs2 = tokenizer(text2, padding=True, truncation=True, return_tensors="pt")
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

        # Lấy vector nhúng [CLS]
        embeddings1 = outputs1.last_hidden_state[:, 0, :]  
        embeddings2 = outputs2.last_hidden_state[:, 0, :]  

        # Tính cosine similarity
        similarity = torch.cosine_similarity(embeddings1, embeddings2)

    return similarity.item()  # Trả về giá trị scalar từ tensor

# Bước 3: Chuẩn bị dữ liệu cho mô hình NLP và huấn luyện
def train_model(data):
    X = [' '.join(word_tokenize(d['review'])) for d in data]
    y = [1] * len(X) # Gán nhãn 1 cho tất cả các đánh giá (đã được lọc)

    model = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )
    model.fit(X, y)
    return model

# Đọc dữ liệu từ file đánh giá mới
def read_reviews_from_file(file_path):
    reviews = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                reviews.append(line.strip())
    except OSError as e:
        print(f"Lỗi khi mở file: {e}")
    return reviews

# Dự đoán đánh giá cho các bài đánh giá mới
def predict_ratings(model, reviews):
    predictions = []
    for review in reviews:
        processed_review = ' '.join(word_tokenize(review))
        predicted_rating = model.predict([processed_review])[0]
        predictions.append(predicted_rating)
    return predictions

# Hàm ghi dữ liệu đã lọc vào file
def save_filtered_data(data, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            for item in data:
                file.write(f"{item['product']},{item['category']},{item['rating']},{item['review']}\n") # Ghi tất cả các trường
    except OSError as e:
        print(f"Lỗi khi ghi file: {e}")

# Hàm ghi keywords vào file
def save_keywords(keywords, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            for keyword in keywords:
                file.write(f"{keyword}\n")
    except OSError as e:
        print(f"Lỗi khi ghi file: {e}")

# Xử lý dữ liệu và trích xuất từ khóa
def preprocess_data(input_file, cleaned_file, filtered_file, spam_file, top_n=20):
    clean_and_save(input_file, cleaned_file)
    filtered_data, spam_data, keywords = process_data(cleaned_file)
    save_filtered_data(filtered_data, filtered_file)
    save_filtered_data(spam_data, spam_file)
    return keywords

# --- Chạy chương trình ---
clean_and_save(input_file, cleaned_file)
filtered_data, spam_data, keywords = process_data(cleaned_file)  # Chỉ gọi process_data một lần
print("Các từ khóa được trích xuất:", keywords)
save_filtered_data(filtered_data, filtered_file) # Lưu dữ liệu đã lọc
save_filtered_data(spam_data, spam_file) # Lưu dữ liệu spam
model = train_model(filtered_data) # Huấn luyện mô hình trên dữ liệu đã lọc

# Lưu keywords vào file
keyword_file = os.path.join(data_dir, 'keyword.txt')
save_keywords(keywords, keyword_file)

# --- Dự đoán cho file đánh giá mới ---
new_reviews = read_reviews_from_file(new_reviews_file)
predicted_ratings = predict_ratings(model, new_reviews)

for review, rating in zip(new_reviews, predicted_ratings):
    print(f"Bài đánh giá: {review}")
    print(f"Dự đoán đánh giá: {rating} sao")
    print("----")