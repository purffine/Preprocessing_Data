import os
import re
import torch
from underthesea import word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import chardet
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pyvi import ViTokenizer
from sklearn.svm import LinearSVC
from sentence_transformers import SentenceTransformer, util

# Đường dẫn đến thư mục chứa dữ liệu
data_dir = "D:\\Pythontest\\"
input_file = os.path.join(data_dir, 'shopee_reviews.txt')
cleaned_file = os.path.join(data_dir, 'cleaned.txt')
filtered_file = os.path.join(data_dir, 'filtered.txt')
spam_file = os.path.join(data_dir, 'spam.txt')
new_reviews_file = os.path.join(data_dir, 'new_reviews.txt')
stopwords_file = os.path.join(data_dir, 'vietnamese-stopwords.txt')
trained_svm_file = os.path.join(data_dir, 'trained.txt')  # File dữ liệu huấn luyện SVM

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
quality_keywords = [
    "ngon", "chất lượng", "tốt", "tuyệt vời", "xuất sắc", "hương vị", "thơm ngon",
    "giao hàng", "nhanh", "chậm", "đóng gói", "dịch vụ", "chăm sóc", "tư vấn",
    "bền", "đẹp", "chắc chắn", "sắc nét", "rõ ràng", "tinh tế", "sắc sảo", "tinh xảo",
    "hiện đại", "sang trọng", "cao cấp", "chất liệu tốt", "giao hàng nhanh", "phục vụ tốt",
    "tư vấn nhiệt tình", "thân thiện", "chu đáo", "chuyên nghiệp", "uy tín", "dễ sử dụng",
    "tiện lợi", "hiệu quả", "hài lòng", "hạn sử dụng", "chắc chắn", "cẩn thận", "gói hàng"
]

# Hàm kiểm tra xem bài đánh giá có chứa từ khóa liên quan đến chất lượng/dịch vụ hay không
def contains_quality_keywords(review_text):
    review_words = word_tokenize(review_text.lower())
    return any(keyword in review_words for keyword in quality_keywords)

# Hàm phân loại spam dựa trên mô hình học máy (SVM)
def classify_spam_svm(review_text, model, vectorizer):
    review_vectorized = vectorizer.transform([review_text])
    prediction = model.predict(review_vectorized)[0]
    return prediction  # 1: spam, 0: không spam

# Khởi tạo Sentence Transformer
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Hàm tính toán độ tương đồng ngữ nghĩa
def calculate_semantic_similarity(text1, text2):
    embeddings1 = sentence_model.encode(text1, convert_to_tensor=True)
    embeddings2 = sentence_model.encode(text2, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_similarity.item()

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
                category = parts[1] # Lấy tên phân loại sản phẩm
                rating = parts[2]
            else:
                continue

            if review_words and (len(review_words) > 1 or not review_words[0].isdigit()):
                review_vector_bert = model(**tokenizer(review_words, padding=True, truncation=True, return_tensors="pt"))[0][
                    0].mean(dim=0)
                product_vector_bert = model(
                    **tokenizer(product_name, padding=True, truncation=True, return_tensors="pt"))[0][0].mean(dim=0)
                similarity_bert = torch.cosine_similarity(review_vector_bert, product_vector_bert, dim=0)

                # Tính toán độ tương đồng ngữ nghĩa
                semantic_similarity = calculate_semantic_similarity(', '.join(parts[3:]), product_name)

                # Kiểm tra spam (sử dụng gộp điểm)
                spam_score = 0

                if contains_quality_keywords(parts[3]):
                    spam_score += 2

                if similarity_bert >= 0.1:  # Điều chỉnh ngưỡng độ tương đồng
                    spam_score += 1
                
                # Thêm kiểm tra liên quan đến tên phân loại sản phẩm
                if calculate_similarity(parts[3], category) > 0.3: 
                    spam_score += 2  # Tăng điểm nếu liên quan đến phân loại

                # Phân tích cú pháp với underthesea
                pos_tags = pos_tag(parts[3])
                num_verbs = sum([1 for word, tag in pos_tags if tag.startswith('V')])
                num_nouns = sum([1 for word, tag in pos_tags if tag.startswith('N')])

                if num_verbs >= 2 and num_nouns >= 3:  # Điều chỉnh ngưỡng phân tích cú pháp
                    spam_score += 1

                # 1. Tính toán độ tương đồng cosine với Sentence Embedding
                product_embedding = sentence_model.encode(product_name, convert_to_tensor=True)
                review_embedding = sentence_model.encode(parts[3], convert_to_tensor=True)
                similarity_score = util.pytorch_cos_sim(review_embedding, product_embedding).item()

                # 2. Phân tích từ khóa chung
                product_words = word_tokenize(product_name.lower())
                review_words = word_tokenize(parts[3].lower())
                common_words = set(product_words) & set(review_words)
                if similarity_score < 0.2:
                    spam_score -= 2
                elif similarity_score < 0.3 and len(common_words) < 3:
                    spam_score -= 1

                # Kiểm tra tỉ lệ chữ IN HOA (ví dụ: không quá 20%)
                uppercase_ratio = sum(1 for c in parts[3] if c.isupper()) / len(parts[3])
                if uppercase_ratio > 0.2:
                    spam_score -= 1  # Trừ điểm nếu tỉ lệ chữ in hoa quá cao

                if classify_spam_svm(parts[3], svm_model, vectorizer):
                    spam_score -= 3  # Tăng điểm trừ cho SVM

                if semantic_similarity < 0.5:  # Điều chỉnh ngưỡng độ tương đồng ngữ nghĩa
                    spam_score -= 2

                if spam_score < 0:  # Điều chỉnh ngưỡng tổng điểm
                    is_spam = True
                else:
                    is_spam = False

                if not is_spam:
                    data.append({
                        'product': product_name,
                        'category': category,
                        'rating': rating,
                        'review': ', '.join(parts[3:]) # Nối các phần tử từ parts[3] trở đi
                    })
                else:
                    spam_data.append({
                        'product': product_name,
                        'category': category,
                        'rating': rating,
                        'review': ', '.join(parts[3:])  # Nối các phần tử từ parts[3] trở đi
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
    y = [1] * len(X)  # Gán nhãn 1 cho tất cả các đánh giá (đã được lọc)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra (ví dụ: 80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )
    model.fit(X_train, y_train)

    # Dự đoán nhãn trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Đánh giá hiệu suất
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))

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
                file.write(
                    f"{item['product']},{item['category']},{item['rating']},{item['review']}\n")  # Ghi tất cả các trường
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


# --- Chuẩn bị dữ liệu huấn luyện cho mô hình SVM từ file trained.txt ---
train_data = []
with open(trained_svm_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")  # Tách dòng bằng dấu phẩy

        # Kiểm tra xem có đủ 2 phần tử sau khi tách không
        if len(parts) >= 2: 
            text = ",".join(parts[:-1]) # Lấy tất cả phần tử trừ phần tử cuối cùng làm text 
            label = parts[-1]  # Lấy phần tử cuối cùng làm label
            train_data.append({"text": text, "label": int(label)})
        else:
            print(f"Bỏ qua dòng không hợp lệ: {line}")

train_texts = [item["text"] for item in train_data]
train_labels = [item["label"] for item in train_data]

# Huấn luyện mô hình SVM
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
svm_model = LinearSVC()
svm_model.fit(X_train, train_labels)

# --- Khởi tạo mô hình BERT và Sentence Transformer ---
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- Chạy chương trình ---

# --- Xử lý dữ liệu và trích xuất từ khóa ---
clean_and_save(input_file, cleaned_file)
filtered_data, spam_data, keywords = process_data(cleaned_file)
print("Các từ khóa được trích xuất:", keywords)
save_filtered_data(filtered_data, filtered_file)
save_filtered_data(spam_data, spam_file)
keyword_file = os.path.join(data_dir, 'keyword.txt')
save_keywords(keywords, keyword_file)

# --- Huấn luyện mô hình phân loại ---
model = train_model(filtered_data)

# --- Dự đoán cho file đánh giá mới ---
new_reviews = read_reviews_from_file(new_reviews_file)
predicted_ratings = predict_ratings(model, new_reviews)

for review, rating in zip(new_reviews, predicted_ratings):
    print(f"Bài đánh giá: {review}")
    print(f"Dự đoán đánh giá: {rating} sao")
    print("----")
