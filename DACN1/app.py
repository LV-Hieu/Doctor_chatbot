from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Tải mô hình và tokenizer
model_name = './health_chatbot_model'  # Đường dẫn đến mô hình đã lưu
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Đảm bảo mô hình ở chế độ đánh giá
model.eval()

# Route để hiển thị trang chủ (giao diện người dùng)
@app.route('/')
def index():
    return render_template('index.html')

# Route để xử lý phản hồi từ chatbot
@app.route('/get_response', methods=['POST'])
def get_bot_response():
    data = request.get_json()  # Nhận dữ liệu từ client
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({"response": "Không có nội dung nhập từ người dùng."})

    try:
        # Tạo phản hồi từ mô hình
        inputs = tokenizer.encode(user_input, return_tensors='pt')  # Mã hóa đầu vào
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1)  # Tạo đầu ra
        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Giải mã đầu ra

        return jsonify({"response": bot_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
