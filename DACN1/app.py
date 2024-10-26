from flask import Flask, render_template, request, jsonify
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth import is_bfloat16_supported
from transformers import AutoTokenizer, TextStreamer
import torch

app = Flask(__name__)

# Kiểm tra xem CUDA có sẵn không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải mô hình và tokenizer
model_name = './fine_tuned_llama3b'  

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = FastLanguageModel.from_pretrained(model_name)

# Di chuyển mô hình vào thiết bị (CPU hoặc GPU)
model.to(device)

# Đảm bảo mô hình ở chế độ đánh giá
model.eval()

# Route để hiển thị trang chủ (giao diện người dùng)
@app.route('/')
def index():
    return render_template('index.html')

# Format
alpaca_prompt = """Dưới đây là hướng dẫn mô tả một nhiệm vụ. Viết một phản hồi hoàn thành yêu cầu một cách thích hợp.

### Nhiệm vụ:
{}

### Đầu vào:
{}

### Câu trả lời:
{}"""

# Route để xử lý phản hồi từ chatbot
@app.route('/get_response', methods=['POST'])
def get_bot_response():
    data = request.get_json()  # Nhận dữ liệu từ client
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({"response": "Không có nội dung nhập từ người dùng."})

    try:
        # Sử dụng Alpaca prompt để tạo phản hồi từ mô hình
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    "If you are a doctor, please answer the medical questions based on the patient's description.",  # instruction
                    user_input,  # input - dùng biến user_input từ đầu vào của người dùng
                    "",  # output - để trống cho phần tạo văn bản mới
                )
            ], 
            return_tensors="pt"
        ).to(device)

        # Sử dụng TextStreamer để stream kết quả
        text_streamer = TextStreamer(tokenizer)
        outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128*8)

        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Giải mã đầu ra

        return jsonify({"response": bot_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)