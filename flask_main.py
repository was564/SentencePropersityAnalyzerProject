from flask import Flask, request, make_response, jsonify, render_template
from ai_main import predict_sentiment

# Initialize Flask app
app = Flask(__name__)

# Define a route for your web page
@app.route("/")
def index():
    return render_template("chat_js.html")  # Render HTML template

@app.route("/chat", methods=["POST"])
def chat():
    message = request.get_json().get("message")
    probability = predict_sentiment(message)
    if probability > 0.66:
        response = "긍정적인 답변입니다."
        response += f" (확률: {probability:.2f})"
    elif probability < 0.33:
        response = "부정적인 답변입니다."
        response += f" (확률: {1 - probability:.2f})"
    else:
        response = "모호한 답변입니다."

    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app in debug mode
