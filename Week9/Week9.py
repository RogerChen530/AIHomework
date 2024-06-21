import openai

# 請將 'your-api-key' 替換為你自己的API密鑰
openai.api_key = 'your-api-key'

def ask_openai(question):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=question,
      max_tokens=150
    )
    return response.choices[0].text.strip()

def main():
    print("歡迎使用OpenAI問答應用！")
    while True:
        question = input("請輸入你的問題（或輸入 '退出' 來結束）： ")
        if question.lower() == '退出':
            break
        answer = ask_openai(question)
        print("回答：", answer)

if __name__ == "__main__":
    main()
