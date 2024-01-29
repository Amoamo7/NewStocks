# ライブラリのインポート
from flask import (
    Flask,
    render_template,
    request,
    session,
)

app = Flask(__name__)  # Flask の起動


from app_scr import function
from app_model import model
from app_ggf_old import data_1, data_2
from flask import Flask, render_template, request


@app.route("/")
def index():
    return render_template("index_1.html")


@app.route("/", methods=["POST"])
def post():
    corpo = request.form.get("q")
    textlist = function(corpo)  # textがlist型で入っている。
    sum = model(textlist)  # 感情分析結果の合計が0.9999116659164429の形で入っている。
    raw_data = data_1(corpo)  # data_valueとしてndarrayが入っている(sum合計前)
    data_2(raw_data, sum)  # 合計した後、画像を保存

    return render_template(
        "result.html",
    )


if __name__ == "__main__":
    app.run(debug=True)
