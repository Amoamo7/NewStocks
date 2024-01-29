# 以下のコードをすべて関数化
def data_1(corpo_line):
    # ライブラリのインポート
    from time import sleep
    import time
    import requests
    from selenium.webdriver.common.keys import Keys
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from bs4 import BeautifulSoup
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    from flask import Flask, render_template, request

    # Google Finance を開く
    # ChromeDriverの実行可能ファイルへの絶対パスを指定
    try:
        chrome_driver_path = "C:\\Users\\81808\\OneDrive\\デスクトップ\\My pands\\chromedriver-win32\\chromedriver.exe"
    except:
        pass
    # ChromeDriverのサービスオブジェクトを作成
    chrome_service = webdriver.chrome.service.Service(chrome_driver_path)

    # Google Financeのトップページを表示
    options = Options()
    options.add_argument("--ignore-certificate-errors")
    driver = webdriver.Chrome(options=options)
    driver.get("https://www.google.com/finance/?hl=ja")
    time.sleep(1)

    # 検索窓に"～～"を入力して銘柄のコードを取得する
    sele__1 = "#yDmH0d > c-wiz.zQTmif.SSPGKf.ccEnac > div > div.KdK6Xc > div.e1AOyf > div > div > div > div.d1dlne > input.Ax4B8.ZAGvjd"
    search_code_1 = driver.find_element(By.CSS_SELECTOR, sele__1)

    search_code_1.send_keys(corpo_line)
    search_code_1.send_keys(Keys.RETURN)
    time.sleep(1)
    # 検索完了

    # 銘柄コードを取得

    cur_url = driver.current_url

    # BeautifulSoupでHTMLを解析する

    url = cur_url
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "html.parser")
    text_code_market = soup.find(class_="PdOqHc").get_text()
    # 株の銘柄を取得
    print(text_code_market[3:7])
    com_code = text_code_market[3:7]

    # 市場名を取得
    print(text_code_market[10:])
    market_name = text_code_market[10:]

    # 以下、application_gspから引用
    # import
    import gspread
    import googlefinance
    import gspread
    import json
    import pandas as pd

    # ServiceAccountCredentials：Googleの各サービスへアクセスできるservice変数を生成します。
    from oauth2client.service_account import ServiceAccountCredentials

    # 2つのAPIを記述しないとリフレッシュトークンを3600秒毎に発行し続けなければならない
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    # 認証情報設定
    # ダウンロードしたjsonファイル名をクレデンシャル変数に設定（秘密鍵、Pythonファイルから読み込みしやすい位置に置く）
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        "my-project-scr-403611-993e42646ecb.json", scope
    )

    # OAuth2の資格情報を使用してGoogle APIにログインします。
    gc = gspread.authorize(credentials)

    # 共有設定したスプレッドシートキーを変数[SPREADSHEET_KEY]に格納する。
    SPREADSHEET_KEY = "1ZPMtkZPOQIBgPAL3E9hfNu-i2S3KZSWoiATBI7-Orh0"

    # 共有設定したスプレッドシートのシート1を開く
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1

    # 入力する企業のコードを設定
    code_1 = market_name + ":" + com_code
    "NASDAQ:MSFT"

    # 自動入力する関数を設定

    formula = '=GOOGLEFINANCE("' + code_1 + '","price","2014/01/01",TODAY(),"DAILY")'
    # formula = '=GOOGLEFINANCE("NASDAQ:MSFT","price","2020/11/30","2020/12/30","DAILY")'
    # セルに関数を設定
    cell_range = "A1"  # 関数を入力したいセルの範囲を指定

    # value_input_option パラメータを使用して関数を評価モードに設定
    value_input_option = "USER_ENTERED"

    worksheet.update(cell_range, [[formula]], value_input_option=value_input_option)

    # スプレッドシートの値をデータフレーム型で取得
    # A2からB22のデータを取得
    data = worksheet.get_all_values()

    # データをDataFrameに変換
    df_data = pd.DataFrame(data[1:], columns=data[0])

    df_date_data = df_data.iloc[1:, 0]
    df_close_data = df_data.iloc[1:, 1]

    # df_date_dataの/を-に置換する
    df_date_data = df_date_data.str.strip("16:00:00")
    df_date_data = df_date_data.str.strip("13:00:00")
    df_date_data_replace = df_date_data.str.replace("/", "-")

    # データフレームを結合
    df_2dim = pd.concat([df_date_data_replace, df_close_data], axis=1)
    data_value = df_close_data.values
    # 返す
    return data_value


def data_2(data_value, sum_to_add):
    # 取得したデータの標準化
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # data_valueにsumを加算してdata_valueを再定義
    data_value_new = data_value.astype(float)  # strからfloatに変換→sumを加算できるようにする
    data_value_new[-3:] += sum_to_add
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(data_value_new.reshape(-1, 1))

    # normalized_valuesは標準化した株価のデータ

    # LSTMのモデル構築
    # 過去10回分を学習データとして作成、
    maxlen = 10
    x_data = []
    y_data_price = []
    for i in range(len(normalized_values) - maxlen):
        x_data.append(normalized_values[i : i + maxlen])
        y_data_price.append(normalized_values[i + maxlen])

    # 訓練データ・テストデータの作成
    x_data = np.asarray(x_data)
    y_data_price = np.asarray(y_data_price)

    # 全データセットの内、80% を訓練データ、残り20% をテストデータ
    train_size = int(x_data.shape[0] * 0.8)

    # 訓練データ、train_sizeまでをスライスを用いて取り出し、訓練データとする
    x_train = x_data[:train_size]
    y_train_price = y_data_price[:train_size]

    # 残りのデータをテストデータとしている
    x_test = x_data[train_size:]
    y_test_price = y_data_price[train_size:]

    # ライブラリのインポート
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.layers import Activation
    from tensorflow.keras import metrics
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import matplotlib
    import shutil

    matplotlib.use("Agg")  # <--ここを追加
    import matplotlib.pyplot as plt

    from datetime import datetime, timedelta
    import pandas as pd

    early_stopping = EarlyStopping(
        monitor="val_mean_absolute_error", mode="auto", patience=7
    )

    out_neurons = 1
    units = 300
    model = Sequential()
    model.add(LSTM(units, batch_input_shape=(None, maxlen, 1), return_sequences=False))
    model.add(Dense(out_neurons))

    # モデルをコンパイルし学習に関する設定
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=[metrics.mae])

    # 構築したモデルの学習、histに代入することで、モデルの学習過程を保存する
    hist = model.fit(
        x_train,
        y_train_price,
        batch_size=60,
        epochs=15,
        validation_split=0.1,
        callbacks=[early_stopping],
    )

    # 学習過程が保存されているhistを見ていきます
    """ loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    epochs = len(loss)
    plt.plot(range(epochs), loss, label="loss(training data)")
    plt.plot(range(epochs), val_loss, label="val_loss(evaluation data)")
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show() """

    predicted = model.predict(x_test)

    predicted_N = scaler.inverse_transform(predicted)
    y_test_price_N = scaler.inverse_transform(y_test_price)
    """ 
    plt.plot(range(len(predicted)), predicted_N, marker=".", label="predicted")
    plt.plot(range(len(y_test_price)), y_test_price_N, marker=".", label="y_test_price")
    plt.grid()
    plt.xlabel("DATE")
    plt.ylabel("Close price")
    plt.show()

    # 決定係数を求める。
    from sklearn.metrics import r2_score

    r2 = r2_score(predicted_N, y_test_price_N) """

    # 未来の予測点数
    future_points = 10

    # 予測のためのデータを複製
    future_data_points = x_test[-1:].copy()

    # 未来の予測を格納するリスト
    future_predictions = []

    # 未来の予測をまとめて生成
    for _ in range(future_points):
        # 現在の未来データポイントを使用して予測
        future_prediction = model.predict(future_data_points.reshape(1, maxlen, 1))

        # 予測結果をリストに追加
        future_predictions.append(future_prediction[0, 0])

        # 予測結果を次の未来データポイントとして使用するために追加
        future_data_points = np.roll(future_data_points, -1)
        future_data_points[-1] = future_prediction[0, 0]

    # 未来の予測結果を逆変換
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    # 元のテストデータと未来の予測結果を含むプロット
    plt.plot(range(len(y_test_price)), y_test_price_N, marker=".", label="y_test_price")
    plt.plot(range(len(predicted)), predicted_N, marker=".", label="predicted")
    plt.plot(
        range(len(y_test_price), len(y_test_price) + future_points),
        future_predictions,
        marker=".",
        label="future_predictions",
    )
    plt.grid()
    plt.xlabel("DATE")
    plt.ylabel("Close price")
    plt.legend()
    # plt.savefig("result_plot.jpg", dpi=300, bbox_inches="tight")    これが現段階で正しい
    # new_path = shutil.move("result_plot.jpg", "static")

    import os
    import shutil

    save_path = "result_plot.jpg"
    new_path = "static"

    # 保存先パスのファイルが既に存在する場合
    if os.path.exists(os.path.join(new_path, "result_plot.jpg")):
        os.remove(os.path.join(new_path, "result_plot.jpg"))

    # プロットを保存
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # プロットを新しいパスに移動
    shutil.move(save_path, os.path.join(new_path, "result_plot.jpg"))


# corpo = "amazon"
# data_value = data_1(corpo)
# sum = 0.9
# data_2(data_value, sum)
