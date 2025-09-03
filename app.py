from flask import Flask, render_template, request
from main import run_prediction

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    data = None
    if request.method == "POST":
        ticker = request.form.get("ticker").upper()
        try:
            today, tomorrow, comparison_df, plot_url = run_prediction(ticker)
            table_html = comparison_df.to_html(
                classes="table table-bordered table-striped text-center",
                index=False,
                justify="center",
                float_format = "%.2f"
            )
            data = {
                "ticker": ticker,
                "today": f"{today:.2f}",
                "tomorrow": f"{tomorrow:.2f}",
                "plot_url": plot_url,
                "table_html": table_html
            }
        except Exception as e:
            data = {"error": str(e)}
    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)
