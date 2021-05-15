# Stock price predictor

This repo contains a brief Machine Learning project, where we tried to predict
the IBM stock price through learning models.

## Step by step to run the code

- Add your Quandl API private key as an environmental variable to your local
  environment, name it as `QUANDL_API_KEY`
- Create virtual env: `virtualenv -p python3 ibm-stock`
- Activate virtual env: `source ibm-stock/bin/activate`
- Install requirements: `pip3 install -r requirements.txt`
- Run `main.py`: `python3 main.py`

`main.py` will create a menu where you will choose which model do you want to
evaluate at the given time.

The next instructions will be shown directly on the terminal.
