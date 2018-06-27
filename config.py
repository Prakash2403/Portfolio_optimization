with open('secrets.txt', 'r') as f:
    keys = f.read()
QUANDL_KEY = eval(keys)['quandl_key']
START_DATE = '2016-06-06'
END_DATE = '2018-06-06'
DATASET_PATH = 'datasets/'
ASSET = 100000
