# gunicorn.conf.py
bind = "0.0.0.0:10000"
workers = 1
threads = 1
timeout = 30
preload_app = True
loglevel = "debug"
accesslog = "-"
errorlog = "-"
