# gunicorn.conf.py
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 1
threads = 1
timeout = 30
preload_app = True
loglevel = "debug"
accesslog = "-"
errorlog = "-"
