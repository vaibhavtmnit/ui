uvicorn main:app --reload

python write_data.py --ns team-alpha --n-large 12000

pip install \
  pandas==2.2.2 \
  numpy==1.26.4 \
  pyarrow==17.0.0 \
  fastparquet==2024.5.0 \
  plotly==5.24.1 \
  dash==2.17.1 \
  dash-bootstrap-components==1.6.0 \
  fastapi==0.115.0 \
  uvicorn==0.31.0

  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

http://127.0.0.1:8000/dash/?ns=test
http://127.0.0.1:8000/data/test/df4.csv
http://127.0.0.1:8000/data/test/df4.csv
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
If you see “404 Not Found” on /dash, make sure:

main.py mounts Dash correctly:

app.mount("/dash", WSGIMiddleware(dashboard_ui.server))
